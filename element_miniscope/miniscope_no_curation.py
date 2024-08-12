import csv
import cv2
import importlib
import inspect
import json
import pathlib
from datetime import datetime
from typing import Union

import datajoint as dj
import numpy as np
import pandas as pd
from element_interface.utils import dict_to_uuid, find_full_path, find_root_directory

from . import miniscope_report

logger = dj.logger

schema = dj.Schema()

_linking_module = None


def activate(
    miniscope_schema_name: str,
    *,
    create_schema: bool = True,
    create_tables: bool = True,
    linking_module: str = None,
):
    """Activate this schema.

    Args:
        miniscope_schema_name (str): schema name on the database server
        create_schema (bool): when True (default), create schema in the database if it
            does not yet exist.
        create_tables (str): when True (default), create schema takes in the database
            if they do not yet exist.
        linking_module (str): a module (or name) containing the required dependencies.

    Dependencies:

    Upstream tables:
        Session: parent table to Recording, identifying a recording session.
        Device: Reference table for Recording, specifying the acquisition device.
        AnatomicalLocation: Reference table for anatomical region for recording acquisition.

    Functions:
        get_miniscope_root_data_dir(): Returns absolute path for root data director(y/ies)
            with all subject/sessions data, as (list of) string(s).
        get_session_directory(session_key: dict) Returns the session directory with all
            data for the session in session_key, as a string.
        get_processed_root_data_dir(): Returns absolute path for all processed data as
            a string.
    """

    if isinstance(linking_module, str):
        linking_module = importlib.import_module(linking_module)
    assert inspect.ismodule(
        linking_module
    ), "The argument 'dependency' must be a module's name or a module"

    global _linking_module
    _linking_module = linking_module

    schema.activate(
        miniscope_schema_name,
        create_schema=create_schema,
        create_tables=create_tables,
        add_objects=_linking_module.__dict__,
    )
    miniscope_report.activate(f"{miniscope_schema_name}_report", miniscope_schema_name)


# Functions required by the element-miniscope  -----------------------------------------


def get_miniscope_root_data_dir() -> list:
    """Fetches absolute data path to miniscope data directory.

    The absolute path here is used as a reference for all downstream relative paths used in DataJoint.

    Returns:
        A list of the absolute path to miniscope data directory.
    """

    root_directories = _linking_module.get_miniscope_root_data_dir()
    if isinstance(root_directories, (str, pathlib.Path)):
        root_directories = [root_directories]

    if hasattr(_linking_module, "get_processed_root_data_dir"):
        root_directories.append(_linking_module.get_processed_root_data_dir())

    return root_directories


def get_processed_root_data_dir() -> Union[str, pathlib.Path]:
    """Retrieve the root directory for all processed data.

    All data paths and directories in DataJoint Elements are recommended to be stored as
    relative paths (posix format), with respect to some user-configured "root"
    directory, which varies from machine to machine (e.g. different mounted drive
    locations).

    Returns:
        dir (str| pathlib.Path): Absolute path of the processed miniscope root data
            directory.
    """

    if hasattr(_linking_module, "get_processed_root_data_dir"):
        return _linking_module.get_processed_root_data_dir()
    else:
        return get_miniscope_root_data_dir()[0]


def get_session_directory(session_key: dict) -> str:
    """Pulls session directory information from database.

    Args:
        session_key (dict): a dictionary containing session information.

    Returns:
        Session directory as a string.
    """
    return _linking_module.get_session_directory(session_key)


# Experiment and analysis meta information -------------------------------------


@schema
class AcquisitionSoftware(dj.Lookup):
    """Software used for miniscope acquisition.

    Required to define a miniscope recording.

    Attributes:
        acq_software (str): Name of the miniscope acquisition software."""

    definition = """
    acq_software: varchar(24)
    """
    contents = zip(["Miniscope-DAQ-V3", "Miniscope-DAQ-V4", "Inscopix"])


@schema
class Channel(dj.Lookup):
    """Number of channels in the miniscope recording.

    Attributes:
        channel (tinyint): Number of channels in the miniscope acquisition starting at zero.
    """

    definition = """
    channel     : tinyint  # 0-based indexing
    """
    contents = zip(range(5))


@schema
class Recording(dj.Manual):
    """Recording defined by a measurement done using a scanner and an acquisition software.

    Attributes:
        Session (foreign key): A primary key from Session.
        recording_id (int): Unique recording ID.
        Device (foreign key, optional): A primary key from Device.
        AcquisitionSoftware (foreign key): A primary key from AcquisitionSoftware.
        recording_notes (str, optional): notes about the recording session.
    """

    definition = """
    -> Session
    recording_id: int
    ---
    -> [nullable] Device
    -> AcquisitionSoftware
    recording_notes='' : varchar(4095) # free-notes
    """


@schema
class RecordingLocation(dj.Manual):
    """Brain location where the miniscope recording is acquired.

    Attributes:
        Recording (foreign key): A primary key from Recording.
        AnatomicalLocation (foreign key): A primary key from AnatomicalLocation.
    """

    definition = """
    # Brain location where this miniscope recording is acquired
    -> Recording
    ---
    -> AnatomicalLocation
    """


@schema
class RecordingInfo(dj.Imported):
    """Information about the recording extracted from the recorded files.

    Attributes:
        Recording (foreign key): A primary key from Recording.
        nchannels (tinyint): Number of recording channels.
        nframes (int): Number of recorded frames.
        px_height (smallint): Height in pixels.
        px_width (smallint): Width in pixels.
        um_height (float): Height in microns.
        um_width (float): Width in microns.
        fps (float): Frames per second, (Hz).
        gain (float): Recording gain.
        spatial_downsample (tinyint): Amount of downsampling applied.
        led_power (float): LED power used for the recording.
        recording_datetime (datetime): Datetime of the recording.
        recording_duration (float): Total recording duration (seconds).
    """

    definition = """
    # Store metadata about recording
    -> Recording
    ---
    nchannels            : tinyint   # number of channels
    nframes              : int       # number of recorded frames
    ndepths=1            : tinyint   # number of depths
    px_height            : smallint  # height in pixels
    px_width             : smallint  # width in pixels
    fps                  : float     # (Hz) frames per second
    recording_datetime=null   : datetime  # datetime of the recording
    recording_duration=null   : float     # (seconds) duration of the recording
    """

    class Config(dj.Part):
        """Recording metadata and configuration.

        Attributes:
            Recording (foreign key): A primary key from RecordingInfo.
            config (longblob): Recording metadata and configuration.
        """

        definition = """
        -> master
        ---
        config: longblob  # recording metadata and configuration
        """

    class Timestamps(dj.Part):
        """Recording timestamps for each frame.

        Attributes:
            Recording (foreign key): A primary key from RecordingInfo.
            timestamps (longblob): Recording timestamps for each frame.
        """

        definition = """
        -> master
        ---
        timestamps: longblob
        """

    class File(dj.Part):
        """File path to recording file relative to root data directory.

        Attributes:
            Recording (foreign key): Recording primary key.
            file_id (foreign key, smallint): Unique file ID.
            path_path (varchar(255) ): Relative file path to recording file.
        """

        definition = """
        -> master
        file_id : smallint unsigned
        ---
        file_path: varchar(255)      # relative to root data directory
        """

    def make(self, key):
        """Populate table with recording file metadata."""

        # Search recording directory for miniscope raw files
        acq_software = (Recording & key).fetch1("acq_software")
        recording_directory = get_session_directory(key)

        recording_path = find_full_path(
            get_miniscope_root_data_dir(), recording_directory
        )

        recording_filepaths = (
            [file_path.as_posix() for file_path in recording_path.glob("*.avi")]
            if acq_software != "Inscopix"
            else [file_path.as_posix() for file_path in recording_path.rglob("*.avi")]
        )
        if not recording_filepaths:
            raise FileNotFoundError(f"No .avi files found in " f"{recording_directory}")

        if acq_software == "Miniscope-DAQ-V3":
            recording_timestamps = recording_path / "timestamp.dat"
            if not recording_timestamps.exists():
                raise FileNotFoundError(
                    f"No timestamp file found in " f"{recording_directory}"
                )

            nchannels = 1  # Assumes a single channel

            # Parse number of frames from timestamp.dat file
            with open(recording_timestamps) as f:
                next(f)
                nframes = sum(1 for line in f if int(line[0]) == 0)

            # Parse image dimension and frame rate
            video = cv2.VideoCapture(recording_filepaths[0])
            _, frame = video.read()
            frame_size = np.shape(frame)
            px_height = frame_size[0]
            px_width = frame_size[1]

            fps = video.get(cv2.CAP_PROP_FPS)

        elif acq_software == "Miniscope-DAQ-V4":
            try:
                recording_metadata = next(recording_path.glob("metaData.json"))
            except StopIteration:
                raise FileNotFoundError(
                    f"No .json file found in " f"{recording_directory}"
                )
            try:
                recording_timestamps = next(recording_path.glob("timeStamps.csv"))
            except StopIteration:
                raise FileNotFoundError(
                    f"No timestamp (*.csv) file found in " f"{recording_directory}"
                )

            with open(recording_metadata.as_posix()) as f:
                metadata = json.loads(f.read())

            with open(recording_timestamps, newline="") as f:
                time_stamps = list(csv.reader(f, delimiter=","))

            nchannels = 1  # Assumes a single channel
            nframes = len(time_stamps) - 1
            px_height = metadata["ROI"]["height"]
            px_width = metadata["ROI"]["width"]
            fps = int(metadata["frameRate"].replace("FPS", ""))
            time_stamps = np.array(time_stamps[1:], dtype=float)[:, 0]

        elif acq_software == "Inscopix":
            inscopix_metadata = next(recording_path.glob("session.json"))
            timestamps_file = next(recording_path.glob("*/*timestamps.csv"))
            metadata = json.load(open(inscopix_metadata))
            recording_timestamps = pd.read_csv(timestamps_file)

            nchannels = len(metadata["manual"]["mScope"]["ledMaxPower"])
            nframes = len(recording_timestamps)
            fps = metadata["microscope"]["fps"]["fps"]
            time_stamps = (recording_timestamps[" time (ms)"] / 1000).values
            px_height = metadata["microscope"]["fov"]["height"]
            px_width = metadata["microscope"]["fov"]["width"]

        else:
            raise NotImplementedError(
                f"Loading routine not implemented for {acq_software}"
                " acquisition software"
            )

        # Insert in RecordingInfo
        self.insert1(
            dict(
                key,
                nchannels=nchannels,
                nframes=nframes,
                px_height=px_height,
                px_width=px_width,
                fps=fps,
                recording_duration=nframes / fps,
            )
        )

        # Insert file(s)
        recording_files = [
            pathlib.Path(f)
            .relative_to(find_root_directory(get_miniscope_root_data_dir(), f))
            .as_posix()
            for f in recording_filepaths
        ]

        self.File.insert(
            [
                {**key, "file_id": i, "file_path": f}
                for i, f in enumerate(recording_files)
            ]
        )

        if acq_software in ("Inscopix", "Miniscope-DAQ-V4"):
            self.Timestamps.insert1(dict(**key, timestamps=time_stamps))
            self.Config.insert1(
                dict(
                    **key,
                    config=metadata,
                )
            )


# Trigger a processing routine -------------------------------------------------


@schema
class ProcessingMethod(dj.Lookup):
    """Package used for processing of miniscope data (e.g. CaImAn, etc.).

    Attributes:
        processing_method (str): Processing method.
        processing_method_desc (str): Processing method description.
    """

    definition = """# Package used for processing of calcium imaging data (e.g. Suite2p, CaImAn, etc.).
    processing_method: varchar(16)
    ---
    processing_method_desc: varchar(1000)
    """

    contents = [("caiman", "caiman analysis suite")]


@schema
class ProcessingParamSet(dj.Lookup):
    """Parameter set used for the processing of miniscope recordings.,
    including both the analysis suite and its respective input parameters.

    A hash of the parameters of the analysis suite is also stored in order
    to avoid duplicated entries.

    Attributes:
        paramset_idx (int): Unique parameter set ID.
        ProcessingMethod (foreign key): A primary key from ProcessingMethod.
        paramset_desc (str): Parameter set description.
        paramset_set_hash (uuid): A universally unique identifier for the parameter set.
        params (longblob): Parameter set, a dictionary of all applicable parameters to the analysis suite.
    """

    definition = """# Processing Parameter set
    paramset_idx:  smallint     # Unique parameter set ID.
    ---
    -> ProcessingMethod
    paramset_desc: varchar(1280)    # Parameter set description
    param_set_hash: uuid    # A universally unique identifier for the parameter set unique index (param_set_hash)
    params: longblob  # Parameter set, a dictionary of all applicable parameters to the analysis suite.
    """

    @classmethod
    def insert_new_params(
        cls,
        processing_method: str,
        paramset_idx: int,
        paramset_desc: str,
        params: dict,
    ):
        """Insert new parameter set.

        Args:
            processing_method (str): Name of the processing method or software.
            paramset_idx (int): Unique number for the set of processing parameters.
            paramset_desc (str): Description of the processing parameter set.
            params (dict): Dictionary of processing parameters for the selected processing_method.
            processing_method_desc (str, optional): Description of the processing method. Defaults to "".

        Raises:
            dj.DataJointError: A parameter set with arguments in this function already exists in the database.
        """

        ProcessingMethod.insert1(
            {
                "processing_method": processing_method,
                "processing_method_desc": "caiman_analysis",
            },
            skip_duplicates=True,
        )
        param_dict = {
            "processing_method": processing_method,
            "paramset_idx": paramset_idx,
            "paramset_desc": paramset_desc,
            "params": params,
            "param_set_hash": dict_to_uuid(params),
        }
        q_param = cls & {"param_set_hash": param_dict["param_set_hash"]}

        if q_param:  # If the specified param-set already exists
            pname = q_param.fetch1("paramset_idx")
            if pname == paramset_idx:  # If the existed set has the same name: job done
                return
            else:  # If not same name: human error, try adding with different name
                raise dj.DataJointError(
                    "The specified param-set already exists - name: {}".format(pname)
                )
        else:
            cls.insert1(param_dict)


@schema
class MaskType(dj.Lookup):
    """Possible classifications of a segmented mask.

    Attributes:
        mask_type (foreign key, varchar(16) ): Type of segmented mask.
    """

    definition = """ # Possible classifications for a segmented mask
    mask_type        : varchar(16)
    """

    contents = zip(["soma", "axon", "dendrite", "neuropil", "artefact", "unknown"])


@schema
class ProcessingTask(dj.Manual):
    """A pairing of processing params and recordings to be loaded or triggered.

    This table defines a miniscope recording processing task for a combination of a
    `Recording` and a `ProcessingParamSet` entries, including all the inputs (recording, method,
    method's parameters). The task defined here is then run in the downstream table
    `Processing`. This table supports definitions of both loading of pre-generated results
    and the triggering of new analysis for all supported analysis methods.

    Attributes:
        RecordingInfo (foreign key): Primary key from RecordingInfo.
        ProcessingParamSet (foreign key): Primary key from ProcessingParamSet.
        processing_output_dir (str): Output directory of the processed scan relative to the root data directory.
        task_mode (str): One of 'load' (load computed analysis results) or 'trigger'
            (trigger computation).
    """

    definition = """# Manual table for defining a processing task ready to be run
    -> RecordingInfo
    -> ProcessingParamSet
    ---
    processing_output_dir : varchar(255)    # relative to the root data directory
    task_mode='load'      : enum('load', 'trigger') # 'load': load existing results
                                                    # 'trigger': trigger procedure
    """

    @classmethod
    def infer_output_dir(cls, key, relative=False, mkdir=False):
        """Infer an output directory for an entry in ProcessingTask table.

        Args:
            key (dict): Primary key from the ProcessingTask table.
            relative (bool): If True, processing_output_dir is returned relative to
                imaging_root_dir. Default False.
            mkdir (bool): If True, create the processing_output_dir directory.
                Default True.

        Returns:
            dir (str): A default output directory for the processed results (processed_output_dir
                in ProcessingTask) based on the following convention:
                processed_dir / scan_dir / {processing_method}_{paramset_idx}
                e.g.: sub4/sess1/scan0/suite2p_0
        """
        acq_software = (Recording & key).fetch1("acq_software")
        recording_dir = find_full_path(
            get_miniscope_root_data_dir(),
            get_session_directory(key),
        )
        root_dir = find_root_directory(get_miniscope_root_data_dir(), recording_dir)

        method = (
            (ProcessingParamSet & key).fetch1("processing_method").replace(".", "-")
        )

        processed_dir = pathlib.Path(get_processed_root_data_dir())
        output_dir = (
            processed_dir
            / recording_dir.relative_to(root_dir)
            / f'{method}_{key["paramset_idx"]}'
        )

        if mkdir:
            output_dir.mkdir(parents=True, exist_ok=True)

        return output_dir.relative_to(processed_dir) if relative else output_dir

    @classmethod
    def generate(cls, recording_key, paramset_idx=0):
        """Generate a ProcessingTask for a Recording using an parameter ProcessingParamSet

        Generate an entry in the ProcessingTask table for a particular recording using an
        existing parameter set from the ProcessingParamSet table.

        Args:
            recording_key (dict): Primary key from Recording.
            paramset_idx (int): Unique parameter set ID.
        """
        key = {**recording_key, "paramset_idx": paramset_idx}

        processed_dir = get_processed_root_data_dir()
        output_dir = cls.infer_output_dir(key, relative=False, mkdir=True)

        method = (ProcessingParamSet & {"paramset_idx": paramset_idx}).fetch1(
            "processing_method"
        )

        try:
            if method == "caiman":
                from element_interface import caiman_loader

                caiman_loader.CaImAn(output_dir)
            else:
                raise NotImplementedError(
                    "Unknown/unimplemented method: {}".format(method)
                )
        except FileNotFoundError:
            task_mode = "trigger"
        else:
            task_mode = "load"

        cls.insert1(
            {
                **key,
                "processing_output_dir": output_dir.relative_to(
                    processed_dir
                ).as_posix(),
                "task_mode": task_mode,
            }
        )

    auto_generate_entries = generate


@schema
class Processing(dj.Computed):
    """Perform the computation of an entry (task) defined in the ProcessingTask table.
    The computation is performed only on the recordings with RecordingInfo inserted.


    Attributes:
        ProcessingTask (foreign key): Primary key from ProcessingTask.
        processing_time (datetime): Process completion datetime.
        package_version (str, optional): Version of the analysis package used in processing the data.
    """

    definition = """
    -> ProcessingTask
    ---
    processing_time     : datetime  # generation time of processed results
    package_version=''  : varchar(16)
    """

    def make(self, key):
        """
        Execute the miniscope analysis defined by the ProcessingTask.
        - task_mode: 'load', confirm that the results are already computed.
        - task_mode: 'trigger' runs the analysis.
        """
        task_mode, output_dir = (ProcessingTask & key).fetch1(
            "task_mode", "processing_output_dir"
        )

        if not output_dir:
            output_dir = ProcessingTask.infer_output_dir(key, relative=True, mkdir=True)
            # update processing_output_dir
            ProcessingTask.update1(
                {**key, "processing_output_dir": output_dir.as_posix()}
            )
        try:
            output_dir = find_full_path(
                get_miniscope_root_data_dir(), output_dir
            )
        except FileNotFoundError as e:
            if task_mode == "trigger":
                processed_dir = pathlib.Path(get_processed_root_data_dir())
                output_dir = processed_dir / output_dir
                output_dir.mkdir(parents=True, exist_ok=True)
            else:
                raise e

        if task_mode == "load":
            method, loaded_result = get_loader_result(key, ProcessingTask)
            if method == "caiman":
                loaded_caiman = loaded_result
                key = {**key, "processing_time": loaded_caiman.creation_time}
            else:
                raise NotImplementedError(
                    f"Loading of {method} data is not yet supported"
                )
        elif task_mode == "trigger":
            method = (
                ProcessingTask * ProcessingParamSet * ProcessingMethod * Recording & key
            ).fetch1("processing_method")

            if method == "caiman":
                import caiman
                from element_interface.run_caiman import run_caiman

                avi_files = (
                    Recording * RecordingInfo * RecordingInfo.File & key
                ).fetch("file_path")
                avi_files = [
                    find_full_path(get_miniscope_root_data_dir(), avi_file).as_posix()
                    for avi_file in avi_files
                ]

                params = (ProcessingTask * ProcessingParamSet & key).fetch1("params")
                sampling_rate = (
                    ProcessingTask * Recording * RecordingInfo & key
                ).fetch1("fps")

                run_caiman(
                    file_paths=avi_files,
                    parameters=params,
                    sampling_rate=sampling_rate,
                    output_dir=output_dir.as_posix(),
                    is3D=False,
                )


                _, imaging_dataset = get_loader_result(key, ProcessingTask)
                caiman_dataset = imaging_dataset
                key["processing_time"] = caiman_dataset.creation_time
                key["package_version"] = caiman.__version__
            else:
                raise NotImplementedError(
                    f"Automatic triggering of {method} analysis"
                    f" is not yet supported"
                )
        else:
            raise ValueError(f"Unknown task mode: {task_mode}")

        self.insert1(key)


# Motion Correction --------------------------------------------------------------------


@schema
class MotionCorrection(dj.Imported):
    """Automated table performing motion correction analysis.

    Attributes:
        Processing (foreign key): Processing primary key.
        Channel.proj(motion_correct_channel='channel'): Channel used for motion correction.
    """

    definition = """
    -> Processing
    ---
    -> Channel.proj(motion_correct_channel='channel') # channel used for
                                                      # motion correction
    """

    class RigidMotionCorrection(dj.Part):
        """Details of rigid motion correction performed on the imaging data.

        Attributes:
            MotionCorrection (foreign key): Primary key from MotionCorrection.
            outlier_frames (longblob): Mask with true for frames with outlier shifts
                (already corrected).
            y_shifts (longblob): y motion correction shifts (pixels).
            x_shifts (longblob): x motion correction shifts (pixels).
            z_shifts (longblob, optional): z motion correction shifts (z-drift, pixels).
            y_std (float): standard deviation of y shifts across all frames (pixels).
            x_std (float): standard deviation of x shifts across all frames (pixels).
            z_std (float, optional): standard deviation of z shifts across all frames
                (pixels).
        """

        definition = """# Details of rigid motion correction performed on the imaging data
        -> master
        ---
        outlier_frames=null : longblob  # mask with true for frames with outlier shifts (already corrected)
        y_shifts            : longblob  # (pixels) y motion correction shifts
        x_shifts            : longblob  # (pixels) x motion correction shifts
        z_shifts=null       : longblob  # (pixels) z motion correction shifts (z-drift)
        y_std               : float     # (pixels) standard deviation of y shifts across all frames
        x_std               : float     # (pixels) standard deviation of x shifts across all frames
        z_std=null          : float     # (pixels) standard deviation of z shifts across all frames
        """

    class NonRigidMotionCorrection(dj.Part):
        """Piece-wise rigid motion correction - tile the FOV into multiple 3D
        blocks/patches.

        Attributes:
            MotionCorrection (foreign key): Primary key from MotionCorrection.
            outlier_frames (longblob, null): Mask with true for frames with outlier
                shifts (already corrected).
            block_height (int): Block height in pixels.
            block_width (int): Block width in pixels.
            block_depth (int): Block depth in pixels.
            block_count_y (int): Number of blocks tiled in the y direction.
            block_count_x (int): Number of blocks tiled in the x direction.
            block_count_z (int): Number of blocks tiled in the z direction.
        """

        definition = """# Details of non-rigid motion correction performed on the imaging data
        -> master
        ---
        outlier_frames=null : longblob # mask with true for frames with outlier shifts (already corrected)
        block_height        : int      # (pixels)
        block_width         : int      # (pixels)
        block_depth         : int      # (pixels)
        block_count_y       : int      # number of blocks tiled in the y direction
        block_count_x       : int      # number of blocks tiled in the x direction
        block_count_z       : int      # number of blocks tiled in the z direction
        """

    class Block(dj.Part):
        """FOV-tiled blocks used for non-rigid motion correction.

        Attributes:
            NonRigidMotionCorrection (foreign key): Primary key from
                NonRigidMotionCorrection.
            block_id (int): Unique block ID.
            block_y (longblob): y_start and y_end in pixels for this block
            block_x (longblob): x_start and x_end in pixels for this block
            block_z (longblob): z_start and z_end in pixels for this block
            y_shifts (longblob): y motion correction shifts for every frame in pixels
            x_shifts (longblob): x motion correction shifts for every frame in pixels
            z_shift=null (longblob, optional): x motion correction shifts for every frame
                in pixels
            y_std (float): standard deviation of y shifts across all frames in pixels
            x_std (float): standard deviation of x shifts across all frames in pixels
            z_std=null (float, optional): standard deviation of z shifts across all frames
                in pixels
        """

        definition = """# FOV-tiled blocks used for non-rigid motion correction
        -> master.NonRigidMotionCorrection
        block_id        : int
        ---
        block_y         : longblob  # (y_start, y_end) in pixel of this block
        block_x         : longblob  # (x_start, x_end) in pixel of this block
        block_z         : longblob  # (z_start, z_end) in pixel of this block
        y_shifts        : longblob  # (pixels) y motion correction shifts for every frame
        x_shifts        : longblob  # (pixels) x motion correction shifts for every frame
        z_shifts=null   : longblob  # (pixels) z motion correction shifts for every frame
        y_std           : float     # (pixels) standard deviation of y shifts across all frames
        x_std           : float     # (pixels) standard deviation of x shifts across all frames
        z_std=null      : float     # (pixels) standard deviation of z shifts across all frames
        """

    class Summary(dj.Part):
        """Summary images for each field and channel after corrections.

        Attributes:
            MotionCorrection (foreign key): Primary key from MotionCorrection.
            ref_image (longblob): Image used as alignment template.
            average_image (longblob): Mean of registered frames.
            correlation_image (longblob, optional): Correlation map (computed during
                cell detection).
            max_proj_image (longblob, optional): Max of registered frames.
        """

        definition = """# Summary images for each field and channel after corrections
        -> master
        ---
        ref_image               : longblob  # image used as alignment template
        average_image           : longblob  # mean of registered frames
        correlation_image=null  : longblob  # correlation map (computed during cell detection)
        max_proj_image=null     : longblob  # max of registered frames
        """

    def make(self, key):
        """Populate tables with motion correction data."""
        method, loaded_result = get_loader_result(key, ProcessingTask)

        if method == "caiman":
            caiman_dataset = loaded_result

            self.insert1(
                {**key, "motion_correct_channel": caiman_dataset.alignment_channel}
            )

            # -- rigid motion correction --
            if caiman_dataset.is_pw_rigid:
                # -- non-rigid motion correction --
                (
                    nonrigid_correction,
                    nonrigid_blocks,
                ) = caiman_dataset.extract_pw_rigid_mc()
                nonrigid_correction.update(**key)
                nonrigid_blocks.update(**key)
                self.NonRigidMotionCorrection.insert1(nonrigid_correction)
                self.Block.insert(nonrigid_blocks)
            else:
                # -- rigid motion correction --
                rigid_correction = caiman_dataset.extract_rigid_mc()
                rigid_correction.update(**key)
                self.RigidMotionCorrection.insert1(rigid_correction)

            # -- summary images --
            summary_images = {
                **key,
                "ref_image": caiman_dataset.ref_image.transpose(2, 0, 1),
                "average_image": caiman_dataset.mean_image.transpose(2, 0, 1),
                "correlation_image": caiman_dataset.correlation_map.transpose(2, 0, 1),
                "max_proj_image": caiman_dataset.max_proj_image.transpose(2, 0, 1),
            }
            self.Summary.insert1(summary_images)

        else:
            raise NotImplementedError("Unknown/unimplemented method: {}".format(method))


# Segmentation -------------------------------------------------------------------------


@schema
class Segmentation(dj.Computed):
    """Automated table computes different mask segmentations.

    Attributes:
        Processing (foreign key): Processing primary key.
    """

    definition = """ # Different mask segmentations.
    -> Processing
    """

    class Mask(dj.Part):
        """Details of the masks identified from the Segmentation procedure.

        Attributes:
            Segmentation (foreign key): Primary key from Segmentation.
            mask (int): Unique mask ID.
            Channel.proj(segmentation_channel='channel') (foreign key): Channel
                used for segmentation.
            mask_npix (int): Number of pixels in ROIs.
            mask_center_x (int): Center x coordinate in pixel.
            mask_center_y (int): Center y coordinate in pixel.
            mask_center_z (int): Center z coordinate in pixel.
            mask_xpix (longblob): X coordinates in pixels.
            mask_ypix (longblob): Y coordinates in pixels.
            mask_zpix (longblob): Z coordinates in pixels.
            mask_weights (longblob): Weights of the mask at the indices above.
        """

        definition = """ # A mask produced by segmentation.
        -> master
        mask               : smallint
        ---
        -> Channel.proj(segmentation_channel='channel')  # channel used for segmentation
        mask_npix          : int       # number of pixels in ROIs
        mask_center_x      : int       # center x coordinate in pixel
        mask_center_y      : int       # center y coordinate in pixel
        mask_center_z=null : int       # center z coordinate in pixel
        mask_xpix          : longblob  # x coordinates in pixels
        mask_ypix          : longblob  # y coordinates in pixels
        mask_zpix=null     : longblob  # z coordinates in pixels
        mask_weights       : longblob  # weights of the mask at the indices above
        """

    def make(self, key):
        """Populates table with segmentation data."""
        method, loaded_result = get_loader_result(key, ProcessingTask)

        if method == "caiman":
            caiman_dataset = loaded_result

            # infer "segmentation_channel" - from params if available, else from caiman loader
            params = (ProcessingParamSet * ProcessingTask & key).fetch1("params")
            segmentation_channel = params.get(
                "segmentation_channel", caiman_dataset.segmentation_channel
            )

            masks, cells = [], []
            for mask in caiman_dataset.masks:
                masks.append(
                    {
                        **key,
                        "segmentation_channel": segmentation_channel,
                        "mask": mask["mask_id"],
                        "mask_npix": mask["mask_npix"],
                        "mask_center_x": mask["mask_center_x"],
                        "mask_center_y": mask["mask_center_y"],
                        "mask_center_z": mask["mask_center_z"],
                        "mask_xpix": mask["mask_xpix"],
                        "mask_ypix": mask["mask_ypix"],
                        "mask_zpix": mask["mask_zpix"],
                        "mask_weights": mask["mask_weights"],
                    }
                )
                if mask["accepted"]:
                    cells.append(
                        {
                            **key,
                            "mask_classification_method": "caiman_default_classifier",
                            "mask": mask["mask_id"],
                            "mask_type": "soma",
                        }
                    )

            self.insert1(key)
            self.Mask.insert(masks, ignore_extra_fields=True)

            if cells:
                MaskClassification.insert1(
                    {
                        **key,
                        "mask_classification_method": "caiman_default_classifier",
                    },
                    allow_direct_insert=True,
                )
                MaskClassification.MaskType.insert(
                    cells, ignore_extra_fields=True, allow_direct_insert=True
                )

        else:
            raise NotImplementedError(f"Unknown/unimplemented method: {method}")


@schema
class MaskClassificationMethod(dj.Lookup):
    """Method to classify segmented masks.

    Attributes:
        mask_classification_method (foreign key, varchar(48) ): Method by which masks
            are classified into mask types.
    """

    definition = """
    mask_classification_method: varchar(48)
    """

    contents = zip(["caiman_default_classifier"])


@schema
class MaskClassification(dj.Computed):
    """Automated table with mask classification data.

    Attributes:
        Segmentation (foreign key): Segmentation primary key.
        MaskClassificationMethod (foreign key): MaskClassificationMethod primary key.
    """

    definition = """
    -> Segmentation
    -> MaskClassificationMethod
    """

    class MaskType(dj.Part):
        """Automated table storing mask type data.

        Attributes:
            MaskClassification (foreign key): MaskClassification primary key.
            Segmentation.Mask (foreign key): Segmentation.Mask primary key.
            MaskType (dict): Select mask type from entries within `MaskType` look up table.
            confidence (float): Statistical confidence of mask classification.
        """

        definition = """
        -> master
        -> Segmentation.Mask
        ---
        -> MaskType
        confidence=null: float
        """

    def make(self, key):
        raise NotImplementedError(
            "To add to this table, use `insert` with allow_direct_insert=True"
        )


# Fluorescence & Activity Traces -------------------------------------------------------


@schema
class Fluorescence(dj.Computed):
    """Extracts fluorescence trace information.

    Attributes:
        Segmentation (foreign key): Segmentation primary key.
    """

    definition = """  # fluorescence traces before spike extraction or filtering
    -> Segmentation
    """

    class Trace(dj.Part):
        """Automated table with Fluorescence traces

        Attributes:
            Fluorescence (foreign key): Fluorescence primary key.
            Segmentation.Mask (foreign key): Segmentation.Mask primary key.
            Channel.proj(fluorescence_channel='channel') (foreign key, query): Channel
                used for this trace.
            fluorescence (longblob): A fluorescence trace associated with a given mask.
            neuropil_fluorescence (longblob): A neuropil fluorescence trace.
        """

        definition = """
        -> master
        -> Segmentation.Mask
        -> Channel.proj(fluorescence_channel='channel')  # channel used for this trace
        ---
        fluorescence                : longblob  # fluorescence trace associated
                                                # with this mask
        neuropil_fluorescence=null  : longblob  # Neuropil fluorescence trace
        """

    def make(self, key):
        """Populates table with fluorescence trace data."""
        method, loaded_result = get_loader_result(key, ProcessingTask)

        if method == "caiman":
            caiman_dataset = loaded_result

            # infer "segmentation_channel" - from params if available, else from caiman loader
            params = (ProcessingParamSet * ProcessingTask & key).fetch1("params")
            segmentation_channel = params.get(
                "segmentation_channel", caiman_dataset.segmentation_channel
            )

            fluo_traces = []
            for mask in caiman_dataset.masks:
                fluo_traces.append(
                    {
                        **key,
                        "mask": mask["mask_id"],
                        "fluorescence_channel": segmentation_channel,
                        "fluorescence": mask["inferred_trace"],
                    }
                )

            self.insert1(key)
            self.Trace.insert(fluo_traces)

        else:
            raise NotImplementedError("Unknown/unimplemented method: {}".format(method))


@schema
class ActivityExtractionMethod(dj.Lookup):
    """Lookup table for activity extraction methods.

    Attributes:
        extraction_method (foreign key, varchar(32) ): Extraction method from CaImAn.
    """

    definition = """
    extraction_method: varchar(32)
    """

    contents = zip(["caiman_deconvolution", "caiman_dff"])


@schema
class Activity(dj.Computed):
    """Inferred neural activity from the fluorescence trace.

    Attributes:
        Fluorescence (foreign key): Fluorescence primary key.
        ActivityExtractionMethod (foreign key): ActivityExtractionMethod primary key.
    """

    definition = """
    # inferred neural activity from fluorescence trace - e.g. dff, spikes
    -> Fluorescence
    -> ActivityExtractionMethod
    """

    class Trace(dj.Part):
        """Automated table with activity traces.

        Attributes:
            Activity (foreign key): Activity primary key.
            Fluorescence.Trace (foreign key): fluorescence.Trace primary key.
            activity_trace (longblob): Inferred activity trace.
        """

        definition = """
        -> master
        -> Fluorescence.Trace
        ---
        activity_trace: longblob
        """

    @property
    def key_source(self):
        """Defines the order of keys when the `make` function is called."""
        caiman_key_source = (
            Fluorescence
            * ActivityExtractionMethod
            * ProcessingParamSet.proj("processing_method")
            & 'processing_method = "caiman"'
            & 'extraction_method LIKE "caiman%"'
        )

        return caiman_key_source.proj()

    def make(self, key):
        """Populates table with activity trace data."""
        method, loaded_result = get_loader_result(key, ProcessingTask)

        if method == "caiman":
            caiman_dataset = loaded_result

            if key["extraction_method"] in (
                "caiman_deconvolution",
                "caiman_dff",
            ):
                attr_mapper = {
                    "caiman_deconvolution": "spikes",
                    "caiman_dff": "dff",
                }

                # infer "segmentation_channel" - from params if available, else from caiman loader
                params = (ProcessingParamSet * ProcessingTask & key).fetch1("params")
                segmentation_channel = params.get(
                    "segmentation_channel", caiman_dataset.segmentation_channel
                )

                self.insert1(key)
                self.Trace.insert(
                    dict(
                        key,
                        mask=mask["mask_id"],
                        fluorescence_channel=segmentation_channel,
                        activity_trace=mask[attr_mapper[key["extraction_method"]]],
                    )
                    for mask in caiman_dataset.masks
                )

        else:
            raise NotImplementedError("Unknown/unimplemented method: {}".format(method))


@schema
class ProcessingQualityMetrics(dj.Computed):
    """Quality metrics used to evaluate the results of the calcium imaging analysis pipeline.

    Attributes:
        Fluorescence (foreign key): Primary key from Fluorescence.
    """

    definition = """
    -> Fluorescence
    """

    class Trace(dj.Part):
        """Quality metrics used to evaluate the fluorescence traces.

        Attributes:
            Fluorescence (foreign key): Primary key from Fluorescence.
            Fluorescence.Trace (foreign key): Primary key from Fluorescence.Trace.
            skewness (float): Skewness of the fluorescence trace.
            variance (float): Variance of the fluorescence trace.
        """

        definition = """
        -> master
        -> Fluorescence.Trace
        ---
        skewness: float   # Skewness of the fluorescence trace.
        variance: float   # Variance of the fluorescence trace.
        """

    def make(self, key):
        """Populate the ProcessingQualityMetrics table and its part tables."""
        from scipy.stats import skew

        (
            fluorescence,
            fluorescence_channels,
            mask_ids,
        ) = (
            Segmentation.Mask * RecordingInfo * Fluorescence.Trace & key
        ).fetch("fluorescence", "fluorescence_channel", "mask")

        fluorescence = np.stack(fluorescence)

        self.insert1(key)

        self.Trace.insert(
            dict(
                key,
                fluorescence_channel=fluorescence_channel,
                mask=mask_id,
                skewness=skewness,
                variance=variance,
            )
            for fluorescence_channel, mask_id, skewness, variance in zip(
                fluorescence_channels,
                mask_ids,
                skew(fluorescence, axis=1),
                fluorescence.std(axis=1),
            )
        )


# Helper Functions ---------------------------------------------------------------------

_table_attribute_mapper = {
    "ProcessingTask": "processing_output_dir",
}


def get_loader_result(key, table) -> tuple:
    """Retrieve the loaded processed imaging results from the loader (e.g. caiman, etc.)

    Args:
        key (dict): the `key` to one entry of ProcessingTask.
        table (str): the class defining the table to retrieve
            the loaded results from (e.g. ProcessingTask).

    Returns:
        method, loaded_output (tuple): method string and loader object with results (e.g. caiman.CaImAn, etc.)
    """

    method, output_dir = (ProcessingParamSet * table & key).fetch1(
        "processing_method", _table_attribute_mapper[table.__name__]
    )

    output_dir = find_full_path(get_miniscope_root_data_dir(), output_dir)

    if method == "caiman":
        from element_interface import caiman_loader

        loaded_output = caiman_loader.CaImAn(output_dir)
    else:
        raise NotImplementedError("Unknown/unimplemented method: {}".format(method))

    return method, loaded_output
