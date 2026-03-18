import csv
import copy
import cv2
import gc
import importlib
import inspect
import json
import os
import pathlib
import time
from datetime import datetime, timezone
from typing import Union

import datajoint as dj
import numpy as np
import pandas as pd
from element_interface.utils import (
    dict_to_uuid,
    find_full_path,
    find_root_directory,
    memoized_result,
)

logger = dj.logger

schema = dj.schema()

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

# ─── Configuration ────────────────────────────────────────────────────────

# Docker image for minian processing.
# Override with env var MINIAN_DOCKER_IMAGE if using a custom registry.
MINIAN_DOCKER_IMAGE = os.getenv(
    "MINIAN_DOCKER_IMAGE", "datajoint/minian-py38:latest"
)

# Timeout for the docker run command (seconds). Default: 24 hours.
# Large datasets (70k+ frames) can take many hours.
MINIAN_DOCKER_TIMEOUT = int(os.getenv("MINIAN_DOCKER_TIMEOUT", "86400"))

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
    contents = zip(["Miniscope-DAQ-V3", "Miniscope-DAQ-V4", "Inscopix", "Bonsai"])


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
            metadata = None  # Initialize to handle the no-metadata case

            try:
                recording_metadata = next(recording_path.glob("metaData.json"))
                with open(recording_metadata.as_posix()) as f:
                    metadata = json.loads(f.read())
                px_height = metadata["ROI"]["height"]
                px_width = metadata["ROI"]["width"]
                fps = int(metadata["frameRate"].replace("FPS", ""))
            except StopIteration:
                logger.warning(
                    f"No metaData.json file found in {recording_directory}\n"
                    "Extracting metadata from the .avi file header instead."
                )
                miniscope_video = cv2.VideoCapture(recording_filepaths[0])
                px_height = int(miniscope_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
                px_width = int(miniscope_video.get(cv2.CAP_PROP_FRAME_WIDTH))
                fps = miniscope_video.get(cv2.CAP_PROP_FPS)
                miniscope_video.release()

            # Handle timestamps for multiple vs single AVI files
            time_stamps = None
            if len(recording_filepaths) > 1:
                # Multiple AVI files - look for timestamp CSV paired with each AVI
                timestamps_files = sorted(recording_path.glob("*.csv"))
                all_timestamps = []
                for timestamps_file in timestamps_files:
                    with open(timestamps_file, newline="") as f:
                        reader = csv.reader(f, delimiter=",")
                        next(reader)  # Skip header for each file
                        all_timestamps.extend(list(reader))
                if all_timestamps:
                    time_stamps = np.array(all_timestamps, dtype=float)[:, 0]
                    nframes = len(time_stamps)
                else:
                    logger.warning(
                        f"No timestamp CSV files found in {recording_directory}"
                    )
            else:
                try:
                    recording_timestamps = next(recording_path.glob("timeStamps.csv"))
                    with open(recording_timestamps, newline="") as f:
                        reader = csv.reader(f, delimiter=",")
                        next(reader)  # Skip header
                        time_stamps = np.array(list(reader), dtype=float)[:, 0]
                    nframes = len(time_stamps)
                except StopIteration:
                    logger.warning(
                        f"No timeStamps.csv file found in {recording_directory}"
                    )

            # Fallback: get nframes from video if timestamps not available
            if time_stamps is None or len(time_stamps) == 0:
                total_frames = 0
                for avi_file in recording_filepaths:
                    video = cv2.VideoCapture(avi_file)
                    total_frames += int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                    video.release()
                nframes = total_frames

            nchannels = 1  # Assumes a single channel

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

        elif acq_software == "Bonsai":
            logger.warning(
                f"Limited support for Bonsai recordings. Metadata will be extracted directly from the `.avi` files. To improve support, please contact the developers or open an issue on the GitHub repository."
            )

            miniscope_video = cv2.VideoCapture(recording_filepaths[0])

            nchannels = 1
            nframes = int(miniscope_video.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = miniscope_video.get(cv2.CAP_PROP_FPS)
            px_height = int(miniscope_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            px_width = int(miniscope_video.get(cv2.CAP_PROP_FRAME_WIDTH))

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

        if acq_software == "Inscopix":
            self.Timestamps.insert1(dict(**key, timestamps=time_stamps))
            self.Config.insert1(dict(**key, config=metadata))
        elif acq_software == "Miniscope-DAQ-V4":
            if time_stamps is not None and len(time_stamps) > 0:
                self.Timestamps.insert1(dict(**key, timestamps=time_stamps))
            if metadata is not None:
                self.Config.insert1(dict(**key, config=metadata))


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

    contents = [
        ("caiman", "caiman analysis suite"),
        ("minian", "minian analysis suite"),
    ]


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
        Recording (foreign key): Primary key from Recording.
        ProcessingParamSet (foreign key): Primary key from ProcessingParamSet.
        processing_output_dir (str): Output directory of the processed scan relative to the root data directory.
        task_mode (str): One of 'load' (load computed analysis results) or 'trigger'
            (trigger computation).
    """

    definition = """# Manual table for defining a processing task ready to be run
    -> Recording
    -> ProcessingParamSet
    ---
    processing_output_dir='': varchar(255)    # relative to the root data directory
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

    class File(dj.Part):
        definition = """
        -> master
        file_name: varchar(255)     # file name
        ---
        file: filepath@miniscope-processed
        """

    @property
    def key_source(self):
        return ProcessingTask & RecordingInfo

    def make_fetch(self, key):
        task_mode, processing_output_dir = (ProcessingTask & key).fetch1(
            "task_mode", "processing_output_dir"
        )
        method = (ProcessingParamSet & key).fetch1("processing_method")
        avi_files = (RecordingInfo.File & key).fetch("file_path")
        processing_params = (ProcessingParamSet & key).fetch1("params")
        sampling_rate = (RecordingInfo & key).fetch1("fps")
        px_height, px_width = (RecordingInfo & key).fetch1("px_height", "px_width")

        return (
            task_mode,
            processing_output_dir,
            method,
            avi_files,
            processing_params,
            sampling_rate,
            px_height,
            px_width,
        )

    def make_compute(
        self,
        key,
        task_mode,
        processing_output_dir,
        method,
        avi_files,
        processing_params,
        sampling_rate,
        px_height,
        px_width,
    ):
        """
        Execute the miniscope analysis defined by the ProcessingTask.
        - task_mode: 'load', confirm that the results are already computed.
        - task_mode: 'trigger' runs the analysis.
        """
        if method not in ["caiman", "minian"]:
            raise NotImplementedError(f"Method {method} is not supported")

        params = copy.deepcopy(processing_params)

        if not processing_output_dir:
            output_dir = ProcessingTask.infer_output_dir(key, relative=True, mkdir=True)
        else:
            output_dir = processing_output_dir
        try:
            output_dir = find_full_path(get_processed_root_data_dir(), output_dir)
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
            elif method == "minian":
                loaded_minian = loaded_result
                key = {**key, "processing_time": loaded_minian.creation_time}
            else:
                raise NotImplementedError(
                    f"Loading of {method} data is not yet supported"
                )
            file_entries = [
                {
                    **key,
                    "file_name": f.relative_to(
                        get_processed_root_data_dir()
                    ).as_posix(),
                    "file": f.as_posix(),
                }
                for f in output_dir.rglob("*")
                if f.is_file()
            ]
        elif task_mode == "trigger":
            avi_files = [
                find_full_path(get_miniscope_root_data_dir(), avi_file).as_posix()
                for avi_file in avi_files
            ]
            if method == "caiman":
                import multiprocessing
                import caiman as cm
                from caiman.motion_correction import MotionCorrect
                from caiman.source_extraction.cnmf.cnmf import CNMF
                from caiman.source_extraction.cnmf.params import CNMFParams
                from element_interface.run_caiman import _save_mc

                extra_params = params.pop("extra_dj_params", {})

                params["fnames"] = avi_files
                params["fr"] = sampling_rate
                params["is3D"] = False
                if "indices" in params:
                    params["motion"] = {
                        "indices": (
                            slice(*params.get("indices")[0]),
                            slice(*params.get("indices")[1]),
                        )
                    }
                else:
                    params["motion"] = {"indices": (slice(None), slice(None))}

                @memoized_result(
                    uniqueness_dict=params,
                    output_directory=output_dir,
                )
                def _run_processing():
                    mc_indices = params["motion"].get("indices")
                    caiman_temp = os.environ.get("CAIMAN_TEMP")
                    os.environ["CAIMAN_TEMP"] = str(output_dir)
                    n_processes = np.floor(multiprocessing.cpu_count() * 0.6)
                    n_processes = int(os.getenv("CAIMAN_MC_N_PROCESSES", n_processes))
                    _, dview, n_processes = cm.cluster.setup_cluster(
                        backend="multiprocessing",
                        n_processes=n_processes,
                        maxtasksperchild=1,
                    )
                    try:
                        opts = CNMFParams(params_dict=params)
                        cnm = CNMF(n_processes, params=opts, dview=dview)
                        fnames = cnm.params.get("data", "fnames")
                        mc = MotionCorrect(fnames, dview=cnm.dview, **cnm.params.motion)
                        mc_base_attrs = list(mc.__dict__)
                        logger.info("Starting motion correction (CaImAn)...")
                        mc.motion_correct(save_movie=mc_indices is None)
                        mc_results = {
                            k: v
                            for k, v in mc.__dict__.items()
                            if k not in mc_base_attrs
                        }
                        if cnm.params.get("motion", "pw_rigid"):
                            mc_results["b0"] = np.ceil(
                                np.max(np.abs(mc.shifts_rig))
                            ).astype(int)
                            cnm.estimates.shifts = mc.shifts_rig
                            if cnm.params.get("motion", "is3D"):
                                cnm.estimates.shifts = [
                                    mc.x_shifts_els,
                                    mc.y_shifts_els,
                                    mc.z_shifts_els,
                                ]
                            else:
                                cnm.estimates.shifts = [
                                    mc.x_shifts_els,
                                    mc.y_shifts_els,
                                ]
                        else:
                            mc_results["b0"] = np.ceil(
                                np.max(np.abs(mc.shifts_rig))
                            ).astype(int)
                            cnm.estimates.shifts = mc.shifts_rig

                        base_name = pathlib.Path(fnames[0]).stem
                        fname_mc = (
                            mc.fname_tot_els
                            if cnm.params.motion["pw_rigid"]
                            else mc.fname_tot_rig
                        )
                        if all(fname_mc):
                            logger.info("Generating C-order memmap file...")
                            border_to_0 = (
                                0 if mc.border_nan == "copy" else mc.border_to_0
                            )
                            fname_new = cm.mmapping.save_memmap(
                                fname_mc,
                                base_name=base_name + "_mc",
                                order="C",
                                var_name_hdf5=cnm.params.get("data", "var_name_hdf5"),
                                border_to_0=border_to_0,
                            )
                        else:
                            logger.info(
                                "Applying shifts, then generating C-order memmap file..."
                            )
                            fname_new = mc.apply_shifts_movie(
                                fnames,
                                save_memmap=True,
                                save_base_name=base_name + "_mc",
                                order="C",
                            )
                            mc.mmap_file = [fname_new]
                        Yr, dims, T = cm.mmapping.load_memmap(fname_new)
                        images = np.reshape(Yr.T, [T] + list(dims), order="F")
                        cnm.mmap_file = fname_new
                        # terminate the previous cluster and setup a new one with fewer
                        # processes for CNMF because it is memory intensive
                        dview.terminate()
                        n_processes = np.floor(multiprocessing.cpu_count() * 0.2)
                        n_processes = int(
                            os.getenv("CAIMAN_CNMF_N_PROCESSES", n_processes)
                        )
                        _, dview, n_processes = cm.cluster.setup_cluster(
                            backend="multiprocessing",
                            n_processes=n_processes,
                            maxtasksperchild=1,
                        )
                        cnm.dview = dview
                        logger.info(
                            f"Starting CNMF analysis with {n_processes} processes..."
                        )

                        cnm.fit(images, indices=(slice(None), slice(None)))
                        cnm.estimates.evaluate_components(
                            images, cnm.params, dview=cnm.dview
                        )
                        cnm.estimates.detrend_df_f(quantileMin=8, frames_window=250)
                        logger.info("Computing summary images...")
                        correlation_image, _ = cm.summary_images.correlation_pnr(
                            images[:: max(T // 1000, 1)],
                            gSig=cnm.params.init["gSig"][0],
                            swap_dim=False,
                        )
                        correlation_image[np.isnan(correlation_image)] = 0
                        cnm.estimates.Cn = correlation_image
                        fname_hdf5 = cnm.mmap_file[:-4] + "hdf5"
                        cnm.save(fname_hdf5)
                        cnmf_output_file = pathlib.Path(fname_hdf5)
                        summary_images = {
                            "average_image": np.mean(
                                images[:: max(T // 1000, 1)], axis=0
                            ),
                            "max_image": np.max(images[:: max(T // 1000, 1)], axis=0),
                            "correlation_image": correlation_image,
                        }
                        _save_mc(
                            mc,
                            cnmf_output_file.as_posix(),
                            params["is3D"],
                            summary_images=summary_images,
                        )
                    except Exception as e:
                        dview.terminate()
                        raise e
                    else:
                        cm.stop_server(dview=dview)
                        logger.info("CNMF analysis complete. Resulted saved.")
                        caiman_temp = os.environ.get("CAIMAN_TEMP")
                        if caiman_temp is not None:
                            os.environ["CAIMAN_TEMP"] = caiman_temp
                        else:
                            del os.environ["CAIMAN_TEMP"]

                _run_processing()
                _, imaging_dataset = get_loader_result(
                    key, ProcessingTask, full_output_dir=output_dir
                )
                caiman_dataset = imaging_dataset
                key["processing_time"] = caiman_dataset.creation_time
                key["package_version"] = cm.__version__
                file_entries = [
                    {
                        **key,
                        "file_name": f.relative_to(
                            get_processed_root_data_dir()
                        ).as_posix(),
                        "file": f.as_posix(),
                    }
                    for f in output_dir.rglob("*")
                    if f.is_file()
                ]
            elif method == "minian":
                logger.info("Running minian via Docker container...")

                # Run the minian pipeline in a Docker container
                status = _run_minian_in_container(
                    avi_files=avi_files,
                    output_dir=output_dir,
                    params=params,
                )

                # Set processing time from container output
                key["processing_time"] = status.get(
                    "timestamp", time.strftime("%Y-%m-%dT%H:%M:%S")
                )

                # Get minian version (from status.json or default)
                key["package_version"] = status.get("minian_version", "")

                # Collect output files for insertion into DataJoint
                file_entries = [
                    {
                        **key,
                        "file_name": f.name,
                        "file": f.as_posix(),
                    }
                    for f in output_dir.rglob("*")
                    if f.is_file()
                ]
            else:
                raise ValueError(f"Unknown task mode: {task_mode}")
        return (file_entries, output_dir)

    def make_insert(self, key, file_entries, output_dir):
        # update processing_output_dir
        ProcessingTask.update1(
            {
                **key,
                "processing_output_dir": output_dir.relative_to(
                    get_processed_root_data_dir()
                ).as_posix(),
            }
        )
        self.insert1(
            dict(
                **key,
                processing_time=key.get(
                    "processing_time", datetime.now(timezone.utc)
                ),
            )
        )


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
                self.NonRigidMotionCorrection.insert1(nonrigid_correction)
                self.Block.insert(
                    [{**block, **key} for block in nonrigid_blocks.values()]
                )
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

        elif method == "minian":
            minian_dataset = loaded_result

            self.insert1(
                {**key, "motion_correct_channel": minian_dataset.alignment_channel}
            )

            # Minian uses rigid motion correction
            rigid_correction = minian_dataset.extract_rigid_mc()
            if rigid_correction is not None:
                rigid_correction.update(**key)
                self.RigidMotionCorrection.insert1(rigid_correction)

            # -- summary images --
            ref_image = minian_dataset.ref_image
            mean_image = minian_dataset.mean_image
            max_proj_image = minian_dataset.max_proj_image
            correlation_image = minian_dataset.correlation_map

            summary_images = {
                **key,
                "ref_image": (
                    ref_image if ref_image is not None else np.zeros((1, 1, 1))
                ),
                "average_image": (
                    mean_image if mean_image is not None else np.zeros((1, 1, 1))
                ),
                "correlation_image": correlation_image,
                "max_proj_image": max_proj_image,
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

        elif method == "minian":
            minian_dataset = loaded_result

            # infer "segmentation_channel" - from params if available, else from minian loader
            params = (ProcessingParamSet * ProcessingTask & key).fetch1("params")
            segmentation_channel = params.get(
                "segmentation_channel", minian_dataset.segmentation_channel
            )

            masks, cells = [], []
            for mask in minian_dataset.masks:
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
                            "mask_classification_method": "minian_default_classifier",
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
                        "mask_classification_method": "minian_default_classifier",
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

    contents = zip(["caiman_default_classifier", "minian_default_classifier"])


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

        elif method == "minian":
            minian_dataset = loaded_result

            # infer "segmentation_channel" - from params if available, else from minian loader
            params = (ProcessingParamSet * ProcessingTask & key).fetch1("params")
            segmentation_channel = params.get(
                "segmentation_channel", minian_dataset.segmentation_channel
            )

            fluo_traces = []
            for mask in minian_dataset.masks:
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
        extraction_method (foreign key, varchar(32) ): Extraction method from CaImAn or Minian.
    """

    definition = """
    extraction_method: varchar(32)
    """

    contents = zip(["caiman_deconvolution", "caiman_dff", "minian_deconvolution"])


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

        minian_key_source = (
            Fluorescence
            * ActivityExtractionMethod
            * ProcessingParamSet.proj("processing_method")
            & 'processing_method = "minian"'
            & 'extraction_method LIKE "minian%"'
        )

        return caiman_key_source.proj() + minian_key_source.proj()

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

        elif method == "minian":
            minian_dataset = loaded_result

            if key["extraction_method"] == "minian_deconvolution":
                # infer "segmentation_channel" - from params if available, else from minian loader
                params = (ProcessingParamSet * ProcessingTask & key).fetch1("params")
                segmentation_channel = params.get(
                    "segmentation_channel", minian_dataset.segmentation_channel
                )

                self.insert1(key)
                self.Trace.insert(
                    dict(
                        key,
                        mask=mask["mask_id"],
                        fluorescence_channel=segmentation_channel,
                        activity_trace=mask["spikes"],
                    )
                    for mask in minian_dataset.masks
                    if "spikes" in mask
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


class MinianLoader:
    """Loader class for Minian analysis results.

    Provides a consistent interface for accessing Minian outputs similar to CaImAn loader.
    """

    def __init__(self, output_dir):
        """Initialize the MinianLoader.

        Args:
            output_dir: Path to the directory containing Minian zarr outputs.
        """
        import xarray as xr

        self.output_dir = pathlib.Path(output_dir)

        # Load each zarr variable into a combined Dataset
        # (replicates what minian's open_minian does without requiring
        # the minian package to be installed)
        ds = xr.Dataset()
        for zarr_path in sorted(self.output_dir.glob("*.zarr")):
            try:
                var_ds = xr.open_zarr(str(zarr_path))
                for name, da in var_ds.data_vars.items():
                    ds[name] = da
            except Exception:
                logger.debug(f"Skipping non-zarr path: {zarr_path}")
        self._minian_ds = ds

        # Load core arrays
        self._A = self._minian_ds.get(
            "A"
        )  # Spatial footprints (unit_id, height, width)
        self._C = self._minian_ds.get("C")  # Temporal traces (unit_id, frame)
        self._S = self._minian_ds.get(
            "S"
        )  # Deconvolved activity/spikes (unit_id, frame)
        self._b = self._minian_ds.get("b")  # Background spatial (height, width)
        self._f = self._minian_ds.get("f")  # Background temporal (frame)
        self._b0 = self._minian_ds.get("b0")  # Baseline (unit_id, frame)
        self._c0 = self._minian_ds.get("c0")  # Initial calcium (unit_id, frame)

        # Try to load motion correction data
        self._motion = self._minian_ds.get("motion")

        # Summary images
        self._max_proj = self._minian_ds.get("max_proj")
        self._mean_proj = self._minian_ds.get("mean_proj")
        self._ref_image = self._minian_ds.get("ref_image")

    @property
    def minian_dataset(self):
        """Return the raw Minian xarray Dataset."""
        return self._minian_ds

    @property
    def creation_time(self):
        """Get the creation time of the Minian output."""
        # Use the modification time of the output directory
        return datetime.fromtimestamp(self.output_dir.stat().st_mtime, tz=timezone.utc)

    @property
    def alignment_channel(self):
        """Channel used for motion correction (default 0 for miniscope)."""
        return 0

    @property
    def segmentation_channel(self):
        """Channel used for segmentation (default 0 for miniscope)."""
        return 0

    @property
    def is_pw_rigid(self):
        """Minian uses rigid motion correction by default."""
        return False

    @property
    def motion_shifts(self):
        """Return motion correction shifts as dict with 'x' and 'y' keys."""
        if self._motion is not None:
            motion_data = self._motion.compute()
            return {
                "x": motion_data.sel(shift_dim="width").values,
                "y": motion_data.sel(shift_dim="height").values,
            }
        return None

    def extract_rigid_mc(self):
        """Extract rigid motion correction data in format compatible with MotionCorrection table."""
        shifts = self.motion_shifts
        if shifts is None:
            return None

        return {
            "x_shifts": shifts["x"],
            "y_shifts": shifts["y"],
            "x_std": np.std(shifts["x"]),
            "y_std": np.std(shifts["y"]),
        }

    @property
    def ref_image(self):
        """Return reference image used for motion correction."""
        if self._ref_image is not None:
            return self._ref_image.compute().values[np.newaxis, :, :]
        return None

    @property
    def mean_image(self):
        """Return mean image (average across frames)."""
        if self._mean_proj is not None:
            return self._mean_proj.compute().values[np.newaxis, :, :]
        return None

    @property
    def max_proj_image(self):
        """Return maximum projection image."""
        if self._max_proj is not None:
            return self._max_proj.compute().values[np.newaxis, :, :]
        return None

    @property
    def correlation_map(self):
        """Return correlation image (computed during initialization)."""
        # Minian doesn't store correlation image by default
        # Return None or compute if needed
        return None

    @property
    def masks(self):
        """Extract mask information in format compatible with Segmentation table.

        Yields dict for each unit with mask properties.
        """
        if self._A is None:
            return []

        A_data = self._A.compute()
        C_data = self._C.compute() if self._C is not None else None
        S_data = self._S.compute() if self._S is not None else None

        masks = []
        for unit_idx, unit_id in enumerate(A_data.coords["unit_id"].values):
            footprint = A_data.sel(unit_id=unit_id).values

            # Find non-zero pixels
            mask_indices = np.where(footprint > 0)
            if len(mask_indices[0]) == 0:
                continue

            y_pix = mask_indices[0]
            x_pix = mask_indices[1]
            weights = footprint[y_pix, x_pix]

            mask_dict = {
                "mask_id": int(unit_id),
                "mask_npix": len(x_pix),
                "mask_center_x": int(np.mean(x_pix)),
                "mask_center_y": int(np.mean(y_pix)),
                "mask_center_z": None,
                "mask_xpix": x_pix,
                "mask_ypix": y_pix,
                "mask_zpix": None,
                "mask_weights": weights,
                "accepted": True,  # All units accepted (no manual curation)
            }

            # Add trace data if available
            if C_data is not None:
                mask_dict["inferred_trace"] = C_data.sel(unit_id=unit_id).values
            if S_data is not None:
                mask_dict["spikes"] = S_data.sel(unit_id=unit_id).values

            masks.append(mask_dict)

        return masks

    @property
    def num_units(self):
        """Return number of detected units."""
        if self._A is not None:
            return len(self._A.coords["unit_id"])
        return 0


def get_loader_result(key, table, full_output_dir=None) -> tuple:
    """Retrieve the loaded processed imaging results from the loader (e.g. caiman, etc.)

    Args:
        key (dict): the `key` to one entry of ProcessingTask.
        table (str): the class defining the table to retrieve
            the loaded results from (e.g. ProcessingTask).

    Returns:
        method, loaded_output (tuple): method string and loader object with results (e.g. caiman.CaImAn, etc.)
    """

    if full_output_dir is None:
        output_dir = (ProcessingParamSet * table & key).fetch1("processing_output_dir")
        output_dir = find_full_path(get_processed_root_data_dir(), output_dir)
    else:
        output_dir = full_output_dir
    method = (ProcessingParamSet * table & key).fetch1("processing_method")
    if method == "caiman":
        from element_interface import caiman_loader

        loaded_output = caiman_loader.CaImAn(output_dir)
    elif method == "minian":
        loaded_output = MinianLoader(output_dir)
    else:
        raise NotImplementedError("Unknown/unimplemented method: {}".format(method))

    return method, loaded_output


def _run_minian_in_container(
    avi_files: list,
    output_dir: pathlib.Path,
    params: dict,
    docker_image: str = MINIAN_DOCKER_IMAGE,
    timeout: int = MINIAN_DOCKER_TIMEOUT,
):
    import json
    import subprocess

    # ── Resolve input directory ──────────────────────────────────────────
    input_dir = pathlib.Path(avi_files[0]).parent.resolve()

    for avi in avi_files:
        if pathlib.Path(avi).parent.resolve() != input_dir:
            raise ValueError(
                f"All AVI files must be in the same directory. "
                f"Found files in {input_dir} and {pathlib.Path(avi).parent}"
            )

    # ── Prepare output directory ─────────────────────────────────────────
    output_dir = pathlib.Path(output_dir).resolve()
    os.makedirs(output_dir, exist_ok=True)

    # ── Prepare params (under output_dir so it's on a shared volume) ─────
    params_dir = output_dir / ".params"
    os.makedirs(params_dir, exist_ok=True)
    params_file = params_dir / "params.json"

    def _serialize_params(obj):
        if isinstance(obj, tuple):
            return list(obj)
        if isinstance(obj, dict):
            return {k: _serialize_params(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_serialize_params(v) for v in obj]
        return obj
    
    serializable_params = _serialize_params(params)

    config = {
        "input_dir": "/data/input",
        "output_dir": "/data/output",
        "params": serializable_params,
    }

    with open(params_file, "w") as f:
        json.dump(config, f, indent=2)

    logger.info(f"Params written to {params_file}")

    # ── Translate container paths → host paths ───────────────────────────
    # The host Docker daemon resolves bind-mount paths on the HOST
    # filesystem, not inside this worker container. We use env vars
    # set by docker-compose to map container paths to host paths.
    host_s3_root = os.getenv("HOST_S3_ROOT")
    host_outbox = os.getenv("HOST_OUTBOX")

    def _to_host_path(container_path, container_prefix, host_prefix):
        s = str(container_path)
        if host_prefix and s.startswith(container_prefix):
            return host_prefix + s[len(container_prefix):]
        return s  # fallback: use as-is (works if running directly on host)

    host_input_dir = _to_host_path(input_dir, "/home/jovyan/s3", host_s3_root)
    host_output_dir = _to_host_path(output_dir, "/home/jovyan/efs/outbox", host_outbox)
    host_params_dir = _to_host_path(params_dir, "/home/jovyan/efs/outbox", host_outbox)

    # ── Build docker run command ─────────────────────────────────────────
    docker_cmd = [
        "docker", "run", "--rm",
        "-v", f"{host_input_dir}:/data/input:ro",
        "-v", f"{host_output_dir}:/data/output",
        "-v", f"{host_params_dir}:/data/params:ro",
    ]

    # Pass through resource configuration and Dask env vars
    _passthrough_env_vars = [
        "MINIAN_NWORKERS",
        "MINIAN_MEMORY_LIMIT",
        # Dask distributed config (set via docker-compose with defaults)
        "DASK_DISTRIBUTED__SCHEDULER__WORKER_TTL",
        "DASK_DISTRIBUTED__COMM__TIMEOUTS__TCP",
        "DASK_DISTRIBUTED__COMM__TIMEOUTS__CONNECT",
        "DASK_DISTRIBUTED__SCHEDULER__WORK_STEALING",
        "DASK_DISTRIBUTED__WORKER__MEMORY__TARGET",
        "DASK_DISTRIBUTED__WORKER__MEMORY__SPILL",
        "DASK_DISTRIBUTED__WORKER__MEMORY__PAUSE",
        "DASK_DISTRIBUTED__WORKER__MEMORY__TERMINATE",
    ]
    for env_var in _passthrough_env_vars:
        val = os.getenv(env_var)
        if val:
            docker_cmd.extend(["-e", f"{env_var}={val}"])

    # Memory allocator tuning (reduces heap fragmentation under heavy load)
    docker_cmd.extend(["-e", "MALLOC_TRIM_THRESHOLD_=131072"])
    docker_cmd.extend(["-e", "MALLOC_MMAP_THRESHOLD_=131072"])

    # Memory limit for the container itself (if set)
    container_memory = os.getenv("MINIAN_CONTAINER_MEM_LIMIT")
    if container_memory:
        docker_cmd.extend(["--memory", container_memory])

    # CPU limit (if set)
    container_cpus = os.getenv("MINIAN_CONTAINER_CPUS")
    if container_cpus:
        docker_cmd.extend(["--cpus", container_cpus])

    # SHM size — Dask workers need shared memory for inter-process comms
    docker_cmd.extend(["--shm-size", os.getenv("MINIAN_SHM_SIZE", "8g")])

    # The image
    docker_cmd.append(docker_image)

    logger.info(f"Docker command: {' '.join(docker_cmd)}")

    # ── Pull image if not present ────────────────────────────────────────
    try:
        subprocess.run(
            ["docker", "image", "inspect", docker_image],
            capture_output=True,
            check=True,
        )
        logger.info(f"Docker image {docker_image} found locally")
    except subprocess.CalledProcessError:
        logger.info(f"Pulling Docker image {docker_image}...")
        subprocess.run(
            ["docker", "pull", docker_image],
            check=True,
            timeout=600,  # 10 min pull timeout
        )

    # ── Run the container ────────────────────────────────────────────────
    logger.info("Starting minian container...")
    start_time = time.time()

    process = subprocess.Popen(
        docker_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,  # Line-buffered
    )

    # Stream container logs to the pipeline logger in real time
    try:
        for line in process.stdout:
            line = line.rstrip()
            if line:
                logger.info(f"[minian-container] {line}")

        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        logger.error(
            f"Minian container timed out after {timeout}s. Killing..."
        )
        process.kill()
        process.wait()
        raise RuntimeError(
            f"Minian container exceeded timeout of {timeout} seconds"
        )

    elapsed = time.time() - start_time
    logger.info(
        f"Container exited with code {process.returncode} "
        f"after {elapsed:.1f}s ({elapsed/3600:.1f}h)"
    )

    # ── Check results ────────────────────────────────────────────────────
    status_file = output_dir / "status.json"

    if process.returncode != 0:
        error_msg = f"Minian container exited with code {process.returncode}"
        if status_file.exists():
            with open(status_file) as f:
                status = json.load(f)
            error_msg += f": {status.get('message', 'unknown error')}"
        raise RuntimeError(error_msg)

    if not status_file.exists():
        raise RuntimeError(
            "Minian container exited successfully but no status.json found"
        )

    with open(status_file) as f:
        status = json.load(f)

    if status.get("status") != "success":
        raise RuntimeError(
            f"Minian container reported failure: {status.get('message', 'unknown')}"
        )

    logger.info(
        f"Minian processing complete: {status.get('n_units', '?')} units detected"
    )

    # ── Cleanup temp params dir ──────────────────────────────────────────
    import shutil

    shutil.rmtree(params_dir, ignore_errors=True)

    return status
