import datajoint as dj
import numpy as np
import pathlib
from datetime import datetime
import importlib
import inspect
import cv2
import json
import csv
from element_interface.utils import dict_to_uuid, find_full_path, find_root_directory

schema = dj.Schema()

_linking_module = None


def activate(
    miniscope_schema_name,
    *,
    create_schema=True,
    create_tables=True,
    linking_module=None,
):
    """
    activate(miniscope_schema_name, *, create_schema=True, create_tables=True,
             linking_module=None)
        :param miniscope_schema_name: schema name on the database server to activate the
                                      `miniscope` module
        :param create_schema: when True (default), create schema in the database if it
                              does not yet exist.
        :param create_tables: when True (default), create tables in the database if they
                              do not yet exist.
        :param linking_module: a module name or a module containing the
         required dependencies to activate the `miniscope` module:
            Upstream tables:
                + Session: parent table to Recording,
                           typically identifying a recording session
                + Equipment: Reference table for Recording,
                             specifying the equipment used for the acquisition
                + AnatomicalLocation: Reference table for RecordingLocation, specifying
                the brain location where the recording is acquired
            Functions:
                + get_miniscope_root_data_dir() -> list
                    Retrieve the root data directory
                    Contains all subject/sessions data
                    :return: a string for full path to the root data directory
                + get_session_directory(session_key: dict) -> str
                    Retrieve the session directory containing the recorded data for a
                    given session
                    :param session_key: a dictionary of one session `key`
                    :return: a string for full path to the session directory
                + get_processed_root_data_dir() -> str:
                    Retrieves the root directory for all processed data
                    :return: a string for full path to the root dir for processed data
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


# Functions required by the element-miniscope  -----------------------------------------


def get_miniscope_root_data_dir() -> list:
    """
    All data paths, directories in DataJoint Elements are recommended to be stored as
    relative paths (posix format), with respect to some user-configured "root"
    directory, which varies from machine to machine (e.g. different mounted drive
    locations).

    get_miniscope_root_data_dir() -> list
        This user-provided function retrieves the possible root data directories
         containing the miniscope data for all subjects and sessions
         (e.g. acquired Miniscope-DAQ-V4 raw files,
         output files from processing routines, etc.).

        :return: a string for full path to the root data directory,
         or list of strings for possible root data directories
    """

    root_directories = _linking_module.get_miniscope_root_data_dir()
    if isinstance(root_directories, (str, pathlib.Path)):
        root_directories = [root_directories]

    if hasattr(_linking_module, "get_processed_root_data_dir"):
        root_directories.append(_linking_module.get_processed_root_data_dir())

    return root_directories


def get_session_directory(session_key: dict) -> str:
    """
    get_session_directory(session_key: dict) -> str
        Retrieve the session directory containing the
         recorded data for a given session
        :param session_key: a dictionary of one session `key`
        :return: a string for relative or full path to the session directory
    """
    return _linking_module.get_session_directory(session_key)


def get_processed_root_data_dir() -> str:
    """
    get_processed_root_data_dir() -> str:
        Retrieves the root directory for all processed data
        :return: a string for full path to the root directory for processed data
    """

    if hasattr(_linking_module, "get_processed_root_data_dir"):
        return _linking_module.get_processed_root_data_dir()
    else:
        return get_miniscope_root_data_dir()[0]


# Experiment and analysis meta information -------------------------------------


@schema
class AcquisitionSoftware(dj.Lookup):
    definition = """
    acquisition_software: varchar(24)
    """
    contents = zip(["Miniscope-DAQ-V3", "Miniscope-DAQ-V4"])


@schema
class Channel(dj.Lookup):
    definition = """
    channel     : tinyint  # 0-based indexing
    """
    contents = zip(range(5))


@schema
class Recording(dj.Manual):
    definition = """
    -> Session
    recording_id: int
    ---
    -> Equipment
    -> AcquisitionSoftware
    recording_directory: varchar(255)  # relative to root data directory
    recording_notes='' : varchar(4095) # free-notes
    """


@schema
class RecordingLocation(dj.Manual):
    definition = """
    # Brain location where this miniscope recording is acquired
    -> Recording
    ---
    -> AnatomicalLocation
    """


@schema
class RecordingInfo(dj.Imported):
    definition = """
    # Store metadata about recording
    -> Recording
    ---
    nchannels            : tinyint   # number of channels
    nframes              : int       # number of recorded frames
    px_height=null       : smallint  # height in pixels
    px_width=null        : smallint  # width in pixels
    um_height=null       : float     # height in microns
    um_width=null        : float     # width in microns
    fps                  : float     # (Hz) frames per second
    gain=null            : float     # recording gain
    spatial_downsample=1 : tinyint   # e.g. 1, 2, 4, 8. 1 for no downsampling
    led_power            : float     # LED power used in the given recording
    time_stamps          : longblob  # time stamps of each frame
    recording_datetime=null   : datetime  # datetime of the recording
    recording_duration=null   : float     # (seconds) duration of the recording
    """

    class File(dj.Part):
        definition = """
        -> master
        file_id : smallint unsigned
        ---
        file_path: varchar(255)      # relative to root data directory
        """

    def make(self, key):

        # Search recording directory for miniscope raw files
        acquisition_software, recording_directory = (Recording & key).fetch1(
            "acquisition_software", "recording_directory"
        )

        recording_path = find_full_path(
            get_miniscope_root_data_dir(), recording_directory
        )

        recording_filepaths = [
            file_path.as_posix() for file_path in recording_path.glob("*.avi")
        ]

        if not recording_filepaths:
            raise FileNotFoundError(f"No .avi files found in " f"{recording_directory}")

        if acquisition_software == "Miniscope-DAQ-V3":
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

        elif acquisition_software == "Miniscope-DAQ-V4":
            recording_metadata = list(recording_path.glob("metaData.json"))[0]
            recording_timestamps = list(recording_path.glob("timeStamps.csv"))[0]

            if not recording_metadata.exists():
                raise FileNotFoundError(
                    f"No .json file found in " f"{recording_directory}"
                )
            if not recording_timestamps.exists():
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
            gain = metadata["gain"]
            spatial_downsample = 1  # Assumes no spatial downsampling
            led_power = metadata["led0"]
            time_stamps = np.array(
                [list(map(int, time_stamps[i])) for i in range(1, len(time_stamps))]
            )
        else:
            raise NotImplementedError(
                f"Loading routine not implemented for {acquisition_software}"
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
                gain=gain,
                spatial_downsample=spatial_downsample,
                led_power=led_power,
                time_stamps=time_stamps,
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


# Trigger a processing routine -------------------------------------------------


@schema
class ProcessingMethod(dj.Lookup):
    definition = """
    # Method, package, analysis software used for processing of miniscope data 
    # (e.g. CaImAn, etc.)
    processing_method: char(16)
    ---
    processing_method_desc='': varchar(1000)
    """

    contents = [("caiman", "caiman analysis suite")]


@schema
class ProcessingParamSet(dj.Lookup):
    definition = """
    # Parameter set used for processing of miniscope data
    paramset_id:  smallint
    ---
    -> ProcessingMethod
    paramset_desc: varchar(128)
    param_set_hash: uuid
    unique index (param_set_hash)
    params: longblob  # dictionary of all applicable parameters
    """

    @classmethod
    def insert_new_params(
        cls,
        processing_method: str,
        paramset_id: int,
        paramset_desc: str,
        params: dict,
        processing_method_desc: str = "",
    ):
        ProcessingMethod.insert1(
            {"processing_method": processing_method}, skip_duplicates=True
        )
        param_dict = {
            "processing_method": processing_method,
            "paramset_id": paramset_id,
            "paramset_desc": paramset_desc,
            "params": params,
            "param_set_hash": dict_to_uuid(params),
        }
        q_param = cls & {"param_set_hash": param_dict["param_set_hash"]}

        if q_param:  # If the specified param-set already exists
            pname = q_param.fetch1("paramset_id")
            if pname == paramset_id:  # If the existed set has the same name: job done
                return
            else:  # If not same name: human error, try adding with different name
                raise dj.DataJointError(
                    "The specified param-set already exists - name: {}".format(pname)
                )
        else:
            cls.insert1(param_dict)


@schema
class ProcessingTask(dj.Manual):
    definition = """
    # Manual table marking a processing task to be triggered or manually processed
    -> RecordingInfo
    -> ProcessingParamSet
    ---
    processing_output_dir : varchar(255)    # relative to the root data directory
    task_mode='load'      : enum('load', 'trigger') # 'load': load existing results
                                                    # 'trigger': trigger procedure
    """


@schema
class Processing(dj.Computed):
    definition = """
    -> ProcessingTask
    ---
    processing_time     : datetime  # generation time of processed, segmented results
    package_version=''  : varchar(16)
    """

    def make(self, key):
        task_mode = (ProcessingTask & key).fetch1("task_mode")

        output_dir = (ProcessingTask & key).fetch1("processing_output_dir")
        output_dir = find_full_path(get_miniscope_root_data_dir(), output_dir)

        if task_mode == "load":
            method, loaded_result = get_loader_result(key, ProcessingTask)
            if method == "caiman":
                loaded_caiman = loaded_result
                key = {**key, "processing_time": loaded_caiman.creation_time}
            else:
                raise NotImplementedError(
                    f"Loading of {method} data is not yet" f"supported"
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

                input_hash = dict_to_uuid(dict(**key, **params))
                input_hash_fp = output_dir / f".{input_hash }.json"

                if not input_hash_fp.exists():
                    start_time = datetime.utcnow()
                    run_caiman(
                        file_paths=avi_files,
                        parameters=params,
                        sampling_rate=sampling_rate,
                        output_dir=output_dir.as_posix(),
                        is3D=False,
                    )
                    completion_time = datetime.utcnow()
                    with open(input_hash_fp, "w") as f:
                        json.dump(
                            {
                                "start_time": start_time,
                                "completion_time": completion_time,
                                "duration": (
                                    completion_time - start_time
                                ).total_seconds(),
                            },
                            f,
                            default=str,
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


@schema
class Curation(dj.Manual):
    definition = """
    # Different rounds of curation performed on the processing results of the data 
    # (no-curation can also be included here)
    -> Processing
    curation_id: int
    ---
    curation_time: datetime             # time of generation of these curated results 
    curation_output_dir: varchar(255)   # output directory of the curated results, 
                                        # relative to root data directory
    manual_curation: bool               # has manual curation been performed?
    curation_note='': varchar(2000)  
    """

    def create1_from_processing_task(self, key, is_curated=False, curation_note=""):
        """
        Given a "ProcessingTask", create a new corresponding "Curation"
        """
        if key not in Processing():
            raise ValueError(
                f"No corresponding entry in Processing available for: "
                f"{key}; run `Processing.populate(key)`"
            )

        output_dir = (ProcessingTask & key).fetch1("processing_output_dir")
        method, imaging_dataset = get_loader_result(key, ProcessingTask)

        if method == "caiman":
            caiman_dataset = imaging_dataset
            curation_time = caiman_dataset.creation_time
        else:
            raise NotImplementedError("Unknown method: {}".format(method))

        # Synthesize curation_id
        curation_id = (
            dj.U().aggr(self & key, n="ifnull(max(curation_id)+1,1)").fetch1("n")
        )
        self.insert1(
            {
                **key,
                "curation_id": curation_id,
                "curation_time": curation_time,
                "curation_output_dir": output_dir,
                "manual_curation": is_curated,
                "curation_note": curation_note,
            }
        )


# Motion Correction --------------------------------------------------------------------


@schema
class MotionCorrection(dj.Imported):
    definition = """
    -> Curation
    ---
    -> Channel.proj(motion_correct_channel='channel') # channel used for 
                                                      # motion correction
    """

    class RigidMotionCorrection(dj.Part):
        definition = """
        -> master
        ---
        outlier_frames=null : longblob  # mask with true for frames with outlier shifts 
                                        # (already corrected)
        y_shifts            : longblob  # (pixels) y motion correction shifts
        x_shifts            : longblob  # (pixels) x motion correction shifts
        y_std               : float     # (pixels) standard deviation of 
                                        # y shifts across all frames
        x_std               : float     # (pixels) standard deviation of 
                                        # x shifts across all frames
        """

    class NonRigidMotionCorrection(dj.Part):
        """
        Piece-wise rigid motion correction
        """

        definition = """
        -> master
        ---
        outlier_frames=null             : longblob  # mask with true for frames with 
                                                    # outlier shifts (already corrected)
        block_height                    : int       # (pixels)
        block_width                     : int       # (pixels)
        block_count_y                   : int       # number of blocks tiled in the 
                                                    # y direction
        block_count_x                   : int       # number of blocks tiled in the 
                                                    # x direction
        """

    class Block(dj.Part):
        definition = """  # FOV-tiled blocks used for non-rigid motion correction
        -> master.NonRigidMotionCorrection
        block_id        : int
        ---
        block_y         : longblob  # (y_start, y_end) in pixel of this block
        block_x         : longblob  # (x_start, x_end) in pixel of this block
        y_shifts        : longblob  # (pixels) y motion correction shifts for 
                                    # every frame
        x_shifts        : longblob  # (pixels) x motion correction shifts for 
                                    # every frame
        y_std           : float     # (pixels) standard deviation of y shifts 
                                    # across all frames
        x_std           : float     # (pixels) standard deviation of x shifts 
                                    # across all frames
        """

    class Summary(dj.Part):
        definition = """ # summary images for each field and channel after corrections
        -> master
        ---
        ref_image=null          : longblob  # image used as alignment template
        average_image           : longblob  # mean of registered frames
        correlation_image=null  : longblob  # correlation map 
                                            # (computed during cell detection)
        max_proj_image=null     : longblob  # max of registered frames
        """

    def make(self, key):
        method, loaded_result = get_loader_result(key, ProcessingTask)

        if method == "caiman":
            loaded_caiman = loaded_result

            self.insert1(
                {**key, "motion_correct_channel": loaded_caiman.alignment_channel}
            )

            # -- rigid motion correction --
            if not loaded_caiman.params.motion["pw_rigid"]:
                rigid_correction = {
                    **key,
                    "x_shifts": loaded_caiman.motion_correction["shifts_rig"][:, 0],
                    "y_shifts": loaded_caiman.motion_correction["shifts_rig"][:, 1],
                    "x_std": np.nanstd(
                        loaded_caiman.motion_correction["shifts_rig"][:, 0]
                    ),
                    "y_std": np.nanstd(
                        loaded_caiman.motion_correction["shifts_rig"][:, 1]
                    ),
                    "outlier_frames": None,
                }

                self.RigidMotionCorrection.insert1(rigid_correction)

            # -- non-rigid motion correction --
            else:
                nonrigid_correction = {
                    **key,
                    "block_height": (
                        loaded_caiman.params.motion["strides"][0]
                        + loaded_caiman.params.motion["overlaps"][0]
                    ),
                    "block_width": (
                        loaded_caiman.params.motion["strides"][1]
                        + loaded_caiman.params.motion["overlaps"][1]
                    ),
                    "block_count_x": len(
                        set(loaded_caiman.motion_correction["coord_shifts_els"][:, 0])
                    ),
                    "block_count_y": len(
                        set(loaded_caiman.motion_correction["coord_shifts_els"][:, 2])
                    ),
                    "outlier_frames": None,
                }

                nonrigid_blocks = []
                for b_id in range(
                    len(loaded_caiman.motion_correction["x_shifts_els"][0, :])
                ):
                    nonrigid_blocks.append(
                        {
                            **key,
                            "block_id": b_id,
                            "block_x": np.arange(
                                *loaded_caiman.motion_correction["coord_shifts_els"][
                                    b_id, 0:2
                                ]
                            ),
                            "block_y": np.arange(
                                *loaded_caiman.motion_correction["coord_shifts_els"][
                                    b_id, 2:4
                                ]
                            ),
                            "x_shifts": loaded_caiman.motion_correction["x_shifts_els"][
                                :, b_id
                            ],
                            "y_shifts": loaded_caiman.motion_correction["y_shifts_els"][
                                :, b_id
                            ],
                            "x_std": np.nanstd(
                                loaded_caiman.motion_correction["x_shifts_els"][:, b_id]
                            ),
                            "y_std": np.nanstd(
                                loaded_caiman.motion_correction["y_shifts_els"][:, b_id]
                            ),
                        }
                    )

                self.NonRigidMotionCorrection.insert1(nonrigid_correction)
                self.Block.insert(nonrigid_blocks)

            # -- summary images --
            summary_images = {
                **key,
                "ref_image": loaded_caiman.motion_correction["reference_image"][...][
                    np.newaxis, ...
                ],
                "average_image": loaded_caiman.motion_correction["average_image"][...][
                    np.newaxis, ...
                ],
                "correlation_image": loaded_caiman.motion_correction[
                    "correlation_image"
                ][...][np.newaxis, ...],
                "max_proj_image": loaded_caiman.motion_correction["max_image"][...][
                    np.newaxis, ...
                ],
            }

            self.Summary.insert1(summary_images)

        else:
            raise NotImplementedError("Unknown/unimplemented method: {}".format(method))


# Segmentation -------------------------------------------------------------------------


@schema
class Segmentation(dj.Computed):
    definition = """ # Different mask segmentations.
    -> Curation
    """

    class Mask(dj.Part):
        definition = """ # A mask produced by segmentation.
        -> master
        mask_id              : smallint
        ---
        -> Channel.proj(segmentation_channel='channel')  # channel used for segmentation
        mask_npix            : int       # number of pixels in this mask
        mask_center_x=null   : int       # (pixels) center x coordinate
        mask_center_y=null   : int       # (pixels) center y coordinate
        mask_xpix=null       : longblob  # (pixels) x coordinates
        mask_ypix=null       : longblob  # (pixels) y coordinates
        mask_weights         : longblob  # weights of the mask at the indices above
        """

    def make(self, key):
        method, loaded_result = get_loader_result(key, Curation)

        if method == "caiman":
            loaded_caiman = loaded_result

            # infer `segmentation_channel` from `params` if available,
            # else from caiman loader
            params = (ProcessingParamSet * ProcessingTask & key).fetch1("params")
            segmentation_channel = params.get(
                "segmentation_channel", loaded_caiman.segmentation_channel
            )

            masks, cells = [], []
            for mask in loaded_caiman.masks:
                masks.append(
                    {
                        **key,
                        "segmentation_channel": segmentation_channel,
                        "mask_id": mask["mask_id"],
                        "mask_npix": mask["mask_npix"],
                        "mask_center_x": mask["mask_center_x"],
                        "mask_center_y": mask["mask_center_y"],
                        "mask_xpix": mask["mask_xpix"],
                        "mask_ypix": mask["mask_ypix"],
                        "mask_weights": mask["mask_weights"],
                    }
                )

                if loaded_caiman.cnmf.estimates.idx_components is not None:
                    if mask["mask_id"] in loaded_caiman.cnmf.estimates.idx_components:
                        cells.append(
                            {
                                **key,
                                "mask_classification_method": "caiman_default_classifier",
                                "mask_id": mask["mask_id"],
                                "mask_type": "soma",
                            }
                        )

            self.insert1(key)
            self.Mask.insert(masks, ignore_extra_fields=True)

            if cells:
                MaskClassification.insert1(
                    {**key, "mask_classification_method": "caiman_default_classifier"},
                    allow_direct_insert=True,
                )
                MaskClassification.MaskType.insert(
                    cells, ignore_extra_fields=True, allow_direct_insert=True
                )

        else:
            raise NotImplementedError(f"Unknown/unimplemented method: {method}")


@schema
class MaskType(dj.Lookup):
    definition = """ # Possible classifications for a segmented mask
    mask_type        : varchar(16)
    """

    contents = zip(["soma", "axon", "dendrite", "neuropil", "artefact", "unknown"])


@schema
class MaskClassificationMethod(dj.Lookup):
    definition = """
    mask_classification_method: varchar(48)
    """

    contents = zip(["caiman_default_classifier"])


@schema
class MaskClassification(dj.Computed):
    definition = """
    -> Segmentation
    -> MaskClassificationMethod
    """

    class MaskType(dj.Part):
        definition = """
        -> master
        -> Segmentation.Mask
        ---
        -> MaskType
        confidence=null: float
        """

    def make(self, key):
        pass


# Fluorescence & Activity Traces -------------------------------------------------------


@schema
class Fluorescence(dj.Computed):
    definition = """  # fluorescence traces before spike extraction or filtering
    -> Segmentation
    """

    class Trace(dj.Part):
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
        method, loaded_result = get_loader_result(key, Curation)

        if method == "caiman":
            loaded_caiman = loaded_result

            # infer `segmentation_channel` from `params` if available,
            # else from caiman loader
            params = (ProcessingParamSet * ProcessingTask & key).fetch1("params")
            segmentation_channel = params.get(
                "segmentation_channel", loaded_caiman.segmentation_channel
            )

            self.insert1(key)
            self.Trace.insert(
                [
                    {
                        **key,
                        "mask_id": mask["mask_id"],
                        "fluorescence_channel": segmentation_channel,
                        "fluorescence": mask["inferred_trace"],
                    }
                    for mask in loaded_caiman.masks
                ]
            )

        else:
            raise NotImplementedError("Unknown/unimplemented method: {}".format(method))


@schema
class ActivityExtractionMethod(dj.Lookup):
    definition = """
    extraction_method: varchar(200)
    """

    contents = zip(["caiman_deconvolution", "caiman_dff"])


@schema
class Activity(dj.Computed):
    definition = """
    # inferred neural activity from fluorescence trace - e.g. dff, spikes
    -> Fluorescence
    -> ActivityExtractionMethod
    """

    class Trace(dj.Part):
        definition = """
        -> master
        -> Fluorescence.Trace
        ---
        activity_trace: longblob
        """

    @property
    def key_source(self):
        caiman_key_source = (
            Fluorescence
            * ActivityExtractionMethod
            * ProcessingParamSet.proj("processing_method")
            & 'processing_method = "caiman"'
            & 'extraction_method LIKE "caiman%"'
        )

        return caiman_key_source.proj()

    def make(self, key):
        method, loaded_result = get_loader_result(key, Curation)

        if method == "caiman":
            loaded_caiman = loaded_result

            if key["extraction_method"] in ("caiman_deconvolution", "caiman_dff"):
                attr_mapper = {"caiman_deconvolution": "spikes", "caiman_dff": "dff"}

                # infer `segmentation_channel` from `params` if available,
                # else from caiman loader
                params = (ProcessingParamSet * ProcessingTask & key).fetch1("params")
                segmentation_channel = params.get(
                    "segmentation_channel", loaded_caiman.segmentation_channel
                )

                self.insert1(key)
                self.Trace.insert(
                    [
                        {
                            **key,
                            "mask_id": mask["mask_id"],
                            "fluorescence_channel": segmentation_channel,
                            "activity_trace": mask[
                                attr_mapper[key["extraction_method"]]
                            ],
                        }
                        for mask in loaded_caiman.masks
                    ]
                )

        else:
            raise NotImplementedError("Unknown/unimplemented method: {}".format(method))


# Helper Functions ---------------------------------------------------------------------

_table_attribute_mapper = {
    "ProcessingTask": "processing_output_dir",
    "Curation": "curation_output_dir",
}


def get_loader_result(key, table):
    """
    Retrieve the loaded processed imaging results from the loader (e.g. caiman, etc.)
        :param key: the `key` to one entry of ProcessingTask or Curation
        :param table: the class defining the table to retrieve
         the loaded results from (e.g. ProcessingTask, Curation)
        :return: a loader object of the loaded results
         (e.g. caiman.CaImAn, etc.)
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


def populate_all(display_progress=True, reserve_jobs=False, suppress_errors=False):

    populate_settings = {
        "display_progress": display_progress,
        "reserve_jobs": reserve_jobs,
        "suppress_errors": suppress_errors,
    }

    RecordingInfo.populate(**populate_settings)

    Processing.populate(**populate_settings)

    MotionCorrection.populate(**populate_settings)

    Segmentation.populate(**populate_settings)

    MaskClassification.populate(**populate_settings)

    Fluorescence.populate(**populate_settings)

    Activity.populate(**populate_settings)
