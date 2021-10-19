import datajoint as dj
import numpy as np
import pathlib
from datetime import datetime
import importlib
import inspect
import cv2
import json
import csv
from element_data_loader.utils import dict_to_uuid, find_full_path, find_root_directory

schema = dj.schema()

_linking_module = None


def activate(miniscope_schema_name, *,
             create_schema=True, create_tables=True, linking_module=None):
    """
    activate(miniscope_schema_name, *, create_schema=True, create_tables=True, linking_module=None)
        :param miniscope_schema_name: schema name on the database server to activate the `miniscope` module
        :param create_schema: when True (default), create schema in the database if it does not yet exist.
        :param create_tables: when True (default), create tables in the database if they do not yet exist.
        :param linking_module: a module name or a module containing the
         required dependencies to activate the `miniscope` module:
            Upstream tables:
                + Session: parent table to Recording, 
                           typically identifying a recording session
                + Equipment: Reference table for Recording, 
                             specifying the equipment used for the acquisition
            Functions:
                + get_miniscope_root_data_dir() -> list
                    Retrieve the root data directory
                    Contains all subject/sessions data
                    :return: a string for full path to the root data directory
    """

    if isinstance(linking_module, str):
        linking_module = importlib.import_module(linking_module)
    assert inspect.ismodule(linking_module),\
        "The argument 'dependency' must be a module's name or a module"

    global _linking_module
    _linking_module = linking_module

    schema.activate(miniscope_schema_name, create_schema=create_schema,
                    create_tables=create_tables, add_objects=_linking_module.__dict__)


# Functions required by the element-miniscope  ---------------------------------

def get_miniscope_root_data_dir() -> list:
    """
    get_miniscope_root_data_dir() -> list
        Retrieve the root data directory
        Containing the raw ephys recording files for all subject/sessions.
        :return: a string for full path to the root data directory,
                 or list of strings for possible root data directories
    """
    return _linking_module.get_miniscope_root_data_dir()


# Experiment and analysis meta information -------------------------------------

@schema
class AcquisitionSoftware(dj.Lookup):
    definition = """
    acquisition_software: varchar(24)
    """
    contents = zip([
        'Miniscope-DAQ-V3',
        'Miniscope-DAQ-V4',
        'Inscopix nVoke'])


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
    -> Location
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
    """

    class File(dj.Part):
        definition = """
        -> master
        recording_file_id : smallint unsigned
        ---
        recording_file_path: varchar(255)      # relative to root data directory
        """

    def make(self, key):

        # Search recording directory for miniscope raw files
        acquisition_software, recording_directory = \
            (Recording & key).fetch1('acquisition_software' ,
                                     'recording_directory')
        
        recording_path = find_full_path(get_miniscope_root_data_dir(),
                                        recording_directory)

        recording_filepaths = [file_path.as_posix() for file_path 
                                            in recording_path.glob('*.avi')]
        recording_metadata = list(recording_path.glob('*.json'))[0]
        recording_timestamps = list(recording_path.glob('*.csv'))[0]

        if not recording_filepaths:
            raise FileNotFoundError(f'No .avi files found in {recording_directory}')
        elif not recording_metadata.exists():
            raise FileNotFoundError(f'No .json file found in {recording_directory}')
        elif not recording_timestamps.exists():
            raise FileNotFoundError(f'No .csv file found in {recording_directory}')
        
        if acquisition_software == 'Miniscope-DAQ-V3':
            # Parse image dimension and frame rate
            video = cv2.VideoCapture(recording_filepaths[0])
            fps = video.get(cv2.CAP_PROP_FPS) # TODO: Verify this method extracts correct value
            _, frame = video.read()
            frame_size = np.shape(frame)

            # Parse number of frames from timestamp.dat file
            with open(recording_filepaths[-1]) as f:
                next(f)
                nframes = sum(1 for line in f if int(line[0]) == 0)

            nchannels=1
            px_height=frame_size[0]
            px_width=frame_size[1]

        elif acquisition_software == 'Miniscope-DAQ-V4':
            with open(recording_metadata.as_posix()) as f:
                metadata = json.loads(f.read())
        
            with open(recording_timestamps, newline= '') as f:
                time_stamps = list(csv.reader(f, delimiter=','))

            time_stamps = np.array([list(map(int, time_stamps[i]))
                                       for i in range(1,len(time_stamps))])
            nchannels = 1
            nframes = len(time_stamps)
            px_height = metadata['ROI']['height']
            px_width = metadata['ROI']['width']
            fps = int(metadata['frameRate'].replace('FPS',''))
            gain = metadata['gain']
            spatial_downsample = 1
            led_power = metadata['led0']

        else:
            raise NotImplementedError(
                f'Loading routine not implemented for {acquisition_software}'
                ' acquisition software')

        # Insert in RecordingInfo
        self.insert1(dict(key,
                          nchannels=nchannels,
                          nframes=nframes,
                          px_height=px_height,
                          px_width=px_width,
                          fps=fps,
                        #   um_height=0,
                        #   um_width=0,
                          gain=gain,
                          spatial_downsample=spatial_downsample,
                          led_power=led_power,
                          time_stamps=time_stamps))

        # Insert file(s)
        recording_files = [pathlib.Path(f).relative_to(
                                            find_root_directory(
                                                get_miniscope_root_data_dir(),
                                                f)).as_posix() 
                                for f in recording_filepaths]

        self.File.insert([{**key, 
                           'recording_file_id': i, 
                           'recording_file_path': f} 
                                for i, f in enumerate(recording_files)])


# Trigger a processing routine -------------------------------------------------

@schema
class ProcessingTask(dj.Manual):
    definition = """
    # Manual table marking a processing task to be triggered or manually processed
    -> RecordingInfo
    processing_task_idx     : smallint     # processing task
    ---
    -> MotionCorrectionParamSet
    -> SegmentationParamSet
    processing_motion_correction_output_dir : varchar(255)            # relative directory of motion relative to the root data directory
    processing_segmentation_output_dir      : varchar(255)            # relative directory of segmentation result respect to root directory
    motion_correction_task_mode='load'      : enum('load', 'trigger') # 'load': load existing motion correction results, 'trigger': trigger motion correction procedure
    segmentation_task_mode='load'           : enum('load', 'trigger') # 'load': load existing segmentation results, 'trigger': trigger
    """

@schema
class Processing(dj.Computed):
    definition = """
    -> ProcessingTask
    ---
    processing_time     : datetime  # time of generation of this set of processed, segmented results
    package_version=''  : varchar(16)
    """

    def make(self, key):
        task_mode = (ProcessingTask & key).fetch1('task_mode')
        method, loaded_result = get_loader_result(key, ProcessingTask)

        if task_mode == 'load':
            if method == 'caiman':
                loaded_caiman = loaded_result
                key = {**key, 'processing_time': loaded_caiman.creation_time}
            elif method == 'mcgill_miniscope_analysis':
                loaded_miniscope_analysis = loaded_result
                key = {**key, 'processing_time': loaded_miniscope_analysis.creation_time}
            else:
                raise NotImplementedError('Unknown method: {}'.format(method))
        elif task_mode == 'trigger':
            raise NotImplementedError(f'Automatic triggering of {method} analysis'
                                      f' is not yet supported')
        else:
            raise ValueError(f'Unknown task mode: {task_mode}')

        self.insert1(key)


# Motion Correction ------------------------------------------------------------

@schema
class MotionCorrection(dj.Imported):
    definition = """
    -> Processing
    ---
    -> Channel.proj(motion_correct_channel='channel') # channel used for motion correction in this processing task
    """

    class RigidMotionCorrection(dj.Part):
        definition = """
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
        """
        Piece-wise rigid motion correction
        """
        definition = """
        -> master
        ---
        outlier_frames=null             : longblob      # mask with true for frames with outlier shifts (already corrected)
        block_height                    : int           # (pixels)
        block_width                     : int           # (pixels)
        block_depth                     : int           # (pixels)
        block_count_y                   : int           # number of blocks tiled in the y direction
        block_count_x                   : int           # number of blocks tiled in the x direction
        block_count_z                   : int           # number of blocks tiled in the z direction
        """

    class Block(dj.Part):
        definition = """  # FOV-tiled blocks used for non-rigid motion correction
        -> master.NonRigidMotionCorrection
        block_id        : int
        ---
        block_y         : longblob  # (y_start, y_end) in pixel of this block
        block_x         : longblob  # (x_start, x_end) in pixel of this block
        block_z         : longblob  # (z_start, z_end) in pixel of this block
        y_shifts        : longblob  # (pixels) y motion correction shifts for every frame
        x_shifts        : longblob  # (pixels) x motion correction shifts for every frame
        z_shifts=null   : longblob  # (pixels) x motion correction shifts for every frame
        y_std           : float     # (pixels) standard deviation of y shifts across all frames
        x_std           : float     # (pixels) standard deviation of x shifts across all frames
        z_std=null      : float     # (pixels) standard deviation of z shifts across all frames
        """

    class Summary(dj.Part):
        definition = """ # summary images for each field and channel after corrections
        -> master
        ---
        ref_image=null          : longblob  # image used as alignment template
        average_image           : longblob  # mean of registered frames
        correlation_image=null  : longblob  # correlation map (computed during cell detection)
        max_proj_image=null     : longblob  # max of registered frames
        """

    def make(self, key):
        method, loaded_result = get_loader_result(key, ProcessingTask)

        if method == 'caiman':
            loaded_caiman = loaded_result

            self.insert1({**key, 'motion_correct_channel': loaded_caiman.alignment_channel})

            is3D = loaded_caiman.params.motion['is3D']
            # -- rigid motion correction --
            if not loaded_caiman.params.motion['pw_rigid']:
                rigid_correction = {
                    **key,
                    'x_shifts': loaded_caiman.motion_correction['shifts_rig'][:, 0],
                    'y_shifts': loaded_caiman.motion_correction['shifts_rig'][:, 1],
                    'z_shifts': (loaded_caiman.motion_correction['shifts_rig'][:, 2]
                                 if is3D
                                 else np.full_like(
                        loaded_caiman.motion_correction['shifts_rig'][:, 0], 0)),
                    'x_std': np.nanstd(loaded_caiman.motion_correction['shifts_rig'][:, 0]),
                    'y_std': np.nanstd(loaded_caiman.motion_correction['shifts_rig'][:, 1]),
                    'z_std': (np.nanstd(loaded_caiman.motion_correction['shifts_rig'][:, 2])
                              if is3D
                              else np.nan),
                    'outlier_frames': None}

                self.RigidMotionCorrection.insert1(rigid_correction)

            # -- non-rigid motion correction --
            else:
                nonrigid_correction = {
                    **key,
                    'block_height': (loaded_caiman.params.motion['strides'][0]
                                     + loaded_caiman.params.motion['overlaps'][0]),
                    'block_width': (loaded_caiman.params.motion['strides'][1]
                                    + loaded_caiman.params.motion['overlaps'][1]),
                    'block_depth': (loaded_caiman.params.motion['strides'][2]
                                    + loaded_caiman.params.motion['overlaps'][2]
                                    if is3D else 1),
                    'block_count_x': len(
                        set(loaded_caiman.motion_correction['coord_shifts_els'][:, 0])),
                    'block_count_y': len(
                        set(loaded_caiman.motion_correction['coord_shifts_els'][:, 2])),
                    'block_count_z': (len(
                        set(loaded_caiman.motion_correction['coord_shifts_els'][:, 4]))
                                      if is3D else 1),
                    'outlier_frames': None}

                nonrigid_blocks = []
                for b_id in range(len(loaded_caiman.motion_correction['x_shifts_els'][0, :])):
                    nonrigid_blocks.append(
                        {**key, 'block_id': b_id,
                         'block_x': np.arange(*loaded_caiman.motion_correction[
                                                   'coord_shifts_els'][b_id, 0:2]),
                         'block_y': np.arange(*loaded_caiman.motion_correction[
                                                   'coord_shifts_els'][b_id, 2:4]),
                         'block_z': (np.arange(*loaded_caiman.motion_correction[
                                                    'coord_shifts_els'][b_id, 4:6])
                                     if is3D
                                     else np.full_like(
                             np.arange(*loaded_caiman.motion_correction[
                                            'coord_shifts_els'][b_id, 0:2]), 0)),
                         'x_shifts': loaded_caiman.motion_correction[
                                         'x_shifts_els'][:, b_id],
                         'y_shifts': loaded_caiman.motion_correction[
                                         'y_shifts_els'][:, b_id],
                         'z_shifts': (loaded_caiman.motion_correction[
                                          'z_shifts_els'][:, b_id]
                                      if is3D
                                      else np.full_like(
                             loaded_caiman.motion_correction['x_shifts_els'][:, b_id], 0)),
                         'x_std': np.nanstd(loaded_caiman.motion_correction[
                                                'x_shifts_els'][:, b_id]),
                         'y_std': np.nanstd(loaded_caiman.motion_correction[
                                                'y_shifts_els'][:, b_id]),
                         'z_std': (np.nanstd(loaded_caiman.motion_correction[
                                                 'z_shifts_els'][:, b_id])
                                   if is3D
                                   else np.nan)})

                self.NonRigidMotionCorrection.insert1(nonrigid_correction)
                self.Block.insert(nonrigid_blocks)

            # -- summary images --
            field_keys = (scan.ScanInfo.Field & key).fetch('KEY', order_by='field_z')
            summary_images = [
                {**key, **fkey, 'ref_image': ref_image,
                 'average_image': ave_img,
                 'correlation_image': corr_img,
                 'max_proj_image': max_img}
                for fkey, ref_image, ave_img, corr_img, max_img in zip(
                    field_keys,
                    loaded_caiman.motion_correction['reference_image'].transpose(2, 0, 1)
                    if is3D else loaded_caiman.motion_correction[
                        'reference_image'][...][np.newaxis, ...],
                    loaded_caiman.motion_correction['average_image'].transpose(2, 0, 1)
                    if is3D else loaded_caiman.motion_correction[
                        'average_image'][...][np.newaxis, ...],
                    loaded_caiman.motion_correction['correlation_image'].transpose(2, 0, 1)
                    if is3D else loaded_caiman.motion_correction[
                        'correlation_image'][...][np.newaxis, ...],
                    loaded_caiman.motion_correction['max_image'].transpose(2, 0, 1)
                    if is3D else loaded_caiman.motion_correction[
                        'max_image'][...][np.newaxis, ...])]
            self.Summary.insert(summary_images)

        elif method == 'mcgill_miniscope_analysis':
            loaded_miniscope_analysis = loaded_result

            # TODO: add motion correction and block data

            # -- summary images --
            mc_key = (scan.ScanInfo.Field * ProcessingTask & key).fetch1('KEY')
            summary_images = {**mc_key,
                                 'average_image': loaded_miniscope_analysis.average_image,
                                 'correlation_image': loaded_miniscope_analysis.correlation_image}

            self.insert1({**key, 'motion_correct_channel': loaded_miniscope_analysis.alignment_channel})
            self.Summary.insert1(summary_images)

        else:
            raise NotImplementedError('Unknown/unimplemented method: {}'.format(method))

# -------------- Segmentation --------------


@schema
class Segmentation(dj.Computed):
    definition = """ # Different mask segmentations.
    -> MotionCorrection
    """

    class Mask(dj.Part):
        definition = """ # A mask produced by segmentation.
        -> master
        mask                 : smallint
        ---
        -> Channel.proj(segmentation_channel='channel')  # channel used for segmentation
        mask_npix            : int       # number of pixels in this mask
        mask_center_x=null   : int       # center x coordinate in pixel                         # TODO: determine why some masks don't have information, thus null required
        mask_center_y=null   : int       # center y coordinate in pixel
        mask_xpix=null       : longblob  # x coordinates in pixels
        mask_ypix=null       : longblob  # y coordinates in pixels
        mask_weights         : longblob  # weights of the mask at the indices above
        """

    def make(self, key):
        motion_correction_key = (MotionCorrection & key).fetch1('KEY')

        method, loaded_result = get_loader_result(key, Curation)

        if method == 'caiman':
            loaded_caiman = loaded_result

            # infer "segmentation_channel" - from params if available, else from caiman loader
            params = (ProcessingParamSet * ProcessingTask & key).fetch1('params')
            segmentation_channel = params.get('segmentation_channel',
                                              loaded_caiman.segmentation_channel)

            masks, cells = [], []
            for mask in loaded_caiman.masks:
                masks.append({**key,
                              'segmentation_channel': segmentation_channel,
                              'mask': mask['mask_id'],
                              'mask_npix': mask['mask_npix'],
                              'mask_center_x': mask['mask_center_x'],
                              'mask_center_y': mask['mask_center_y'],
                              'mask_center_z': mask['mask_center_z'],
                              'mask_xpix': mask['mask_xpix'],
                              'mask_ypix': mask['mask_ypix'],
                              'mask_zpix': mask['mask_zpix'],
                              'mask_weights': mask['mask_weights']})
                if loaded_caiman.cnmf.estimates.idx_components is not None:
                    if mask['mask_id'] in loaded_caiman.cnmf.estimates.idx_components:
                        cells.append({
                            **key,
                            'mask_classification_method': 'caiman_default_classifier',
                            'mask': mask['mask_id'], 'mask_type': 'soma'})

            self.insert1({**key, **motion_correction_key})
            self.Mask.insert(masks, ignore_extra_fields=True)

            if cells:
                MaskClassification.insert1({
                    **key,
                    'mask_classification_method': 'caiman_default_classifier'},
                    allow_direct_insert=True)
                MaskClassification.MaskType.insert(cells,
                                                   ignore_extra_fields=True,
                                                   allow_direct_insert=True)

        elif method == 'mcgill_miniscope_analysis':
            loaded_miniscope_analysis = loaded_result

            # infer "segmentation_channel" - from params if available, else from miniscope analysis loader
            params = (ProcessingParamSet * ProcessingTask & key).fetch1('params')
            segmentation_channel = params.get('segmentation_channel',
                                              loaded_miniscope_analysis.segmentation_channel)

            self.insert1(key)
            self.Mask.insert([{**key,
                               'segmentation_channel': segmentation_channel,
                               'mask': mask['mask_id'],
                               'mask_npix': mask['mask_npix'],
                               'mask_center_x': mask['mask_center_x'],
                               'mask_center_y': mask['mask_center_y'],
                               'mask_xpix': mask['mask_xpix'],
                               'mask_ypix': mask['mask_ypix'],
                               'mask_weights': mask['mask_weights']}
                               for mask in loaded_miniscope_analysis.masks],
                               ignore_extra_fields=True)

        else:
            raise NotImplementedError(f'Unknown/unimplemented method: {method}')


@schema
class MaskClassificationMethod(dj.Lookup):
    definition = """
    mask_classification_method: varchar(48)
    """

    contents = zip(['caiman_default_classifier',
                    'miniscope_analysis_default_classifier'])


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


# Activity Trace ---------------------------------------------------------------

@schema
class Fluorescence(dj.Computed):
    definition = """  # fluorescence traces before spike extraction or filtering
    -> Segmentation
    """

    class Trace(dj.Part):
        definition = """
        -> master
        -> Segmentation.Mask
        -> Channel.proj(fluorescence_channel='channel')  # the channel that this trace comes from
        ---
        fluorescence                : longblob  # fluorescence trace associated with this mask
        neuropil_fluorescence=null  : longblob  # Neuropil fluorescence trace
        """

    def make(self, key):
        method, loaded_result = get_loader_result(key, Curation)

        if method == 'caiman':
            loaded_caiman = loaded_result

            # infer "segmentation_channel" - from params if available, else from caiman loader
            params = (ProcessingParamSet * ProcessingTask & key).fetch1('params')
            segmentation_channel = params.get('segmentation_channel',
                                              loaded_caiman.segmentation_channel)

            self.insert1(key)
            self.Trace.insert([{**key,
                                'mask': mask['mask_id'],
                                'fluorescence_channel': segmentation_channel,
                                'fluorescence': mask['inferred_trace']}
                                for mask in loaded_caiman.masks])

        elif method == 'mcgill_miniscope_analysis':
            loaded_miniscope_analysis = loaded_result

            # infer "segmentation_channel" - from params if available, else from miniscope analysis loader
            params = (ProcessingParamSet * ProcessingTask & key).fetch1('params')
            segmentation_channel = params.get('segmentation_channel',
                                              loaded_miniscope_analysis.segmentation_channel)

            self.insert1(key)
            self.Trace.insert([{**key,
                                'mask': mask['mask_id'],
                                'fluorescence_channel': segmentation_channel,
                                'fluorescence': mask['raw_trace']}
                                for mask in loaded_miniscope_analysis.masks])

        else:
            raise NotImplementedError('Unknown/unimplemented method: {}'.format(method))


@schema
class ActivityExtractionMethod(dj.Lookup):
    definition = """
    extraction_method: varchar(200)
    """

    contents = zip(['caiman_deconvolution',
                    'caiman_dff',
                    'mcgill_miniscope_analysis_deconvolution',
                    'mcgill_miniscope_analysis_dff'])


@schema
class Activity(dj.Computed):
    definition = """  # inferred neural activity from fluorescence trace - e.g. dff, spikes
    -> Fluorescence
    -> ActivityExtractionMethod
    """

    class Trace(dj.Part):
        definition = """  #
        -> master
        -> Fluorescence.Trace
        ---
        activity_trace: longblob  #
        """

    @property
    def key_source(self):
        caiman_key_source = (Fluorescence * ActivityExtractionMethod
                             * ProcessingParamSet.proj('processing_method')
                             & 'processing_method = "caiman"'
                             & 'extraction_method LIKE "caiman%"')

        miniscope_analysis_key_source = (Fluorescence * ActivityExtractionMethod
                             * ProcessingParamSet.proj('processing_method')
                             & 'processing_method = "mcgill_miniscope_analysis"'
                             & 'extraction_method LIKE "mcgill_miniscope_analysis%"')

        # TODO: fix #caiman_key_source.proj() + miniscope_analysis_key_source.proj()
        return miniscope_analysis_key_source.proj()

    def make(self, key):
        method, loaded_result = get_loader_result(key, Curation)

        if method == 'caiman':
            loaded_caiman = loaded_result

            if key['extraction_method'] in ('caiman_deconvolution', 'caiman_dff'):
                attr_mapper = {'caiman_deconvolution': 'spikes', 'caiman_dff': 'dff'}

                # infer "segmentation_channel" - from params if available, else from caiman loader
                params = (ProcessingParamSet * ProcessingTask & key).fetch1('params')
                segmentation_channel = params.get('segmentation_channel',
                                                  loaded_caiman.segmentation_channel)

                self.insert1(key)
                self.Trace.insert([{**key,
                                    'mask': mask['mask_id'],
                                    'fluorescence_channel': segmentation_channel,
                                    'activity_trace': mask[attr_mapper[key['extraction_method']]]}
                                    for mask in loaded_caiman.masks])

        elif method == 'mcgill_miniscope_analysis':
            if key['extraction_method'] in ('mcgill_miniscope_analysis_deconvolution', 'mcgill_miniscope_analysis_dff'):
                attr_mapper = {'mcgill_miniscope_analysis_deconvolution': 'spikes', 'mcgill_miniscope_analysis_dff': 'dff'}

                loaded_miniscope_analysis = loaded_result

                # infer "segmentation_channel" - from params if available, else from miniscope analysis loader
                params = (ProcessingParamSet * ProcessingTask & key).fetch1('params')
                segmentation_channel = params.get('segmentation_channel',
                                                  loaded_miniscope_analysis.segmentation_channel)

                self.insert1(key)
                self.Trace.insert([{**key,
                                    'mask': mask['mask_id'],
                                    'fluorescence_channel': segmentation_channel,
                                    'activity_trace': mask[attr_mapper[key['extraction_method']]]}
                                    for mask in loaded_miniscope_analysis.masks])

        else:
            raise NotImplementedError('Unknown/unimplemented method: {}'.format(method))


# Helper Functions -------------------------------------------------------------

_table_attribute_mapper = {'ProcessingTask': 'processing_output_dir',
                           'Curation': 'curation_output_dir'}


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
        'processing_method', _table_attribute_mapper[table.__name__])

    root_dir = pathlib.Path(get_miniscope_root_data_dir())
    output_dir = root_dir / output_dir

    if method == 'caiman':
        from element_data_loader import caiman_loader
        loaded_output = caiman_loader.CaImAn(output_dir)
    elif method == 'mcgill_miniscope_analysis':
        from element_data_loader import miniscope_analysis_loader
        loaded_output = miniscope_analysis_loader.MiniscopeAnalysis(output_dir)
    elif method == 'minian':
        from element_data_loader import minian_loader
        loaded_output = minian_loader.MiniAn(output_dir)
    else:
        raise NotImplementedError('Unknown/unimplemented method: {}'.format(method))

    return method, loaded_output


def populate_all(display_progress=True):

    populate_settings = {'display_progress': display_progress, 
                         'reserve_jobs': False, 
                         'suppress_errors': False}

    RecordingInfo.populate(**populate_settings)

    Processing.populate(**populate_settings)

    MotionCorrection.populate(**populate_settings)

    Segmentation.populate(**populate_settings)

    MaskClassification.populate(**populate_settings)

    Fluorescence.populate(**populate_settings)

    Activity.populate(**populate_settings)

