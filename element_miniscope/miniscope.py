import datajoint as dj
import numpy as np
import pathlib
from datetime import datetime
import uuid
import hashlib
import importlib
import inspect

from . import scan

schema = dj.schema()

_linking_module = None


def activate(imaging_schema_name, scan_schema_name=None, *,
             create_schema=True, create_tables=True, linking_module=None):
    """
    activate(imaging_schema_name, *, scan_schema_name=None, create_schema=True, create_tables=True, linking_module=None)
        :param imaging_schema_name: schema name on the database server to activate the `imaging` module
        :param scan_schema_name: schema name on the database server to activate the `scan` module
         - may be omitted if the `scan` module is already activated
        :param create_schema: when True (default), create schema in the database if it does not yet exist.
        :param create_tables: when True (default), create tables in the database if they do not yet exist.
        :param linking_module: a module name or a module containing the
         required dependencies to activate the `imaging` module:
            Upstream tables:
                + Session: parent table to Scan, typically identifying a recording session
            Functions:
                + get_miniscope_root_data_dir() -> str
                    Retrieve the root data directory - e.g. containing all subject/sessions data
                    :return: a string for full path to the root data directory
    """

    if isinstance(linking_module, str):
        linking_module = importlib.import_module(linking_module)
    assert inspect.ismodule(linking_module),\
        "The argument 'dependency' must be a module's name or a module"

    global _linking_module
    _linking_module = linking_module

    scan.activate(scan_schema_name, create_schema=create_schema,
                  create_tables=create_tables, linking_module=linking_module)
    schema.activate(imaging_schema_name, create_schema=create_schema,
                    create_tables=create_tables, add_objects=_linking_module.__dict__)


# -------------- Functions required by the element-calcium-imaging  --------------

def get_miniscope_daq_v3_files(key: dict) -> str:
    """
    Retrieve the Miniscope DAQ V3 files associated with a given Scan
    :param scan_key: key of a Scan
    :return: Miniscope DAQ V3 files full file-path
    """
    return _linking_module.get_miniscope_daq_v3_files(key)


def get_miniscope_root_data_dir() -> str:
    """
    get_miniscope_root_data_dir() -> str
        Retrieve the root data directory - e.g. containing all subject/sessions data
        :return: a string for full path to the root data directory
    """
    return _linking_module.get_miniscope_root_data_dir()


# -------------- Table declarations --------------

@schema
class AcquisitionSoftware(dj.Lookup):
    definition = """  # Name of acquisition software
    acq_software: varchar(24)
    """
    contents = zip([
        'Miniscope-DAQ-V3',
        'Inscopix nVoke'])


@schema
class Channel(dj.Lookup):
    definition = """  # Recording channel
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
    recording_notes='' : varchar(4095)         # free-notes
    """


@schema
class RecordingLocation(dj.Manual):
    definition = """
    -> Recording
    ---
    -> Location
    """


@schema
class RecordingInfo(dj.Imported):
    definition = """ # general data about recording
    -> Recording
    ---
    nchannels            : tinyint   # number of channels
    nplanes              : int       # number of recording planes
    nframes              : int       # number of recorded frames
    x=null               : float     # (um) 0 point in the motor coordinate system
    y=null               : float     # (um) 0 point in the motor coordinate system
    z=null               : float     # (um) 0 point in the motor coordinate system
    fps                  : float     # (Hz) frames per second - volumetric scan rate
    gain=null            : float     # recording gain
    spatial_downsample=1 : tinyint   # e.g. 1, 2, 4, 8. 1 for no downsampling
    temporal_downsample=1: tinyint   # e.g. 1, 2, 4, 8. 1 for no downsampling
    led_power            : float     # LED power used in the given recording
    """

    class Plane(dj.Part):
        definition = """ # field-specific scan information
        -> master
        plane_id          : int
        ---
        px_height=null    : smallint  # height in pixels
        px_width=null     : smallint  # width in pixels
        um_height=null    : float     # height in microns
        um_width=null     : float     # width in microns
        plane_z=null      : float     # (um) relative depth of the recording plane
        """

    class File(dj.Part):
        definition = """
        -> master
        recording_file_path: varchar(255)  # filepath relative to root data directory
        """

    def make(self, key):
        """ Read and store some scan meta information."""
        acq_software = (Recording & key).fetch1('acq_software')

        if acq_software == 'Miniscope-DAQ-V3':
            # Parse image dimension and frame rate
            import cv2
            recording_filepaths = get_miniscope_daq_v3_files(key)
            video = cv2.VideoCapture(recording_filepaths[0])
            fps = video.get(cv2.CAP_PROP_FPS) # TODO: Verify this method extracts correct value
            _, frame = video.read()
            frame_size = np.shape(frame)

            # Parse number of frames from timestamp.dat file
            with open(recording_filepaths[-1]) as f:
                next(f)
                nframes = sum(1 for line in f if int(line[0]) == 0)

            # Insert in RecordingInfo
            self.insert1(dict(key,
                              nchannels=1,
                              nframes=nframes,
                              nplanes=1,
                              fps=fps))

            # Insert Plane(s)
            self.Plane.insert1(dict(key,
                                    plane_id=0,
                                    px_height=frame_size[0],
                                    px_width=frame_size[1]))

        else:
            raise NotImplementedError(
                f'Loading routine not implemented for {acq_software} acquisition software')

        # Insert file(s)
        root = pathlib.Path(get_miniscope_root_data_dir())
        recording_files = [pathlib.Path(f).relative_to(root).as_posix() for f in recording_filepaths]
        self.File.insert([{**key, 'recording_file_path': f} for f in recording_files])


@schema
class PreprocessingMethod(dj.Lookup):
    definition = """
    preprocessing_method : varchar(12)
    ---
    preprocessing_method_desc='': varchar(255)
    """
    contents = [
        ['inscopix', ''],
        ['no preprocessing', '']]


@schema
class Preprocessing(dj.Manual):
    definition = """
    -> RecordingInfo
    ---
    -> PreprocessingMethod
    """

    class Plane(dj.Part):
        definition = """
        -> RecordingInfo.Plane
        ---
        trim_initial_frames=0    : boolean
        cropping_imaging=0       : boolean
        crop_area_left=0         : smallint            # (pixels)
        crop_area_top=0          : smallint            # (pixels)
        crop_area_px_width=null  : smallint            # (pixels)
        crop_area_px_height=null : smallint            # (pixels)
        fix_defective_pixels=0   : boolean
        spatial_downsample=1     : tinyint             # spatial downsample with respect to the raw movie
        temporal_downsampl=1     : tinyint             # temporal downsample with respect to the raw movie
        """

    class File(dj.Part):
        definition = """
        -> master
        preprocessing_filepath  : varchar(255)     # filepath of preprocessed video
        """

    @classmethod
    def create1_from_recording_info(self, key: dict):
        """
        A convenient function to create a new corresponding "Preprocessing" entry for a particular
        "RecordingInfo" entry.
        """
        if key not in RecordingInfo():
            raise ValueError(f'No corresponding entry in RecordingInfo for: {key}')

        self.insert1(dict(
            key, preprocessing_method='no preprocessing'))

        if RecordingInfo.Plane & key:
            self.insert(
                [dict(key, crop_area_width=plane['px_width'],
                      crop_area_height=plane['px_height'])
                 for plane in (RecordingInfo.Plane & key).fetch(as_dict=True)])

        if RecordingInfo.File & key:
            self.insert(
                [dict(key, preprocessing_filepath=file['recording_filepath'])
                 for file in (RecordingInfo.File & key).fetch(as_dict=True)])


@schema
class MotionCorrectionMethod(dj.Lookup):
    definition = """
    motion_correction_method: char(32)
    ---
    motion_correction_method_desc='': varchar(1000)
    """

    contents = [
        ('inscopix', 'Motion correction running through Inscopix software with manual interaction'),
        ('suite2p', 'Motion correction using auto processed Suite2p')
        ('caiman', 'Motion correction using auto processed caiman')
    ]


@schema
class ProcessingMethod(dj.Lookup):
    definition = """
    processing_method: char(32)
    ---
    processing_method_desc: varchar(1000)
    """

    contents = [('caiman', 'CaImAn Analysis Suite'),
                ('mcgill_miniscope_analysis', 'MiniscopeAnalysis from McGill University (https://github.com/etterguillaume/MiniscopeAnalysis)')]


@schema
class ProcessingParamSet(dj.Lookup):
    definition = """
    paramset_idx:  smallint
    ---
    -> ProcessingMethod
    paramset_desc: varchar(128)
    param_set_hash: uuid
    unique index (param_set_hash)
    params: longblob  # dictionary of all applicable parameters
    """

    @classmethod
    def insert_new_params(cls, processing_method: str,
                          paramset_idx: int, paramset_desc: str, params: dict):
        param_dict = {'processing_method': processing_method,
                      'paramset_idx': paramset_idx,
                      'paramset_desc': paramset_desc,
                      'params': params,
                      'param_set_hash': dict_to_uuid(params)}
        q_param = cls & {'param_set_hash': param_dict['param_set_hash']}

        if q_param:  # If the specified param-set already exists
            pname = q_param.fetch1('paramset_idx')
            if pname == paramset_idx:  # If the existed set has the same name: job done
                return
            else:  # If not same name: human error, trying to add the same paramset with different name
                raise dj.DataJointError(
                    'The specified param-set already exists - name: {}'.format(pname))
        else:
            cls.insert1(param_dict)


# -------------- Trigger a processing routine --------------

@schema
class ProcessingTask(dj.Manual):
    definition = """
    -> Preprocessing
    -> ProcessingParamSet
    ---
    processing_output_dir: varchar(255)         #  output directory of the processed scan relative to root data directory
    task_mode='load': enum('load', 'trigger')   # 'load': load computed analysis results, 'trigger': trigger computation
    """


@schema
class Processing(dj.Computed):
    definition = """
    -> ProcessingTask
    ---
    processing_time     : datetime  # time of generation of this set of processed, segmented results
    package_version=''  : varchar(16)
    """

    # Run processing only on Scan with ScanInfo inserted
    @property
    def key_source(self):
        return ProcessingTask & scan.ScanInfo

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


@schema
class Curation(dj.Manual):
    definition = """
    -> Processing
    curation_id: int
    ---
    curation_time: datetime             # time of generation of this set of curated results
    curation_output_dir: varchar(255)   # output directory of the curated results, relative to root data directory
    manual_curation: bool               # has manual curation been performed on this result?
    curation_note='': varchar(2000)
    """

    def create1_from_processing_task(self, key, is_curated=False, curation_note=''):
        """
        A convenient function to create a new corresponding "Curation" for a particular "ProcessingTask"
        """
        if key not in Processing():
            raise ValueError(f'No corresponding entry in Processing available for: {key};'
                             f' do `Processing.populate(key)`')

        output_dir = (ProcessingTask & key).fetch1('processing_output_dir')
        method, loaded_result = get_loader_result(key, ProcessingTask)

        if method == 'caiman':
            loaded_caiman = loaded_result
            curation_time = loaded_caiman.creation_time
        elif method == 'mcgill_miniscope_analysis':
            loaded_miniscope_analysis = loaded_result
            curation_time = loaded_miniscope_analysis.creation_time
        else:
            raise NotImplementedError('Unknown method: {}'.format(method))

        # Synthesize curation_id
        curation_id = dj.U().aggr(self & key, n='ifnull(max(curation_id)+1,1)').fetch1('n')
        self.insert1({**key, 'curation_id': curation_id,
                      'curation_time': curation_time, 'curation_output_dir': output_dir,
                      'manual_curation': is_curated,
                      'curation_note': curation_note})


# -------------- Motion Correction --------------

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

    class Summary(dj.Part):
        definition = """ # summary images for each field and channel after corrections
        -> master
        -> RecordingInfo.Plane
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
        mask_npix            : int       # number of pixels in ROIs
        mask_center_x=null   : int       # center x coordinate in pixel                         # TODO: determine why some masks don't have information, thus null required
        mask_center_y=null   : int       # center y coordinate in pixel
        mask_center_z=null   : int       # center z coordinate in pixel
        mask_xpix=null       : longblob  # x coordinates in pixels
        mask_ypix=null       : longblob  # y coordinates in pixels
        mask_zpix=null       : longblob  # z coordinates in pixels
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


# -------------- Activity Trace --------------


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

# ---------------- HELPER FUNCTIONS ----------------


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
        from .readers import caiman_loader
        loaded_output = caiman_loader.CaImAn(output_dir)
    elif method == 'mcgill_miniscope_analysis':
        from .readers import miniscope_analysis_loader
        loaded_output = miniscope_analysis_loader.MiniscopeAnalysis(output_dir)
    else:
        raise NotImplementedError('Unknown/unimplemented method: {}'.format(method))

    return method, loaded_output


def dict_to_uuid(key):
    """
    Given a dictionary `key`, returns a hash string as UUID
    """
    hashed = hashlib.md5()
    for k, v in sorted(key.items()):
        hashed.update(str(k).encode())
        hashed.update(str(v).encode())
    return uuid.UUID(hex=hashed.hexdigest())
