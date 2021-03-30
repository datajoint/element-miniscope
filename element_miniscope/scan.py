import datajoint as dj
import pathlib
import importlib
import inspect
import numpy as np

schema = dj.schema()

_linking_module = None


def activate(scan_schema_name, *, create_schema=True, create_tables=True, linking_module=None):
    """
    activate(scan_schema_name, *, create_schema=True, create_tables=True, linking_module=None)
        :param scan_schema_name: schema name on the database server to activate the `scan` module
        :param create_schema: when True (default), create schema in the database if it does not yet exist.
        :param create_tables: when True (default), create tables in the database if they do not yet exist.
        :param linking_module: a module name or a module containing the
         required dependencies to activate the `scan` module:
            Upstream tables:
                + Session: parent table to Scan, typically identifying a recording session
                + Equipment: Reference table for Scan, specifying the equipment used for the acquisition of this scan
                + Location: Reference table for ScanLocation, specifying the brain location where this scan is acquired
            Functions:
                + get_imaging_root_data_dir() -> str
                    Retrieve the full path for the root data directory (e.g. the mounted drive)
                    :return: a string with full path to the root data directory
                + get_miniscope_daq_v3_file(scan_key: dict) -> str
                    Retrieve the Miniscope DAQ V3 files associated with a given Scan
                    :param scan_key: key of a Scan
                    :return: Miniscope DAQ V3 files full file-path
    """

    if isinstance(linking_module, str):
        linking_module = importlib.import_module(linking_module)
    assert inspect.ismodule(linking_module),\
        "The argument 'dependency' must be a module's name or a module"

    global _linking_module
    _linking_module = linking_module

    schema.activate(scan_schema_name, create_schema=create_schema,
                    create_tables=create_tables, add_objects=_linking_module.__dict__)


# ---------------- Functions required by the element-miniscope  ----------------


def get_imaging_root_data_dir() -> str:
    """
    Retrieve the full path for the root data directory (e.g. the mounted drive)
    :return: a string with full path to the root data directory
    """
    return _linking_module.get_imaging_root_data_dir()

def get_miniscope_daq_v3_files(scan_key: dict) -> str:
    """
    Retrieve the Miniscope DAQ V3 files associated with a given Scan
    :param scan_key: key of a Scan
    :return: Miniscope DAQ V3 files full file-path
    """
    return _linking_module.get_miniscope_daq_v3_files(scan_key)


# ----------------------------- Table declarations -----------------------------


@schema
class AcquisitionSoftware(dj.Lookup):
    definition = """  # Name of acquisition software
    acq_software: varchar(24)    
    """
    contents = zip(['Miniscope-DAQ-V3'])


@schema
class Channel(dj.Lookup):
    definition = """  # Recording channel
    channel     : tinyint  # 0-based indexing
    """
    contents = zip(range(5))


# ------------------------------------ Scan ------------------------------------


@schema
class Scan(dj.Manual):
    definition = """    
    -> Session
    scan_id: int        
    ---
    -> Equipment  
    -> AcquisitionSoftware  
    scan_notes='' : varchar(4095)         # free-notes
    """


@schema
class ScanLocation(dj.Manual):
    definition = """
    -> Scan   
    ---    
    -> Location      
    """


@schema
class ScanInfo(dj.Imported):
    definition = """ # general data about the reso/meso scans
    -> Scan
    ---
    nfields              : tinyint   # number of fields
    nchannels            : tinyint   # number of channels
    ndepths              : int       # Number of scanning depths (planes)
    nframes              : int       # number of recorded frames
    nrois                : tinyint   # number of regions of interest
    x=null               : float     # (um) 0 point in the motor coordinate system
    y=null               : float     # (um) 0 point in the motor coordinate system
    z=null               : float     # (um) 0 point in the motor coordinate system
    fps                  : float     # (Hz) frames per second - volumetric scan rate
    """

    class Field(dj.Part):
        definition = """ # field-specific scan information
        -> master
        field_idx         : int
        ---
        px_height         : smallint  # height in pixels
        px_width          : smallint  # width in pixels
        um_height=null    : float     # height in microns
        um_width=null     : float     # width in microns
        field_x=null      : float     # (um) center of field in the motor coordinate system
        field_y=null      : float     # (um) center of field in the motor coordinate system
        field_z=null      : float     # (um) relative depth of field
        delay_image=null  : longblob  # (ms) delay between the start of the scan and pixels in this field
        """

    class ScanFile(dj.Part):
        definition = """
        -> master
        file_path: varchar(255)  # filepath relative to root data directory
        """

    def make(self, key):
        """ Read and store some scan meta information."""
        acq_software = (Scan & key).fetch1('acq_software')

        if acq_software == 'Miniscope-DAQ-V3':
            # Parse image dimension and frame rate
            import cv2
            scan_filepaths = get_miniscope_daq_v3_files(key)
            video = cv2.VideoCapture(scan_filepaths[0])
            fps = video.get(cv2.CAP_PROP_FPS) # TODO: Verify this method extracts correct value
            _, frame = video.read()
            frame_size = np.shape(frame)

            # Parse number of frames from timestamp.dat file
            with open(scan_filepaths[-1]) as f:
                next(f)
                nframes = sum(1 for line in f if int(line[0]) == 0)

            # Insert in ScanInfo
            self.insert1(dict(key,
                              nfields=1,
                              nchannels=1,
                              nframes=nframes,
                              ndepths=1,
                              fps=fps,
                              nrois=0))

            # Insert Field(s)
            self.Field.insert([dict(key,
                                   field_idx=0, 
                                   px_height=frame_size[0], 
                                   px_width=frame_size[1])])
        
        else:
            raise NotImplementedError(
                f'Loading routine not implemented for {acq_software} acquisition software')

        # Insert file(s)
        root = pathlib.Path(get_imaging_root_data_dir())
        scan_files = [pathlib.Path(f).relative_to(root).as_posix() for f in scan_filepaths]
        self.ScanFile.insert([{**key, 'file_path': f} for f in scan_files])
