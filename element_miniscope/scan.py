import datajoint as dj
import pathlib
import importlib
import inspect

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
                + get_miniscope_daq_file(scan_key: dict) -> str
                    Retrieve the Miniscope DAQ file (*.json) associated with a given Scan
                    :param scan_key: key of a Scan
                    :return: Miniscope DAQ file full file-path
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

def get_miniscope_daq_file(scan_key: dict) -> str:
    """
    Retrieve the Miniscope DAQ file (*.json) associated with a given Scan
    :param scan_key: key of a Scan
    :return: Miniscope DAQ file full file-path
    """
    return _linking_module.get_miniscope_daq_file(scan_key)

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
    nrois                : tinyint   # number of ROIs
    x                    : float     # (um) 0 point in the motor coordinate system
    y                    : float     # (um) 0 point in the motor coordinate system
    z                    : float     # (um) 0 point in the motor coordinate system
    fps                  : float     # (Hz) frames per second - Volumetric Scan Rate 
    bidirectional        : boolean   # true = bidirectional scanning
    usecs_per_line=null  : float     # microseconds per scan line
    fill_fraction=null   : float     # raster scan temporal fill fraction
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
        field_x           : float     # (um) center of field in the motor coordinate system
        field_y           : float     # (um) center of field in the motor coordinate system
        field_z           : float     # (um) relative depth of field
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
            import cv2
            scan_filepaths = get_miniscope_daq_file(key)
            video = cv2.VideoCapture(scan_filepaths)
            fps = video.get(cv2.CAP_PROP_FPS) # TODO: Verify correct value
            ret, frame = video.read()
            print(np.shape(frame))# TODO: Verify correct order

            # Insert in ScanInfo
            self.insert1(dict(key,
                              nfields=1,
                              nchannels=1,
                              nframes=, # TODO: extract from timestamps.dat and count how many frames for cam 0.
                              ndepths=1,
                              fps=fps,
                              bidirectional=False,
                              nrois=0))

            # Insert Field(s)
            self.Field.insert(dict(key,
                                   field_idx=0, 
                                   px_height=frame[0], 
                                   px_width=frame[1]))
        else:
            raise NotImplementedError(
                f'Loading routine not implemented for {acq_software} acquisition software')

        # Insert file(s)
        root = pathlib.Path(get_imaging_root_data_dir())
        scan_files = [pathlib.Path(f).relative_to(root).as_posix() for f in scan_filepaths]
        self.ScanFile.insert([{**key, 'file_path': f} for f in scan_files])
