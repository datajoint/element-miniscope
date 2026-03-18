"""
Test script for Minian integration in element-miniscope.

This script sets up the database, populates required tables, and runs
the Minian processing pipeline.

Prerequisites:
- DataJoint database connection configured (dj.config)
- Minian package installed
- Test miniscope video files available

Usage:
    python test_minian_pipeline.py --data-dir /path/to/miniscope/videos
"""

import os
import sys
import argparse
import datajoint as dj
import numpy as np
from pathlib import Path
from datetime import datetime

# ============================================================================
# Configuration - Modify these paths for your environment
# ============================================================================

# Root directory containing raw miniscope data
# This should be the parent directory containing session folders with .avi files
DEFAULT_DATA_DIR = "./miniscope_test/"

# Root directory for processed outputs
DEFAULT_PROCESSED_DIR = "./processed/"

# Database schema prefix (schemas will be named: {prefix}_miniscope, etc.)
SCHEMA_PREFIX = "test"


# ============================================================================
# Linking Module - Required functions for element-miniscope
# ============================================================================

def get_miniscope_root_data_dir():
    """Return the root directory for raw miniscope data."""
    return [os.environ.get("MINISCOPE_ROOT_DATA_DIR", DEFAULT_DATA_DIR)]


def get_processed_root_data_dir():
    """Return the root directory for processed data."""
    processed_dir = os.environ.get("MINISCOPE_PROCESSED_DIR", DEFAULT_PROCESSED_DIR)
    os.makedirs(processed_dir, exist_ok=True)
    return processed_dir


def get_session_directory(session_key):
    """Return the session directory for a given session key.
    
    For testing, we assume the directory structure is:
    {root_data_dir}/{subject}/{session_date}/
    """
    from element_miniscope import miniscope
    
    # For a simple test setup, use the session key directly
    subject = session_key.get("subject", "test_subject")
    session_datetime = session_key.get("session_datetime", datetime.now())
    
    if isinstance(session_datetime, datetime):
        session_date = session_datetime.strftime("%Y-%m-%d")
    else:
        session_date = str(session_datetime).split()[0]
    
    return f"{subject}/{session_date}"


# Create a module-like object for the linking module
class LinkingModule:
    get_miniscope_root_data_dir = staticmethod(get_miniscope_root_data_dir)
    get_processed_root_data_dir = staticmethod(get_processed_root_data_dir)
    get_session_directory = staticmethod(get_session_directory)


# ============================================================================
# Schema Setup
# ============================================================================

def setup_schemas():
    """Activate the miniscope schemas."""
    from element_miniscope import miniscope
    from element_miniscope import miniscope_report
    
    # Create minimal upstream tables for testing
    schema = dj.schema(f"{SCHEMA_PREFIX}_lab")
    
    @schema
    class Subject(dj.Manual):
        definition = """
        subject: varchar(32)
        """
    
    @schema
    class Session(dj.Manual):
        definition = """
        -> Subject
        session_datetime: datetime
        """
    
    @schema
    class Device(dj.Lookup):
        definition = """
        device: varchar(32)
        """
        contents = [("Miniscope_V4",)]
    
    @schema
    class AnatomicalLocation(dj.Lookup):
        definition = """
        location: varchar(32)
        """
        contents = [("CA1",), ("mPFC",)]
    
    # Add required attributes to linking module
    LinkingModule.Subject = Subject
    LinkingModule.Session = Session
    LinkingModule.Device = Device
    LinkingModule.AnatomicalLocation = AnatomicalLocation
    
    # Activate miniscope schema
    miniscope.activate(
        f"{SCHEMA_PREFIX}_miniscope",
        linking_module=LinkingModule,
        create_schema=True,
        create_tables=True,
    )
    
    # Activate report schema
    miniscope_report.activate(
        f"{SCHEMA_PREFIX}_miniscope_report",
        create_schema=True,
        create_tables=True,
    )
    
    print(f"Schemas activated: {SCHEMA_PREFIX}_lab, {SCHEMA_PREFIX}_miniscope, {SCHEMA_PREFIX}_miniscope_report")
    
    return Subject, Session, Device, miniscope


def populate_test_data(Subject, Session, miniscope, data_dir):
    """Populate the database with test data entries."""
    
    # Insert test subject
    subject_key = {"subject": "test_mouse"}
    Subject.insert1(subject_key, skip_duplicates=True)
    print(f"Inserted subject: {subject_key}")
    
    # Insert test session
    session_key = {
        **subject_key,
        "session_datetime": datetime.now().replace(microsecond=0),
    }
    Session.insert1(session_key, skip_duplicates=True)
    print(f"Inserted session: {session_key}")
    
    # Insert recording
    recording_key = {
        **session_key,
        "recording_id": 0,
        "acq_software": "Miniscope-DAQ-V4",
    }
    miniscope.Recording.insert1(recording_key, skip_duplicates=True)
    print(f"Inserted recording: {recording_key}")
    
    # Populate RecordingInfo (this reads metadata from the video files)
    miniscope.RecordingInfo.populate(display_progress=True)
    print("Populated RecordingInfo")
    
    return session_key, recording_key


def create_minian_paramset(miniscope):
    """Create a ProcessingParamSet for Minian analysis."""
    
    minian_params = {
        # Video loading
        "load_videos": {
            "pattern": r".*\.avi$",
            "dtype": "uint8",
            "downsample": {"frame": 1, "height": 1, "width": 1},
            "downsample_strategy": "subset",
        },
        # Preprocessing
        "denoise": {"method": "median", "ksize": 7},
        "background_removal": {"method": "tophat", "wnd": 10},
        # Motion correction
        "estimate_motion": {"dim": "frame"},
        # Seeds initialization
        "seeds_init": {
            "wnd_size": 1000,
            "method": "rolling",
            "stp_size": 500,
            "max_wnd": 15,
            "diff_thres": 6.5,
        },
        "pnr_refine": {"noise_freq": 0.05, "thres": 1},
        "ks_refine": {"sig": 0.05},
        "seeds_merge": {"thres_dist": 10, "thres_corr": 0.8, "noise_freq": 0.06},
        "initialize": {"thres_corr": 0.8, "wnd": 10, "noise_freq": 0.06},
        "init_merge": {"thres_corr": 0.8},
        # CNMF
        "get_noise": {"noise_range": (0.06, 0.5)},
        "first_spatial": {
            "dl_wnd": 5,
            "sparse_penal": 0.001,
            "size_thres": (20, None),
        },
        "first_temporal": {
            "noise_freq": 0.06,
            "sparse_penal": 0.001,
            "p": 1,
            "add_lag": 10,
            "jac_thres": 0.1,
        },
        "first_merge": {"thres_corr": 0.6},
        "second_spatial": {
            "dl_wnd": 10,
            "sparse_penal": 0.001,
            "size_thres": (20, None),
        },
        "second_temporal": {
            "noise_freq": 0.06,
            "sparse_penal": 0.01,
            "p": 1,
            "add_lag": 10,
            "jac_thres": 0.2,
            "zero_thres": 1e-10,
        },
    }
    
    paramset_idx = 0
    
    miniscope.ProcessingParamSet.insert_new_params(
        processing_method="minian",
        paramset_idx=paramset_idx,
        paramset_desc="Default Minian parameters for miniscope analysis",
        params=minian_params,
    )
    
    print(f"Created Minian ProcessingParamSet with idx={paramset_idx}")
    return paramset_idx


def create_processing_task(miniscope, recording_key, paramset_idx):
    """Create a ProcessingTask entry."""
    
    task_key = {
        **recording_key,
        "paramset_idx": paramset_idx,
    }
    
    # Infer output directory
    output_dir = miniscope.ProcessingTask.infer_output_dir(
        task_key, relative=True, mkdir=True
    )
    
    miniscope.ProcessingTask.insert1(
        {
            **task_key,
            "processing_output_dir": str(output_dir),
            "task_mode": "trigger",
        },
        skip_duplicates=True,
    )
    
    print(f"Created ProcessingTask: {task_key}")
    print(f"Output directory: {output_dir}")
    
    return task_key


def run_processing(miniscope):
    """Run the processing pipeline."""
    print("\n" + "=" * 60)
    print("Starting Minian Processing...")
    print("=" * 60 + "\n")
    
    miniscope.Processing.populate(display_progress=True)
    
    print("\nProcessing complete!")
    print(f"Entries in Processing table: {len(miniscope.Processing())}")


def populate_downstream_tables(miniscope, miniscope_report):
    """Populate downstream analysis tables."""
    print("\n" + "=" * 60)
    print("Populating downstream tables...")
    print("=" * 60 + "\n")
    
    print("Populating MotionCorrection...")
    miniscope.MotionCorrection.populate(display_progress=True)
    
    print("Populating Segmentation...")
    miniscope.Segmentation.populate(display_progress=True)
    
    print("Populating Fluorescence...")
    miniscope.Fluorescence.populate(display_progress=True)
    
    print("Populating Activity...")
    miniscope.Activity.populate(display_progress=True)
    
    print("Populating visualizations...")
    miniscope_report.MinianProcessingVisualization.populate(display_progress=True)
    
    print("\nDownstream tables populated!")


def verify_results(miniscope):
    """Print summary of results."""
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60 + "\n")
    
    print(f"Processing entries: {len(miniscope.Processing())}")
    print(f"MotionCorrection entries: {len(miniscope.MotionCorrection())}")
    print(f"Segmentation entries: {len(miniscope.Segmentation())}")
    print(f"Segmentation.Mask entries: {len(miniscope.Segmentation.Mask())}")
    print(f"Fluorescence entries: {len(miniscope.Fluorescence())}")
    print(f"Fluorescence.Trace entries: {len(miniscope.Fluorescence.Trace())}")
    print(f"Activity entries: {len(miniscope.Activity())}")
    
    # Show some mask statistics if available
    if len(miniscope.Segmentation.Mask()) > 0:
        masks = miniscope.Segmentation.Mask.fetch()
        print(f"\nDetected {len(masks)} ROIs/cells")


def main():
    parser = argparse.ArgumentParser(description="Test Minian integration")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help="Path to directory containing miniscope video files",
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default=DEFAULT_PROCESSED_DIR,
        help="Path to directory for processed outputs",
    )
    parser.add_argument(
        "--skip-processing",
        action="store_true",
        help="Skip processing and just verify existing results",
    )
    args = parser.parse_args()
    
    # Set environment variables
    os.environ["MINISCOPE_ROOT_DATA_DIR"] = os.path.abspath(args.data_dir)
    os.environ["MINISCOPE_PROCESSED_DIR"] = os.path.abspath(args.processed_dir)
    
    print(f"Data directory: {os.environ['MINISCOPE_ROOT_DATA_DIR']}")
    print(f"Processed directory: {os.environ['MINISCOPE_PROCESSED_DIR']}")
    
    # Import after setting paths
    from element_miniscope import miniscope_report
    
    # Setup schemas
    Subject, Session, Device, miniscope = setup_schemas()
    
    if not args.skip_processing:
        # Populate test data
        session_key, recording_key = populate_test_data(
            Subject, Session, miniscope, args.data_dir
        )
        
        # Create parameter set
        paramset_idx = create_minian_paramset(miniscope)
        
        # Create processing task
        task_key = create_processing_task(miniscope, recording_key, paramset_idx)
        
        # Run processing
        run_processing(miniscope)
        
        # Populate downstream tables
        populate_downstream_tables(miniscope, miniscope_report)
    
    # Verify results
    verify_results(miniscope)
    
    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
