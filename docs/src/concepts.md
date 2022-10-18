# Concepts

## Table Architecture

Each of the DataJoint Elements are a set of tables for common neuroinformatics modalities to organize, preprocess, and analyze data. Each node in the following diagram is either a table in the Element itself or a table that would be connected to the Element.

![element-miniscope diagram](https://raw.githubusercontent.com/datajoint/element-miniscope/main/images/attached_miniscope_element.svg)

- Upstream: 

    - Element Miniscope connects to a ***Session*** table, which is modeled in our [workflow pipeline](https://github.com/datajoint/workflow-miniscope/blob/main/workflow_deeplabcut/pipeline.py). 

    - Although not requried, most choose to connect ***Session*** to a ***Subject** table for managing research subjects.

- Tables:

    + `Recording` - table containing information about the equipment used (e.g. the acquisition hardware information)

    + `RecordingInfo` - metadata about this recording from the Miniscope DAQ software (e.g. frame rate, number of channels, frames, etc.)

    + `MotionCorrection` - motion correction information performed on a recording

    + `MotionCorrection.RigidMotionCorrection` - details of the rigid motion correction (e.g. shifting in x, y)

    + `MotionCorrection.NonRigidMotionCorrection` and `MotionCorrection.Block` tables are used to describe the non-rigid motion correction.

    + `MotionCorrection.Summary` - summary images after motion correction (e.g. average image, correlation image, etc.)

    + `Segmentation` - table specifies the segmentation step and its outputs, following the motion correction step.

    + `Segmentation.Mask` - image mask for the segmented region of interest

    + `MaskClassification` - classification of `Segmentation.Mask` into a type (e.g. soma, axon, dendrite, artifact, etc.)

    + `Fluorescence` - fluorescence traces extracted from each `Segmentation.Mask`

    + `ActivityExtractionMethod` - activity extraction method (e.g. deconvolution) applied on the fluorescence trace

    + `Activity` - computed neuronal activity trace from fluorescence trace (e.g. spikes)