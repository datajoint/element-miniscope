# DataJoint Element - Miniscope Calcium Imaging

+ This repository features DataJoint pipeline design for functional calcium imaging, 
with `Miniscope DAQ V3` acquisition system and `MiniscopeAnalysis` suite for analysis. 

+ The element presented here is not a complete workflow by itself,
 but rather a modular design of tables and dependencies specific to the functional calcium imaging workflow. 

+ This modular element can be flexibly attached downstream to 
any particular design of experiment session, thus assembling 
a fully functional calcium imaging workflow.

+ See the [Element Miniscope documentation](https://elements.datajoint.org/description/miniscope/) for the background information and development timeline.

+ For more information on the DataJoint Elements project, please visit https://elements.datajoint.org.  This work is supported by the National Institutes of Health.

## Element architecture

![element miniscope diagram](images/attached_miniscope_element.svg)

+ As the diagram depicts, `elements-miniscope` starts immediately downstream from `Session`, and also requires some notion of:

     + `Scanner` for equipment/device

     + `Location` as a dependency for `ScanLocation`

## Table definitions
<details>
<summary>Click to expand details</summary>

### Scan

+ A `Session` (more specifically an experimental session) may have multiple scans, where each scan describes a complete 4D dataset (i.e. 3D volume over time) from one scanning session, typically from the moment of pressing the *start* button to pressing the *stop* button.

+ `Scan` - table containing information about the equipment used (e.g. the Scanner information)

+ `ScanInfo` - meta information about this scan, from ScanImage header (e.g. frame rate, number of channels, scanning depths, frames, etc.)

+ `ScanInfo.Field` - a field is a 2D image at a particular xy-coordinate and plane (scanning depth) within the field-of-view (FOV) of the scan.

     + For resonant scanner, a field is usually the 2D image occupying the entire FOV from a certain plane (at some depth).

     + For mesoscope scanner, with much wider FOV, there may be multiple fields on one plane. 

### Motion correction

+ `MotionCorrection` - motion correction information performed on a scan

+ `MotionCorrection.RigidMotionCorrection` - details of the rigid motion correction (e.g. shifting in x, y) at a per `ScanInfo.Field` level

+ `MotionCorrection.NonRigidMotionCorrection` and `MotionCorrection.Block` tables are used to describe the non-rigid motion correction performed on each `ScanInfo.Field`

+ `MotionCorrection.Summary` - summary images for each `ScanInfo.Field` after motion correction (e.g. average image, correlation image)
    
### Segmentation

+ `Segmentation` - table specifies the segmentation step and its outputs, following the motion correction step.
 
+ `Segmentation.Mask` - image mask for the segmented region of interest from a particular `ScanInfo.Field`

+ `MaskClassification` - classification of `Segmentation.Mask` into different type (e.g. soma, axon, dendrite, artifact, etc.)

### Neural activity extraction

+ `Fluorescence` - fluorescence traces extracted from each `Segmentation.Mask`

+ `ActivityExtractionMethod` - activity extraction method (e.g. deconvolution) to be applied on fluorescence trace

+ `Activity` - computed neuronal activity trace from fluorescence trace (e.g. spikes)

</details>

## Citation

+ If your work uses DataJoint and DataJoint Elements, please cite the respective Research Resource Identifiers (RRIDs) and manuscripts.

+ DataJoint for Python or MATLAB
    + Yatsenko D, Reimer J, Ecker AS, Walker EY, Sinz F, Berens P, Hoenselaar A, Cotton RJ, Siapas AS, Tolias AS. DataJoint: managing big scientific data using MATLAB or Python. bioRxiv. 2015 Jan 1:031658. doi: https://doi.org/10.1101/031658

    + DataJoint ([RRID:SCR_014543](https://scicrunch.org/resolver/SCR_014543)) - DataJoint for `<Select Python or MATLAB >` (version `<Enter version number>`)

+ DataJoint Elements
    + Yatsenko D, Nguyen T, Shen S, Gunalan K, Turner CA, Guzman R, Sasaki M, Sitonic D, Reimer J, Walker EY, Tolias AS. DataJoint Elements: Data Workflows for Neurophysiology. bioRxiv. 2021 Jan 1. doi: https://doi.org/10.1101/2021.03.30.437358

    + DataJoint Elements ([RRID:SCR_021894](https://scicrunch.org/resolver/SCR_021894)) - Element Miniscope (version `<Enter version number>`)