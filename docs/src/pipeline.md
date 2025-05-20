# Data Pipeline

Each node in the following diagram represents the analysis code in the workflow and the
corresponding table in the database.  Within the workflow, Element Miniscope
connects to upstream Elements including Lab, Animal, Session, and Event. For more 
detailed documentation on each table, see the API docs for the respective schemas.

![pipeline](https://raw.githubusercontent.com/datajoint/element-miniscope/main/images/pipeline.svg)

## Table descriptions

### `reference` schema

+ For further details see the [reference schema API docs](https://docs.datajoint.com/elements/element-miniscope/0.3/api/workflow_miniscope/reference/)

| Table | Description |
| --- | --- |
| Equipment | Scanner metadata |

### `subject` schema

+ Although not required, most choose to connect the `Session` table to a `Subject` table.

+ For further details see the [subject schema API docs](https://docs.datajoint.com/elements/element-animal/latest/api/element_animal/subject/)

| Table | Description |
| --- | --- |
| Subject | Basic information of the research subject. |

### `session` schema

+ For further details see the [session schema API docs](https://docs.datajoint.com/elements/element-session/latest/api/element_session/session_with_datetime/)

| Table | Description |
| --- | --- |
| Session | Unique experimental session identifier. |

### `miniscope` schema

+ Tables related to importing and analyzing miniscope data.

| Table | Description |
| --- | --- |
| AcquisitionSoftware | Software used for miniscope acquisition |
| Channel | Number of channels in the miniscope recording |
| Recording | Information about the equipment used (e.g. the acquisition hardware information). |
| RecordingLocation | Brain location of the miniscope recording |
| RecordingInfo |  Metadata about this recording from the Miniscope DAQ software (e.g. frame rate, number of channels, frames, etc.) |
| RecordingInfo.File | Relative file paths for the recording files |
| ProcessingMethod | Available analysis suites that can be used in processing of the miniscope datasets |
| ProcessingParamSet | All parameters required to process a miniscope dataset |
| MaskType | All possible classifications of a segmented mask |
| ProcessingTask | Task defined by a combination of Recording and ProcessingParamSet |
| Processing | The core table that executes a ProcessingTask |
| Curation | Curated results |
| MotionCorrection | Results of the motion correction procedure |
| MotionCorrection.RigidMotionCorrection | Details of the rigid motion correction performed on the miniscope data |
| MotionCorrection.NonRigidMotionCorrection | Details of nonrigid motion correction performed on the miniscope data |
| MotionCorrection.NonRigidMotionCorrection.Block | Results of non-rigid motion correction for each block |
| MotionCorrection.Summary | Summary images for each field and channel after motion correction |
| Segmentation | Results of the segmentation |
| Segmentation.Mask | Masks identified in the segmentation procedure |
| MaskClassificationMethod | Method used in the mask classification procedure |
| MaskClassification | Result of the mask classification procedure |
| MaskClassification.MaskType | Classification type assigned to each mask (e.g. soma, axon, dendrite, artifact, etc.) |
| Fluorescence | Fluorescence measurements |
| Fluorescence.Trace | Fluorescence trace for each region of interest |
| ActivityExtractionMethod | Method used in activity extraction |
| Activity | Inferred neural activity |
| Activity.Trace | Inferred neural activity for each fluorescence trace |
| ProcessingQualityMetrics | Quality metrics used to evaluate the results of the calcium imaging analysis pipeline |
| ProcessingQualityMetrics.Mask | Quality metrics used to evaluate the masks |
| ProcessingQualityMetrics.Trace | Quality metrics used to evaluate the fluorescence traces |

### `miniscope_report` schema

+ Tables that provide summary reports of the processed miniscope data.

| Table | Description |
| --- | --- |
| QualityMetrics | A table containing information about CaImAn estimates: </p> + `r_values`: Space correlation </p> + `snr`: Trace SNR </p> + `cnn_preds`: CNN predictions|
