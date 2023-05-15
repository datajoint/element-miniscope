# Data Pipeline

Each node in the following diagram represents the analysis code in the workflow and the
corresponding table in the database.  Within the workflow, Element Miniscope
connects to upstream Elements including Lab, Animal, Session, and Event. For more 
detailed documentation on each table, see the API docs for the respective schemas.

![pipeline](https://raw.githubusercontent.com/datajoint/element-miniscope/main/images/pipeline.svg)

## Table descriptions

### `reference` schema

+ For further details see the [reference schema API docs](https://datajoint.com/docs/elements/element-miniscope/latest/api/workflow_miniscope/reference/)

| Table | Description |
| --- | --- |
| Equipment | Scanner metadata |

### `subject` schema

+ Although not required, most choose to connect the `Session` table to a `Subject` table.

+ For further details see the [subject schema API docs](https://datajoint.com/docs/elements/element-animal/api/element_animal/subject/)

| Table | Description |
| --- | --- |
| Subject | Basic information of the research subject. |

### `session` schema

+ For further details see the [session schema API docs](https://datajoint.com/docs/elements/element-session/latest/api/element_session/session_with_datetime/)

| Table | Description |
| --- | --- |
| Session | Unique experimental session identifier. |

### `miniscope` schema

+ Tables related to importing and analyzing miniscope data.

+ For further details see the [miniscope schema API docs](https://datajoint.com/docs/elements/element-miniscope/latest/api/element_miniscope/miniscope)

| Table | Description |
| --- | --- |
| Recording | A table containing information about the equipment used (e.g. the acquisition hardware information). |
| RecordingInfo |  The metadata about this recording from the Miniscope DAQ software (e.g. frame rate, number of channels, frames, etc.). |
| MotionCorrection | A table with information about motion correction performed on a recording. |
| MotionCorrection.RigidMotionCorrection | A table with details of rigid motion correction (e.g. shifting in x, y). |
| MotionCorrection.NonRigidMotionCorrection and MotionCorrection.Block | These tables describe the non-rigid motion correction. |
| MotionCorrection.Summary | A table containing summary images after motion correction. |
| Segmentation | This table specifies the segmentation step and its outputs, following the motion correction step. |
| Segmentation.Mask | This table contains the image mask for the segmented region of interest. |
| MaskClassification | This table contains information about the classification of `Segmentation.Mask` into a type (e.g. soma, axon, dendrite, artifact, etc.). |
| Fluorescence | This table contains the fluorescence traces extracted from each `Segmentation.Mask`. |
| ActivityExtractionMethod | A table with information about the activity extraction method (e.g. deconvolution) applied on the fluorescence trace. |
| Activity | A table with neuronal activity traces from fluorescence trace (e.g. spikes). |
| ProcessingQualityMetrics | Quality metrics used to evaluate the results of the calcium imaging analysis pipeline |
| ProcessingQualityMetrics.Mask | Quality metrics used to evaluate the masks |
| ProcessingQualityMetrics.Trace | Quality metrics used to evaluate the fluorescence traces |

### `miniscope_report` schema

+ Tables that provide summary reports of the processed miniscope data.

+ For further details see the [miniscope_report API docs](https://datajoint.com/docs/elements/element-miniscope/latest/api/element_miniscope/miniscope_report)

| Table | Description |
| --- | --- |
| QualityMetrics | A table containing information about CaImAn estimates: </p> + `r_values`: Space correlation </p> + `snr`: Trace SNR </p> + `cnn_preds`: CNN predictions|
