# Tutorials

## Installation

Installation of the Element requires an integrated development environment and database. Instructions to setup each of the components can be found on the [User Instructions](datajoint.com/docs/elements/user-instructions) page. These instructions use the example [workflow for Element Miniscope](https://github.com/datajoint/workflow-miniscope), which can be modified for a user's specific experimental requirements.

## YouTube

Our [Element Miniscope tutorial](https://www.youtube.com/watch?v=nWUcPFZOSVw) gives an overview of the workflow directory as well as core concepts related to Miniscope itself.

[![YouTube tutorial](https://img.youtube.com/vi/nWUcPFZOSVw/0.jpg)](https://www.youtube.com/watch?v=nWUcPFZOSVw)

## Notebooks

Each of the [notebooks](https://github.com/datajoint/workflow-miniscope/tree/main/notebooks) in the workflow steps through ways to interact with the Element itself. To try out Elements notebooks in an online Jupyter environment with access to example data, visit [CodeBook](https://codebook.datajoint.io/). (Miniscope notebooks coming soon!)

- [00-DataDownload](https://github.com/datajoint/workflow-miniscope/blob/main/notebooks/00-DataDownload_Optional.ipynb) highlights how to use DataJoint tools to download a sample model for trying out the Element.
- [01-Configure](https://github.com/datajoint/workflow-miniscope/blob/main/notebooks/01-Configure.ipynb) helps configure your local DataJoint installation to point to the correct database.
- [02-WorkflowStructure](https://github.com/datajoint/workflow-miniscope/blob/main/notebooks/02-WorkflowStructure_Optional.ipynb) demonstrates the table architecture of the Element and key DataJoint basics for interacting with these tables.
- [03-Process](https://github.com/datajoint/workflow-miniscope/blob/main/notebooks/03-Process.ipynb) steps through adding data to these tables.
- [05-Visualization](https://github.com/datajoint/workflow-miniscope/blob/main/notebooks/05-Visualization_Optional.ipynb) demonstrates how to fetch data from the Element to generate figures and label data.
- [06-Drop](https://github.com/datajoint/workflow-miniscope/blob/main/notebooks/06-Drop_Optional.ipynb) provides the steps for dropping all the tables to start fresh.
- [07-DownstreamAnalysis](https://github.com/datajoint/workflow-miniscope/blob/main/notebooks/07-downstream-analysis-optional.ipynb) demonstrates how to perform analyses such as activity alignment and exploratory visualizations 