<!-- markdownlint-disable MD053 -->
# Concepts

## Miniscopes in Neuroscience Research

Miniature fluorescence microscopes (miniscopes) are a head-mounted calcium imaging
full-frame video modality first introduced in 2005 by Mark Schnitzer's lab[^1]. Due to
their light weight, these miniscopes allow measuring the dynamic activity of populations
of cortical neurons in freely behaving animals. In 2011, Inscopix Inc. was founded to
support one-photon miniscopes as a commercial neuroscience research platform, providing
proprietary hardware, acquisition software, and analysis software. Today, they estimate
their active user base is 491 labs with a total of 1179 installs.

An open-source alternative was launched by a UCLA team led by Drs. Daniel Aharoni and
Peyman Golshani[^2][^3]. In our conversation with Dr. Aharoni, he estimated about 700
labs currently using the UCLA system alone. The Inscopix user base is smaller but more
established. Several two-photon miniscopes have been developed but lack widespread
adoption likely due to the expensive hardware required for the two-photon
excitation[^3][^4][^5]. Due to the low costs and ability to record during natural
behaviors, one-photon miniscope imaging appears to be the fastest growing calcium
imaging modality in the field today.

The DataJoint team focused efforts on supporting the UCLA platform due rapid growth and
limited standardization in acquisition and processing pipelines. In the future, we will
reach out to Inscopix to support their platform as well.

[^1]:
    Flusberg BA, Jung JC, Cocker ED, Anderson EP, Schnitzer MJ. In vivo brain imaging
    using a portable 3.9 gram two-photon fluorescence microendoscope. *Opt Lett.* 2005
    Sep 1;30(17):2272-4. doi: 10.1364/ol.30.002272. PMID: 16190441.

[^2]:
    Cai DJ, Aharoni D, Shuman T, Shobe J, Biane J, Song W, Wei B, Veshkini M, La-Vu M,
    Lou J, Flores SE, Kim I, Sano Y, Zhou M, Baumgaertel K, Lavi A, Kamata M, Tuszynski
    M, Mayford M, Golshani P, Silva AJ. A shared neural ensemble links distinct
    contextual memories encoded close in time. *Nature.* 2016 Jun 2;534(7605):115-8.
    doi: 10.1038/nature17955. Epub 2016 May 23. PMID: 27251287; PMCID: PMC5063500.

[^3]:
    Aharoni D, Hoogland TM. Circuit Investigations With Open-Source Miniaturized
    Microscopes: Past, Present and Future. *Front Cell Neurosci.* 2019 Apr 5;13:141.
    doi: 10.3389/fncel.2019.00141. PMID: 31024265; PMCID: PMC6461004.

[^4]:
    Helmchen F, Fee MS, Tank DW, Denk W. A miniature head-mounted two-photon
    microscope. high-resolution brain imaging in freely moving animals. *Neuron.* 2001
    Sep 27;31(6):903-12. doi: 10.1016/s0896-6273(01)00421-4. PMID: 11580892.

[^5]:
    Zong W, Wu R, Li M, Hu Y, Li Y, Li J, Rong H, Wu H, Xu Y, Lu Y, Jia H, Fan M, Zhou
    Z, Zhang Y, Wang A, Chen L, Cheng H. Fast high-resolution miniature two-photon
    microscopy for brain imaging in freely behaving mice. *Nat Methods.* 2017
    Jul;14(7):713-719. doi: 10.1038/nmeth.4305. Epub 2017 May 29. PMID: 28553965.

### Acquisition Tools

Dr. Daniel Aharoni's lab has developed iterations of the UCLA Miniscope platform. Based
on interviews, we have found labs using the two most recent versions including
[Miniscope DAQ V3](http://miniscope.org/index.php/Information_on_the_(previous_Version_3)_Miniscope_platform)
and [Miniscope DAQ V4](https://github.com/Aharoni-Lab/Miniscope-v4/wiki). Labs also use
the Bonsai OpenEphys tool for data acquisition with the UCLA miniscope. Inscopix
provides the Inscopix Data Acquisition Software (IDAS) for the nVista and nVoke systems.

### Preprocessing Tools

The preprocessing workflow for miniscope imaging includes denoising, motion correction,
cell segmentation, and calcium event extraction (sometimes described as "deconvolution"
or "spike inference"). For the UCLA Miniscopes, the following [analysis
packages](https://github.com/Aharoni-Lab/Miniscope-v4/wiki/Analysis-Packages) are
commonly used:

- [Miniscope Denoising](https://github.com/Aharoni-Lab/Miniscope-v4/wiki/Removing-Horizontal-Noise-from-Recordings),
  Daniel Aharoni (UCLA), Python
- [NoRMCorre](https://github.com/flatironinstitute/NoRMCorre), Flatiron Institute, MATLAB
- [CNMF-E](https://github.com/zhoupc/CNMF_E), Pengcheng Zhou (Liam Paninski's Lab, Columbia
  University), MATLAB
- [CaImAn](https://github.com/flatironinstitute/CaImAn), Flatiron Institute, Python
- [miniscoPy](https://github.com/PeyracheLab/miniscoPy), Guillaume Viejo (Adrien Peyrache's
  Lab, McGill University), Python
- [MIN1PIPE](https://github.com/JinghaoLu/MIN1PIPE), Jinghao Lu (Fan Wang's Lab, MIT), MATLAB
- [CIAtah](https://github.com/bahanonu/ciatah), Biafra Ahanonu, MATLAB
- [MiniAn](https://github.com/DeniseCaiLab/minian), Phil Dong (Denise Cai's Lab, Mount Sinai),
  Python
- [MiniscopeAnalysis](https://github.com/etterguillaume/MiniscopeAnalysis), Guillaume Etter
  (Sylvain Williams' Lab, McGill University), MATLAB
- [PIMPN](https://github.com/etterguillaume/PIMPN), Guillaume Etter (Sylvain Williams's Lab,
  McGill University), Python
- [CellReg](https://github.com/zivlab/CellReg), Liron Sheintuch (Yaniv Ziv's Lab, Weizmann
  Institute of Science), MATLAB
- Inscopix Data Processing Software (IDPS)
- Inscopix Multimodal Image Registration and Analysis (MIRA)

Based on interviews with UCLA and Inscopix miniscope users and developers, each research
lab uses a different preprocessing workflow. These custom workflows are often closed
source and not tracked with version control software. For the preprocessing tools that
are open source, they are often developed by an individual during their training period
and lack funding for long term maintenance. These factors result in a lack of
standardization for miniscope preprocessing tools, which is a major obstacle to adoption
for new labs.
