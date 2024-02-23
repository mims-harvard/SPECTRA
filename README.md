# SPECTRA: Spectral framework for model evaluation

**Authors**:
- [Yasha Ektefaie](http://michellemli.com)
- [Andrew Shen](https://www.linkedin.com/in/andrew7shen/)
- [Daria Bykova]()
- [Maximillian Marin]()
- [Marinka Zitnik](http://zitniklab.hms.harvard.edu)
- [Maha Farhat](https://scholar.harvard.edu/mahafarhat/home)

## Overview of SPECTRA

Understanding generalizability -- how well a machine learning model performs on unseen data -- is a fundamental challenge for the broad use of computation in biological discovery. Though numerous benchmarks have been developed to assess model performance across datasets, there are still large gaps between model performance during benchmarking and real-world use. 

We introduce the spectral framework for model evaluation (SPECTRA) a new framework for evaluating model generalizability. Given a model, molecular sequencing dataset, and a spectral property definition, SPECTRA generates a series of train-test splits with decreasing overlap, i.e. a spectrum of train-test splits. SPECTRA then plots the model's performance as a function of cross-split overlap generating a spectral performance curve (SPC). We propose the area under this curve (AUSPC), as a new more comprehensive metric for model generalizability.

We apply SPECTRA to 18 sequencing datasets with associated phenotypes ranging from antibiotic resistance in tuberculosis to protein-ligand binding to evaluate the generalizability of 19 state-of-the-art deep learning models, including large language models, graph neural networks, diffusion models, and convolutional neural networks. We show that SB and MB splits provide an incomplete assessment of model generalizability. With SPECTRA, we find as cross-split overlap decreases, deep learning models consistently exhibit a reduction in performance in a task- and model-dependent manner. Although no model consistently achieved the highest performance across all tasks, we show that deep learning models can generalize to previously unseen sequences on specific tasks. SPECTRA paves the way toward a better understanding of how foundation models generalize in biology.

Note: The use of the word spectral here refers only to the framework for model evaluation and should not be confused with other uses of the term in the context of matrix analysis.

## Definition of terms and Github organization

A *molecular sequence property (MSP)* is a property that is either given or calculated inherent to a molecular sequence. *Spectral properties* are MSPs that influence model generalizability. Spectral properties are task and data specific: what may be a spectral property for one data type and task is not one in another data modality and task. For example, for predicting DNA binding motifs, the number of adenines in an input nucleic acid sequence is a molecular sequence property but not a spectral property. For the task of secondary structure prediction, the 3D structure of an amino acid sequence is a MSP which is also a spectral property because the structural motifs present in a train set versus a test set will influence model generalizability. 

The datasets used in the study can be divided into two categories: (1) mutational scan datasets (MSD), which comprise a single set of sequences with different mutations and their effect on phenotype, and (2) sequence-to-sequence datasets (SD), which comprise of different sequences and their properties. 

**SPECTRA_MSD**: Contains code and instructions to run SPECTRA for MSD datasets
**SPECTRA_SD**: Contains code and instructions to run SPECTRA for SD datasets
**SPECTRA_FM**: Contains code and instructions to run SPECTRA for foundation models (ESM2 was probed for the paper)
**SPECTRA_Investigation**: Contains code and instructions that walks through how we investigated a novel spectral property


-- Figure 1 HERE --


The spectral framework consists of three main steps:

1. Defining MSPs and cross-split overlap in each molecular sequencing dataset.
  
2. Constructing MSP graphs and SPECTRA split generation.

3. Generating spectral performance curves.

This repository includes code used to run Spectra on all molecular sequencing datasets considered in the paper <>.
Each molecular sequencing dataset has its own repository containing specific information for how these steps manifest for these datasets.


