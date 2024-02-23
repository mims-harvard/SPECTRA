# Evaluating generalizability of artificial intelligence models for molecular datasets

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

<p align="center">
<img src="img/SPECTRA_overview.png?raw=true" width="700" >
</p>

Note: The use of the word spectral here refers only to the framework for model evaluation and should not be confused with other uses of the term in the context of matrix analysis.

## Definition of terms 

A *molecular sequence property (MSP)* is a property that is either given or calculated inherent to a molecular sequence. *Spectral properties* are MSPs that influence model generalizability. Spectral properties are task and data specific: what may be a spectral property for one data type and task is not one in another data modality and task. For example, for predicting DNA binding motifs, the number of adenines in an input nucleic acid sequence is a molecular sequence property but not a spectral property. For the task of secondary structure prediction, the 3D structure of an amino acid sequence is a MSP which is also a spectral property because the structural motifs present in a train set versus a test set will influence model generalizability. The cross-split overlap between a train-test split is the proportion of samples in the test that share a spectral property with at least one sample in the train.

<p align="center">
<img src="img/term_definitions_fig.png?raw=true" width="700" >
</p>

## Github organization

The datasets used in the study can be divided into two categories: (1) mutational scan datasets (MSD), which comprise a single set of sequences with different mutations and their effect on phenotype, and (2) sequence-to-sequence datasets (SD), which comprise of different sequences and their properties. 

**SPECTRA_MSD**: Contains code and instructions to run SPECTRA for MSD datasets

**SPECTRA_SD**: Contains code and instructions to run SPECTRA for SD datasets

**SPECTRA_FM**: Contains code and instructions to run SPECTRA for foundation models (ESM2 was probed in this study)

**SPECTRA_Investigation**: Contains code and instructions that walks through how we investigated a novel spectral property

## General flow of spectral analysis

The spectral framework consists of three main steps that you will see replicated across directories:

### Defining the spectral properties and cross-split overlap in the given molecular sequencing dataset.

The first step is to define the spectral property of interest. The choice of the spectral property is pivotal for SPECTRA. This property is ideally informed by the biological mechanism of how the sequence encodes the phenotype or cellular characteristic to be predicted. Further, it is important to confirm that the procedure for capturing whether two samples share a spectral property is robust and biologically meaningful. 

For example in SDs, the spectral property of interest is sequence identity. To calculate whether two sequences share sequence identity, we perform a pairwise alignment between input sequences and calculate the proportion of aligned positions to the length of the pairwise alignment. If this proportion is greater than 0.3, then the two sequences share this spectral property. In MSDs, phenotypically meaningful differences are in the scale of single mutations. Thus, using the definition of the spectral property from SDs would underestimate differences between samples. To address this, we represent samples in MSDs by their sample barcode or a string representation of the mutations present in the sample. The spectral property of a sample is its sample barcode. Two samples share this property if their sample barcodes share at least one mutation.
  
### Constructing spectral property graphs and generating SPECTRA splits.

After the spectral property is defined, a spectral property graph (SPG) is constructed where nodes are samples in the input dataset, and edges are between samples that share a spectral property. Finding a split such that no two samples share a spectral property is the same as finding the maximal independent set of the SPG or the maximum set of vertices such that no two nodes share an edge. Finding the maximal independent set is NP-Hard, we approximate it via a greedy random algorithm where we (1) randomly order SPG vertices, (2) choose the first vertex and remove all neighbors, and (3) continue until no vertices remain in the graph. To create an overlap in generated splits, we introduce the spectral parameter to the algorithm. Instead of deleting every neighbor, we delete each neighbor with a probability equal to the spectral parameter. If the spectral parameter is 1, we approximate the maximal independent set; if it is 0, we perform a random split. Given a set of nodes returned by the independent set algorithm, we produce an 80-20 train-test split.  

### Generating the spectral performance curve.

To generate a spectral performance curve, we create splits with spectral parameters between 0 and 1 in 0.05 increments. For each spectral parameter, we generate three splits with different random seeds. We then train and test models on generated splits and plot model test performance versus spectral parameters. The area under this curve is the area under the spectral performance curve (AUSPC). 


## Data

The data used in this study is found in the [Harvard Dataverse page](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/W5UUNN) for the project. Each sub-directory will specify what files are relevant for the running of the relevant scripts.

## Additional Resources

- [Paper]()
- [Project Website](https://zitniklab.hms.harvard.edu/projects/SPECTRA/)

```
@article{spectra,
  title={Evaluating generalizability of artificial intelligence models for molecular datasets},
  author={...},
  journal={bioRxiv},
  year={2024}
}
```

## Questions

Please leave a Github issue or contact Yasha Ektefaie at yasha_ektefaie@g.harvard.edu.
