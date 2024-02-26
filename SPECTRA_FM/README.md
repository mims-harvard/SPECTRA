# Applying SPECTRA to foundation models

Foundation models (FM) are typically pre-trained and then applied to a task-specific dataset. FM performance is averaged across datasets to report overall performance. 

In SPECTRA, FMs receive AUSPCs for each task-specific dataset. These AUSPCs are then plotted against the input protein similarity to the pre-training dataset, creating a multi-task spectral performance curve (SPC) of AUSPC versus pre-training and task-specific dataset cross-split overlap. Improvement of FM generalizability will lead to a better multi-task SPC. 

We demonstrate this capability in the ***SPECTRA can help evaluate the generalizability of foundation models in biology*** section of the paper. Specifically, we perform this analysis for ESM2 using the AUSPCs we calculated for ESM2 on the RIF, PZA, INH, Covid, and GFP datasets. We also perform a retrospective analysis on other protein foundation models.

`Uniref50_Similarity.ipynb` contains all code used for this section. 
To run this code, you must download all relevant fasta files (denoted with SPECTRA_FM) from the [repository](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/W5UUNN) on Harvard Dataverse associated with this project. Place all relevant fasta files in this directory to run the notebook.
