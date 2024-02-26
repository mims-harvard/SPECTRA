# Overview of this directory

This directory was used in Section 3 of the paper where we characterized the full spectral performance curve (SPC) of ML models for the RIF, PZA, INH, Covid, and GFP datasets.
This directory has the code used to run SPECTRA and generate SPCs for ESM2, ESM2-finetuned, CNN, and logistic regression when relevant. 
We highlight important scripts in each directory. 

## Utils directory
`generate_co_mutational_graph.py`: Contains the code used to generate the spectral property graph of these datasets. 
Due to space requirements the generated spectral property graphs can be found on the Harvard Dataverse page.

`independent_set_algorithm.py`: Given a spectral property graph, contains the code to run the independent set algorithm to generate SPECTRA splits.
Once splits are generated, the code then generates train/test files using the subset-sum algorithm when appropriate (read the manuscript/supplement for details on that).

It is important to remember, RIF, PZA, INH, Covid, and GFP are MSDs so nodes in the spectral property graph are *unique* sample barcodes, the independent set algorithm
returns a set of sample barcodes with an amount of cross-split overlap controlled by the spectral parameter. Multiple samples can be represented by a sample barcode so there 
is an extra step here to go from sample barcodes to actual samples in train/test splits, which is why steps occur after the independent set algorithm is run.

## Run directory

`run_baseline.py`: This script pulls it all together. For a given dataset, spectral parameter, model, and number (3 SPECTRA splits were generated for the same spectral parameter),
trains the input model on the SPECTRA train/test split, storing results in weights and biases.

## Data directory

Contains all generated SPECTRA splits in the relevant directories. Filenames are in this format: "INH_0.05_MUTATION_SPLIT_0_TEST", meaning
the test SPECTRA split of the INH dataset at a spectral parameter of 0.05, the 1st replicate. Each filename conatains the sample ID along with its phenotype.
This sample ID then maps to the relevant data file that contains the relevant sequences or sample barcode depending on the model input.
*gfp_barcodes* contains the sample ID to sample barcode mapping while *gfp_sequences* contains the sample ID to sample barcode mapping.
The sequence data for RIF, PZA, and INH could not fit on github so are on the Harvard Dataverse page.

## Dataset directory

Contains the universal dataloader used to process sequence data into the input format necessary for the input model. 

## Model directory

Contains the model architectures for the CNN and logistic regression models.



