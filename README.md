<picture align="center">
  <source media="(prefers-color-scheme: dark)" srcset="https://www.dropbox.com/scl/fi/p4wa5q1z5avmj8y4f4nbp/spectra.svg?rlkey=ixjhi0vsai865ruj16laihw31&st=4xas69iq&raw=1">
  <img alt="Pandas Logo" src="https://www.dropbox.com/scl/fi/p4wa5q1z5avmj8y4f4nbp/spectra.svg?rlkey=ixjhi0vsai865ruj16laihw31&st=g89wrf0y&raw=1">
</picture>

-----------------

# spectra: Python toolkit for the spectral framework for model evaluation

## What is the spectral framework for model evaluation?

Understanding generalizability -- how well a machine learning model performs on unseen data -- is a fundamental challenge for the broad use of computation. Though numerous benchmarks have been developed to assess model performance across datasets, there are still large gaps between model performance during benchmarking and real-world use.

The spectral framework for model evaluation (SPECTRA) is a new framework for evaluating model generalizability. Instead of cross-validation or bootstrapping, SPECTRA, given a model, dataset, and a spectral property definition, generates a series of train-test splits with decreasing overlap, i.e. a spectrum of train-test splits. SPECTRA then plots the model's performance as a function of cross-split overlap generating a spectral performance curve (SPC). More info can be found in the [Background](#background) section.

## Table of Contents

- [Getting started with SPECTRA](#getting-started-with-spectra)
- [How to use SPECTRA](#how-to-use-spectra)
- [SPECTRA tutorials](#spectra-tutorials)
- [Background](#background)
- [Discussion and Development](#discussion-and-development)
- [Features to be released](#features-to-be-released)
- [FAQ](#faq)
- [License](#license)
- [Citing SPECTRA](#citing-spectra)


## Getting started with spectra

To get started use the [Python
Package Index (PyPI)](https://pypi.org/project/spectra).

```sh
pip install spectra
```
## How to use spectra

### Step 1: Define the spectral property, cross-split overlap, and the spectra dataset wrapper

To run spectra you must first define important two abstract classes, Spectra and SpectraDataset. 

SpectraDataset wraps around your input data and defines functions to load in data and retrieve samples by an index.

```python 
from spectra import SpectraDataset 

class [Name]_Dataset(SpectraDataset):
    
    def sample_to_index(self, idx):
        """
        Given a sample, return the data idx
        """
        pass
        
    
    @abstractmethod
    def parse(self, input_file):
        """
        Given a dataset file, parse the dataset file. 
        """
        pass

    @abstractmethod
    def __len__(self):
        """
        Return the length of the dataset
        """
        pass
    
    @abstractmethod
    def __getitem__(self, idx):
        """
        Given a dataset idx, return the element at that index
        """
        pass
```

Spectra implements the user definition of the spectra property and cross split overlap.


```python 
from spectra import Spectra

class [Name]_spectra(spectra):
    
    def spectra_properties(self, sample_one, sample_two):
        '''
            Define this function to return a similarity metric given two samples where the larger the similarity score the more similar the samples. 

            Example: Two small molecules, returns tanimoto similarity.

        '''
        return similarity

    def cross_split_overlap(self, train, test):
        '''
            Define this function to return the overlap between a list of train and test samples.

            Example: Average pairwise similarity between train and test set protein sequences.

        '''
        

        return cross_split_overlap
```
### Step 2: Initialize SPECTRA and precalculate pairwise spectral properties

Initialize SPECTRA, passing in True or False to the binary argument if the spectral property returns a binary or continuous value. Then precalculate the pairwise spectral properties.

```python
init_spectra = [name]_spectra([name]_Dataset, binary = True)
init_spectra.pre_calculate_spectra_properties([name])
```
### Step 3: Initialize SPECTRA and precalculate pairwise spectral properties

Generate SPECTRA splits. The ```generate_spectra_splits``` function takes in 4 important parameters: 
1. ```number_repeats```: the number of times to rerun SPECTRA for the same spectral parameter, the number of repeats must equal the number of seeds as each rerun uses a different seed. 
2. ```random_seed```: the random seeds used by each SPECTRA rerun, [42, 44] indicates two reruns the first of which will use a random seed of 42, the second will use 44. 
3. ```spectra_parameters```: the spectral parameters to run on, they must range from 0 to 1 and be string formatted to the correct number of significant figures to avoid float formatting errors.
4. ```force_reconstruct```: True to force the model to regenerate SPECTRA splits even if they have already been generated.


```python
spectra_parameters = {'number_repeats': 3, 
                      'random_seed': [42, 44, 46],
                      'spectral_parameters': ["{:.2f}".format(i) for i in np.arange(0, 1.05, 0.05)],
                      'force_reconstruct': True,
                                              }

init_spectra.generate_spectra_splits(**spectra_parameters)

```

### Step 4: Investigate generated SPECTRA splits

After SPECTRA has completed, the user should investigate the generated splits. Specifically ensuring that on average the cross-split overlap decreases as the spectral parameter increases. This can be achieved by using ```return_all_split_stats``` to gather the cross_split_overlap, train size, and test size of each generated split. Example outputs can be seen in the tutorials. 

```python
stats = init_spectra.return_all_split_stats()
plt.scatter(stats['SPECTRA_parameter'], stats['cross_split_overlap'])
```

## Spectra tutorials

In the tutorial folder there are jupyter notebooks that outline how to run SPECTRA for the following datasets:
1. [Deep mutational scan datasets](./tutorials/example_DMS.ipynb) from [ProteinGym](https://proteingym.org)
2. [Sequence datasets](./tutorials/example_sequences.ipynb) from [PEER](https://torchprotein.ai/benchmark)
3. [Single-cell perturb datasets](./tutorials/example_single_cell.ipynb) used in the [GEARS model](https://www.nature.com/articles/s41587-023-01905-6)
4. [Small-molecule dataset](./tutorials/example_mol.ipynb) from [Therapeutic Data Commons](https://tdcommons.ai) 

If there are any other tutorials of interest feel free to raise an issue!

## Background

SPECTRA is from a preprint, for more information on the preprint, the method behind SPECTRA, and the initials studies conducted with SPECTRA, check out the paper folder. 

## Discussion and Development

All development discussions take place on GitHub in this repo in the issue tracker. All contributions, bug reports, bug fixes, documentation improvements, enhancements, and ideas are welcome! Issues and merge requests will be monitored daily.

## Features to be released
1. Tutorials on EHR, Image, and Text data.
2. Capabilities to increase speed of SPECTRA by utilizing multiple CPU-cores. 
3. GPU-based spectral property calculations to speed up pairwise spectral property calculations.

## FAQ
1. *Why do generated SPECTRA splits at high spectral parameter (low cross-split overlap) have a much smaller number of samples when compared to the original dataset?*

    Existing datasets used to benchmark models have large amounts of sample to sample similarity. As a result when we create splits that limit cross-split overlap many samples have to be removed as they are too similar to all other samples in the dataset. As cross-split overlap decreases the number of samples decreases as the criteria for inclusion becomes more strict. The amount of decrease in generated SPECTRA splits reflects underlying similarities in the original dataset.

    For example, if I have a dataset of 100 samples and I run SPECTRA with a spectral parameter of 0.05 and my dataset size decreases to 10, then the majority of samples were similar to each other. On the other hand in another dataset of 100 samples, if I run SECTRA with a spectral parameter of 1.0 and my dataset size is 90, then the samples in the original dataset were not very similar to each other to begin with. 

    Now, this is not a bug of SPECTRA: if a dataset has a large amount of sample to sample similarity, then it should not be used to benchmark generalizability in the first place.

2. *I have a foundation model that is pre-trained on a large amount of data. It is not feasible to do pairwise calculations of SPECTRA properties. How can I use SPECTRA?*

    It is still possible to run SPECTRA on the foundation model (FM) and the evaluation dataset. You would use SPECTRA on the evaluation dataset then train and evaluate the foundation model on each SPECTRA split (either through linear probing, fine-tuning, or any other strategy) to calculate the AUSPC. Then you would determine the cross-split overlap between the pre-training dataset and the evaluation dataset. You would repeat this for multiple evaluation datasets, until you could plot FM AUSPC versus cross-split overlap to the evaluation dataset. For more details on what this would look like check out the [publication](https://www.biorxiv.org/content/10.1101/2024.02.25.581982v1), specifically section 5 of the results section. If there is large interest in this FAQ I can release a tutorial on this, just raise an issue! 

3. *I have a foundation model that is pre-trained on a large amount of data and **I do not have access to the pre-training data**. How can I use SPECTRA?*

    This is a bit more tricky but there are [recent publications](https://arxiv.org/abs/2402.03563) that show these foundation models can represent uncertainty in the hidden representations they produce and a model can be trained to predict uncertainty from these representations. This uncertainty could represent the spectral property comparison between the pre-training and evaluation datasets. Though more work needs to be done, porting this work over would allow the application of SPECTRA in these settings. Again if there is large interest in this FAQ I can release a tutorial on this, just raise an issue! 

4. *SPECTRA takes a long time to run is it worth it?*

    The pairwise spectral property comparison is computationally expensive, but only needs to be done once. Generated SPECTRA splits are important resources that should be released to the public so others can utlilize them without spending resources. For more details on the runtime of the method check out the [publication](https://www.biorxiv.org/content/10.1101/2024.02.25.581982v1), specifically section 6 of the results section. The computation can be sped up with cpu cores, which is a feature that will be released.

If there are any other questions please raise them in the issues and I can address them. I'll keep adding to the FAQ as common questions begin to surface.

## License

SPECTRA is under the MIT license found in the LICENSE file in this GitHub repository.

## Citing SPECTRA

Please cite this paper when referring to SPECTRA.

```
@article {spectra,
	author = {Yasha Ektefaie and Andrew Shen and Daria Bykova and Maximillian Marin and Marinka Zitnik and Maha R Farhat},
	title = {Evaluating generalizability of artificial intelligence models for molecular datasets},
	elocation-id = {2024.02.25.581982},
	year = {2024},
	doi = {10.1101/2024.02.25.581982},
	URL = {https://www.biorxiv.org/content/early/2024/02/28/2024.02.25.581982},
	eprint = {https://www.biorxiv.org/content/early/2024/02/28/2024.02.25.581982.full.pdf},
	journal = {bioRxiv}
}
```


