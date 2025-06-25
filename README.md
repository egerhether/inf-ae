# Reproducibiltiy study of Infinite Recommendation Networks (∞-AE)

This is the codebase for the reproducibility study of [∞-AE](https://arxiv.org/abs/2206.02626) paper. It contains the original code from the paper with the following extensions:
 * Reproducible preprocessing with a fixed seed
 * New metrics matching the RecBole standard
 * New diversity metrics
 * New method of sampling negatives (more information in [Hyperparameters](#Hyperparameters)) for AUC computation
 * Strong generalization approach to preprocessing and data loading
 * Approaches dealing with cold users and related experimental data
 * RecBole configuration files for baseline models (in progress)

## Installation

To install the working python virtual environment use the following commands
```
# Create a clean environment
conda create -n inf-ae python=3.9

conda activate inf-ae

# Install JAX with CUDA support from conda-forge
conda install -c conda-forge "jaxlib=*=*cuda*" jax numpy=1.24 scipy

# Install other basic dependencies
conda install -c conda-forge matplotlib pandas

# Install project-specific requirements
pip install -r requirements.txt
```

We also include a slurm job file for the purposes of setting up the environment on a cluster.

### Datasets

This repository contains serveral already preprocessed datasets ready for inference. If you wish to preprocess data yourself, we recommend sourcing datasets (except Douban) from [RecSysDatasets](https://github.com/RUCAIBox/RecSysDatasets). Each dataset (under `conversion_tools/usage`) will have instructions on preprocessing the raw data into format understood by the `preprocess.py` script. After converting datasets using these instructions place them under `data/<dataset_name>` under names `<dataset_name>.inter` (and if dataset contains it) `<dataset_name>_original.item`.

For Douban, original authors use a different source of the datasets than the repository we use. For that reason you can obtain the dataset at [Kaggle](https://www.kaggle.com/datasets/fengzhujoey/douban-datasetratingreviewside-information). Dataset used in the original paper and thus in our reproduction is the movie portion in the source above, so please download `moviereviews_cleaned.txt` and `movies_cleaned.txt`. Place them under `data/douban/douban_dataset` Then run 
```
cd data/douban
python preprocess_movies_only.py
```
to prepare the data for further preprocessing. 

## Usage

The repository contains two runnable scripts `preprocess.py` and `main.py`. The former is responsible for creating necessary data file from RecBole preprocessed datasets, while the latter is performing training and inference on the ∞-AE model. Always run `preprocess.py` before running `main.py`.

### Preprocessing

`preprocess.py` takes two positional arguments: dataset and generalization approach. Dataset argument is required, while if generalization argument is omitted the preprocessing script will default to weak generalization approach. For example to preprocess `steam` dataset using strong generalization we run
```
python preprocess.py steam strong
```
while running
```
python preprocess.py steam weak
```
is equivalent to
```
python preprocess.py steam
```

### Running the model

To run the model with your desired hyperparameter setup (for more information see [Hyperparameters](#Hyperparameters)) run
```
CUDA_VISIBLE_DEVICES=0 python main.py <dataset_name>
```
This will either perform a grid-search for optimal $\lambda$ parameter and run evaluation on the best one, or just run evaluation if the grid-search hyperparameter is passed as `False`.


## Hyperparameters

∞-AE model requires several hyperparameters to be set which we list in the table below.

| Hyperparam | Description |
|------------|-------------|
| `dataset` | str: name of the dataset being trained/evaluated on |
| `item_id` | str: name of the column containing item IDs in the .item file, if present|
| `category_id` | str: name of the column containing categories for an item in the .item file, if present |
| `diversity_metrics` | bool: whether to compute diversity metrics, possible only if .item file is present |
| `float64` | bool: whether the model will use double precision. single precision is used by default |
| `depth` | int: controls depth of the MLP autoencoder |
| `grid_search_lambda` | bool: whether to perform a grid search for best lambda |
| `user_support` | int: number of users to subsample for training, necessary on large datasets. -1 means full train split is used | 
| `seed` | int: random seed used during training and inference |
| `gen` | str: type of generalization used - either `strong` or `weak`. this should match preprocessing used, otherwise results will be subpar |
| `negative_sampling` | str: method used for sampling negatives for AUC calculations. methods supported include `positiveX` and `totalY` where former samples `X * num_positives` while the latter samples `Y` negatives |

## Results

We include experimental results and plots for the ablation study on MovieLens-20M and study into cold start on Steam and MovieLens-1M. Data files, scripts for creating figures and said figures can be found under `results`.
