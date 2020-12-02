# First U-Net Layers Contain More Domain Specific Information Than The Last Ones

This is the repository with the code of experiments for the paper "First U-Net Layers Contain More Domain Specific Information Than The Last Ones"

https://arxiv.org/abs/2008.07357


### Libraries:

###### 1. Add `damri` to the local python:
```
ln -sfn ~/workspace/domain_adaptation_mri/damri ~/miniconda3/lib/python3.*/site-packages/
``` 
where `*` is the version of your python.

###### 2. Install `deep_pipe`:
```
git clone https://github.com/neuro-ml/deep_pipe.git
cd deep_pipe
git checkout develop
pip install -e .
```

###### 3. Install `surface-distance`:
```
git clone https://github.com/deepmind/surface-distance.git
pip install surface-distance/
```

Original repository: https://github.com/deepmind/surface-distance

###### 4. Python & Torch versions we used:
1) Python: 3.7.6
2) Torch: 1.5.0 

### Experiment Reproduction

1) The path to your local copy of CC359 should be specified here: `config/assets/dataset/cc359.config`

2) To run a single experiment, please follow the steps below:

2.1) First, the experiment structure should be created:
```
python -m dpipe build_experiment --config_path "$1" --experiment_path "$2"
```

where the first argument is a path to the `.config` file e.g., `"~/config/experiments/unet2d/unfreeze_first.config"` and the second argument is a path to the folder where the experiment structure will be organized, e.g. `"~/dart_results/unfreeze_first"`

2.2) Then, to run an experiment please go to the experiment folder inside the created structure (`i` corresponds to the particular experiment, i.e. to the particular source-target pair):
```
cd ~/dart_results/unfreeze_first/experiment_{i} 
```

and call the following command to start the experiment:

```
python -m dpipe run_experiment --config_path "../resources.config"
```

where `resources.config` is the general `.config` file of the experiment.

3) Note that switching on/off the augmentation is controlled by `augm_fn` variable in the `.config` files. The number of scans to fine-tune on is controlled by `n_add_ids`, while the share of available slices from a certain scan by `slice_sampling_interval`
