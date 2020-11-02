# domain_adaptation_mri

This is the repository with the code of experiments for the paper "First U-Net Layers Contain More Domain Specific Information Than The Last Ones", which has been Accepted to the workshop DART - Domain Adaptation and Representation Transfer - of MICCAI 2020.

https://arxiv.org/abs/2008.07357


### Setup libraries:

###### 1. Add `damri` to the local python:
```
ln -sfn ~/workspace/domain_adaptation_mri/damri ~/miniconda3/lib/python3.*/site-packages/
``` 
where `*` is a version of your python.

###### 2. Install `deep_pipe`:
```
git clone https://github.com/neuro-ml/deep_pipe.git
cd deep_pipe
git checkout develop
pip install -e .
```

###### 3. Install `cluster-utils`:
```
git clone https://github.com/neuro-ml/cluster-utils.git
cd cluster-utils
pip install -e .
```

###### 4. Install `surface-distance`:
1) git clone https://github.com/deepmind/surface-distance.git
2) <preferable_text_editor> surface-distance/surface_distance/__init__.py
3) change _from metrics import *_ __to__ _from .metrics import *_
4) pip install surface-distance/

Original repository: https://github.com/deepmind/surface-distance

###### 5. More:
in home directory create file `.clusterrc` and fill it with
```
{
 "ram": 16,
 "gpu": 1
}

``` 

### Build and run experiments

###### 1. Build-run an experiment

To build and instantly run the experiment use
```
build_run [CONFIG] [EXP_PATH]
```

Example:
```
build_run ~/workspace/domain_adaptation_mri/config/experiments/test.config /nmnt/x3-hdd/experiments/DA_MRI/test
```

You can pass additional options that could be useful:
- `-max [N]` restrict the maximum number of simultaneously running jobs to N.
- `-g [N]` number of GPUs to use. 0 for CPU computations (could be useful 
to debug an exp while all GPUs are unavailable), additionally you should set
 `device='cpu'` in config . 1 is default and utilizes GPU.
 
###### 2. Separately build and run an experiment

Actually, `build_run` executes 2 following commands: `dpipe-build` and `qexp`

1. In case, if you want to build tons of experiments, then submit them with `-max`
restriction, you use `dpipe-build` until you done:) then use `qexp` on the root
directory of all previously built experiments.

2. In case, if your experiment has been crashed because of bug in the code, you
could just fix the code and re-submit experiment with `qexp`. Probably you also 
need to delete `.lock` file in the experiment folder.
(bug un the code, not config, otherwise you should rebuild experiment)  

They have similar syntax:

```
dpipe-build [CONFIG] [EXP_PATH]
qexp [EXP_PATH] [OPTIONS: {-g, -max}]
```

###### 3. Debugging

All logs are being saved in `~/.cache/cluster-utils/logs`, just `cat` it!
