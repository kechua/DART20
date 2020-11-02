import os

from dpipe.torch import load_model_state


def load_model_state_cv3_wise(architecture, baseline_exp_path):
    val_path = os.path.abspath('.')
    exp = val_path.split('/')[-1]
    n_val = int(exp.split('_')[-1])

    n_fold = n_val // 15
    n_cv_block = n_val % 3

    path_to_pretrained_model = os.path.join(baseline_exp_path,
                                            f'experiment_{n_fold * 3 + n_cv_block}', 'model.pth')
    load_model_state(architecture, path=path_to_pretrained_model)


def load_model_state_fold_wise(architecture, baseline_exp_path):
    val_path = os.path.abspath('.')
    exp = val_path.split('/')[-1]
    n_val = int(exp.split('_')[-1])
    path_to_pretrained_model = os.path.join(baseline_exp_path, f'experiment_{n_val // 5}', 'model.pth')
    load_model_state(architecture, path=path_to_pretrained_model)


def freeze_model(model, exclude_layers=('inconv', )):
    for name, param in model.named_parameters():
        requires_grad = False
        for l in exclude_layers:
            if l in name:
                requires_grad = True
        param.requires_grad = requires_grad


def unfreeze_model(model):
    for params in model.parameters():
        params.requires_grad = True

