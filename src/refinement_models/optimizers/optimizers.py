import torch
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR, ExponentialLR

def build_optimizer(model, config):
    name = config['trainer']['optimizer']
    lr = config['trainer']['true_lr']
    backbone_lr_ratio = config['trainer']['backbone_lr_ratio']

    # Filter the backbone param and others param:
    keyword = 'matcher.backbone'
    backbone_params = [param for name, param in list(filter(lambda kv: keyword in kv[0], model.named_parameters()))]
    base_params = [param for name, param in list(filter(lambda kv: keyword not in kv[0], model.named_parameters()))]
    params = [{'params': backbone_params, 'lr': lr * backbone_lr_ratio}, {'params': base_params}]

    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=config['trainer']['adam_decay'])
    elif name == "adamw":
        # torch.optim.AdamW(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
        return torch.optim.AdamW(params, lr=lr, weight_decay=config['trainer']['adamw_decay'])
    else:
        raise ValueError(f"TRAINER.OPTIMIZER = {name} is not a valid optimizer!")


def build_scheduler(config, optimizer):
    """
    Returns:
        scheduler (dict):{
            'scheduler': lr_scheduler,
            'interval': 'step',  # or 'epoch'
            'monitor': 'val_f1', (optional)
            'frequency': x, (optional)
        }
    """
    scheduler = {'interval': config['trainer']['scheduler_invervel']}
    name = config['trainer']['scheduler']

    if name == 'MultiStepLR':
        scheduler.update(
            {'scheduler': MultiStepLR(optimizer, config['trainer']['mslr_milestones'], gamma=config['trainer']['mslr_gamma'])})
    elif name == 'CosineAnnealing':
        scheduler.update(
            {'scheduler': CosineAnnealingLR(optimizer, config['trainer']['cosa_tmax'])})
    elif name == 'ExponentialLR':
        scheduler.update(
            {'scheduler': ExponentialLR(optimizer, config['trainer']['elr_gamma'])})
    else:
        raise NotImplementedError()

    return scheduler
