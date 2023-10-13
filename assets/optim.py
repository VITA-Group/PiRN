import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

def configure_optimizers(recons_model, model, head, lr=0.1, momentum=0.9, lr_milestones=[12,20,24], lr_gamma=0.1):

    paras_wo_bn, paras_only_bn = split_parameters(model)
    print("--------------Optimizer configration [Adam]---------------\n")
    print("Learning Rate: {}".format(lr))

    optimizer = optim.Adam([{
        'params': list(recons_model.parameters()) + paras_wo_bn + [head.kernel],
        'weight_decay': 5e-4
    }, {
        'params': paras_only_bn
    }],
        lr=lr)

    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=lr_milestones, gamma=lr_gamma)

    return optimizer, scheduler

def split_parameters(module):
    params_decay = []
    params_no_decay = []
    for m in module.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            params_no_decay.extend([*m.parameters()])
        elif len(list(m.children())) == 0:
            params_decay.extend([*m.parameters()])
    assert len(list(module.parameters())) == len(params_decay) + len(params_no_decay)
    return params_decay, params_no_decay
