#!/usr/bin/env python3
import os
import segmentation_models_pytorch as smp
import torch

from tensorboardX import SummaryWriter

from _init_path import init_path
init_path()

from spacenet6_model.configs import load_config
from spacenet6_model.datasets import get_dataloader
from spacenet6_model.evaluations import get_metrics
from spacenet6_model.models import get_model
from spacenet6_model.solvers import (
    get_loss, get_lr_scheduler, get_optimizer
)
from spacenet6_model.utils import (
    config_filename, dump_git_info, experiment_subdir, git_filename,
    weight_best_filename, weight_epoch_filename,
    
)


def main():
    """
    """

    config = load_config()
    print('successfully loaded config:')
    print(config)

    # prepare dataloaders
    train_dataloader = get_dataloader(config, is_train=True)
    val_dataloader = get_dataloader(config, is_train=False)

    # prepare model to train
    model = get_model(config)

    # prepare metrics and loss
    metrics = get_metrics(config)
    loss = get_loss(config)

    # prepare optimizer with lr scheduler
    optimizer = get_optimizer(config, model)
    lr_scheduler = get_lr_scheduler(config, optimizer)

    # prepare train/val epoch runners
    train_epoch = smp.utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=config.MODEL.DEVICE,
        verbose=True,
    )
    val_epoch = smp.utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=config.MODEL.DEVICE,
        verbose=True,
    )

    # prepare directories to output log/weight files
    exp_subdir = experiment_subdir(config.EXP_ID)
    log_dir = os.path.join(config.LOG_ROOT, exp_subdir)
    weight_dir = os.path.join(config.WEIGHT_ROOT, exp_subdir)
    os.makedirs(log_dir, exist_ok=False)
    os.makedirs(weight_dir, exist_ok=False)

    # prepare tensorboard
    tblogger = SummaryWriter(log_dir)

    # save git hash
    dump_git_info(os.path.join(log_dir, git_filename()))

    # dump config to a file
    with open(os.path.join(log_dir, config_filename()), 'w') as f:
        f.write(str(config))

    # train loop
    best_score = 0
    metric_name = config.EVAL.MAIN_METRIC
    split_id = config.INPUT.TRAIN_VAL_SPLIT_ID

    for i in range(config.SOLVER.EPOCHS):
        lr = optimizer.param_groups[0]['lr']
        print(f'\nEpoch: {i}, lr: {lr}')

        # run train/val for 1 epoch
        train_logs = train_epoch.run(train_dataloader)
        val_logs = val_epoch.run(val_dataloader)

        # save model weight every epoch
        torch.save(
            model.state_dict(),
            os.path.join(weight_dir, weight_epoch_filename(i))
        )

        # save model weight if score updated
        if best_score < val_logs[metric_name]:
            best_score = val_logs[metric_name]
            torch.save(
                model.state_dict(),
                os.path.join(weight_dir, weight_best_filename())
            )
            print('Best val score updated!')

        # log lr to tensorboard
        tblogger.add_scalar('lr', lr, i)
        # log train losses and scores
        for k, v in train_logs.items():
            tblogger.add_scalar(f'split_{split_id}/train/{k}', v, i)
        # log val losses and scores
        for k, v in val_logs.items():
            tblogger.add_scalar(f'split_{split_id}/val/{k}', v, i)

        # update lr for the next epoch
        lr_scheduler.step()

    tblogger.close()


if __name__ == '__main__':
    main()
