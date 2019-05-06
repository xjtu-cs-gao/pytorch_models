import os
import torch
import torch.nn as nn
import numpy as np

import __main__, opt


def show_num_params(net):
    params = list(net.parameters())
    k = 0
    for i in params:
        l = 1
        for j in i.size():
            l *= j
        k = k + l

        return k


def savelog(acc, log_dir, TAG=True):
    if TAG:

        with open(log_dir, 'a+') as log:

            log_info = 'Epoch: {epoch:d} | train-loss: {train:.4f} | valid-loss: {valid:.4f} | mse: {mse:.2f} | r2: {r2:.2f}\n'.format(
                **acc)

            log.writelines(log_info)

    elif not TAG:

        with open(log_dir, 'a+') as log:

            log_info = 'Epoch: {epoch:d} | train-loss: {train:.4f} | valid-loss: {valid:.4f} | mse: {mse:.2f} | r2: {r2:.2f}\n'.format(
                **acc)

            log.writelines(log_info)

            hyper_params = 'batch_size: {0} | lr: {1} | random_seed: {2}'.format(
                opt.batch_size,
                opt.lr,
                __main__.SEED
                )

            log.writelines(hyper_params)

    else:
        print('Type dose not exist!')
