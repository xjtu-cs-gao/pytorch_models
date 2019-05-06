# Library
# standard library
import os
import time

# third-party library
import torch

torch.cuda.empty_cache()
SEED = 7
torch.manual_seed(SEED)  # cpu
torch.cuda.manual_seed(SEED)  # gpu

import torch.nn as nn
import torch.utils.data as Data
import numpy as np

np.random.seed(SEED)

from sklearn.metrics import r2_score, mean_squared_error
import sys

# home-brew python file
import function, dataset, darknet, opt


# Data directory
train_x_dir = '../data/train_x.npy'
valid_x_dir = '../data/valid_x.npy'

train_y_dir = '../data/train_y.npy'
valid_y_dir = '../data/valid_y.npy'

# Log directory
os.makedirs('../train_log/MLP', exist_ok=True)

current_time = time.strftime('%Y-%m-%d %H-%M', time.localtime(time.time()))

log_dir = '../train_log/MLP/model_log-' + current_time + '.txt'

mlp_dir = '../train_log/MLP/mlp_' + current_time + '.pkl'


# Prepare dataset
def load_data(image_dir, label_dir):
    '''
    data loader for every mini-batch return in the training/validation/test set
    '''

    data = dataset.MyDataset(image_dir=image_dir, label_dir=label_dir, transform=True)

    loader = Data.DataLoader(dataset=data, batch_size=opt.batch_size, shuffle=True)

    return loader


# Set up the model.
mlp = darknet.MLP()

print('******** MLP ********\n', mlp)
n_params_mlp = function.show_num_params(mlp)
print("\nTotal number of parameters: {0}\n".format(n_params_mlp))

mlp.cuda()

optimizer = torch.optim.Adam(mlp.parameters(), lr=opt.lr)  # optimize all cnn parameters

loss_func = nn.MSELoss()  # loss function

# Train and validate
TAG = True

train_loader = load_data(train_x_dir, train_y_dir)
valid_loader = load_data(valid_x_dir, valid_y_dir)

for epoch in range(opt.epoch):

    mlp.train()
    train_loss = 0.
    for step, (b_x, b_y) in enumerate(train_loader):  # gives batch data

        b_x = b_x.cuda()
        b_y = b_y.cuda()

        # train the main mlp net.
        pred_pressure = mlp(b_x)
        # pred_pressure = pred_pressure

        loss = loss_func(pred_pressure, b_y)

        train_loss += loss.data.cpu()

        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # back-propagation, compute gradients
        optimizer.step()  # apply gradients

        sys.stdout.write('Epoch ' + str(epoch + 1) + '/' + str(opt.epoch) + ' ' + str(step) + '/' + str(
            len(train_loader)) + ' loss:' + str(loss.data.cpu()) + '\r')
        sys.stdout.flush()    # print

    with torch.no_grad():
        mlp.eval()
        valid_loss = mse = r2 = 0.

        for step, (b_x, b_y) in enumerate(valid_loader):

            b_x = b_x.cuda()
            b_y = b_y.cuda()

            # train and test log.
            pred_pressure = mlp(b_x)
            # pred_pressure = pred_pressure

            valid_loss += loss_func(pred_pressure, b_y).data.cpu()

            mse += mean_squared_error(b_y.cpu().numpy(), pred_pressure.cpu().numpy())
            r2 += r2_score(b_y.cpu().numpy(), pred_pressure.cpu().numpy())

        acc = {
            'mse': mse / len(valid_loader),
            'r2': r2 / len(valid_loader),
            'epoch': epoch + 1,
            'train': train_loss / len(train_loader),
            'valid': valid_loss / len(valid_loader),
        }

        print('\n', acc)

    if epoch + 1 == opt.epoch:
        TAG = False
        function.savelog(acc, log_dir, TAG)

    else:
        function.savelog(acc, log_dir, TAG)

    # save the model.
    if epoch == 0:
        temp_val_loss = valid_loss

    if (epoch > 0) and (valid_loss < temp_val_loss):
        temp_val_loss = valid_loss
        torch.save(mlp, mlp_dir)
    else:
        print('\nThe valid_loss is not improved from: ', temp_val_loss / len(valid_loader))
