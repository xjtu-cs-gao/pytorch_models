# Library
# standard library
import os

# third-party library
import torch

torch.cuda.empty_cache()
SEED = 7
torch.manual_seed(SEED)  # cpu
torch.cuda.manual_seed(SEED)  # gpu

import torch.nn.functional as F
import torch.utils.data as Data
import pandas as pd
import numpy as np

np.random.seed(SEED)

import progressbar
import pickle
from sklearn.metrics import r2_score, mean_squared_error

# home-brew python file
import function
import dataset
import darknet
import opt


# Hyper-params, changed according to net_info.
BATCH_SIZE = 14


# Data directory
test_x_dir = '../data/test_x.npy'
test_y_dir = '../data/test_y.npy'


def load_data(image_dir, label_dir):
    # Prepare dataset
    # data loader for easy mini-batch return in testing

    data = dataset.MyDataset(image_dir=image_dir, label_dir=label_dir, transform=True)

    loader = Data.DataLoader(dataset=data, batch_size=BATCH_SIZE, shuffle=False)

    return loader


# Restore the model and show it.

file_list = list(os.listdir('../train_log/ANN'))

print('Available state files:\n')

state_list, index = [], 0

for file in file_list:

    if ('.pkl' in file) and ('ann' in file):
        state_list.append(file)

        print(index, ': ', file)

        index += 1

target = int(input("\nWhich model do you want to test?\n(select it by serial number) "))

ann_dir = '../train_log/ANN/' + state_list[target]

ann = torch.load(ann_dir)

print('******** ANN ********\n', ann)

ann.cuda()


# Test results directory
os.makedirs('../test_log/ANN', exist_ok=True)
log_dir = '../test_log/ANN/log_' + state_list[target].strip('.pkl').strip('ann_') + '.xlsx'
predictions_dir = '../test_log/ANN/predictions_' + state_list[target].strip('.pkl').strip('ann_') + '.xlsx'


# Prepare dataset.
test_loader = load_data(test_x_dir, test_y_dir)


# Set a progressbar
bar = progressbar.ProgressBar(max_value=len(test_loader))


ann.eval()

with torch.no_grad():
    TAG = True

    acc = {
        'pressure_mse': [],
        'pressure_r2': [],
        'pred_pressure': np.zeros((1, 721)),
    }

    for step, (b_x, b_y) in enumerate(test_loader):
        b_x = b_x.cuda()
        b_y = b_y.cuda()

        # train and test log.
        pred_pressure = ann(b_x)
        # pred_pressure = pred_pressure.squeeze(2)

        mse = mean_squared_error(b_y.cpu().numpy(), pred_pressure.cpu().numpy())
        r2 = r2_score(b_y.cpu().numpy(), pred_pressure.cpu().numpy())

        # save log.
        acc['pressure_mse'].append(mse)
        acc['pressure_r2'].append(r2)
        acc['pred_pressure'] = np.concatenate((acc['pred_pressure'], pred_pressure.cpu().numpy()), 0)

        bar.update(step + 1)

    log = pd.DataFrame(np.array([acc['pressure_mse'], acc['pressure_r2']]).T, columns=['MSE', 'R^2'])
    predictions = pd.DataFrame(acc['pred_pressure'][1:])

    log.to_excel(log_dir, index=False)
    predictions.to_excel(predictions_dir, index=False)
