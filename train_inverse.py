from ast import In
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import Subset
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, LambdaLR
import torchvision

import wandb
import argparse
import numpy as np


class RobotDataset(Dataset):
    def __init__(self, data_path, scale=1.0):
        import pickle
        with open(data_path, 'rb') as f:
            self.robot_data = pickle.load(f)
        self.scale = torch.tensor(scale, dtype=torch.float)
        
    def __len__(self):
        return len(self.robot_data['joint_ctrl'])

    def __getitem__(self, idx):
        joint = self.robot_data['joint_ctrl'][idx]
        endeffector = self.robot_data['endeffector'][idx]

        joint = torch.tensor(joint, dtype=torch.float)
        endeffector = torch.tensor(endeffector, dtype=torch.float)

        return endeffector, joint * self.scale

class StandardScalerSubset(Subset):
    def __init__(self, dataset, indices,
                 mean=None, std=None, eps=10**-9):
        super().__init__(dataset=dataset, indices=indices)
        target_tensor = torch.stack([dataset[i][0] for i in indices])
        target_tensor = target_tensor.to(torch.float)
        if mean is None:
            self._mean = torch.mean(target_tensor, dim=0)
        else:
            self._mean = mean
        if std is None:
            self._std = torch.std(target_tensor, dim=0, unbiased=False)
        else:
            self._std = std
        self._eps = eps
        self.std.apply_(lambda x: max(x, self.eps))

    def __getitem__(self, idx):
        dataset_list = list(self.dataset[self.indices[idx]])
        input = dataset_list[0]
        dataset_list[0] = (input - self.mean) / self.std
        return tuple(dataset_list)

    @property
    def mean(self):
        return self._mean

    @property
    def std(self):
        return self._std

    @property
    def eps(self):
        return self._eps

class InverseKinematics(nn.Module): # [1000, 500, 200, 100, 50], [1000, 500, 200, 200, 100]
    def __init__(self, latent_dims=[1000, 500, 200, 200, 100], input_dim=7, output_dim=7, dropout=0.1):
        super(InverseKinematics, self).__init__()
        self.dropout = dropout

        layers = []
        for latent_dim in latent_dims:
            layers.append(nn.Linear(input_dim, latent_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout))
            input_dim = latent_dim
        layers.append(nn.Linear(input_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)




if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description="train inverse kinematics predictor")

    parser.add_argument('--split-ratio', type=float, default=0.2)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--output-scale', type=float, default=1.)
    parser.add_argument('--data-path', type=str, default='dataset/franka.pkl')
    args = parser.parse_args()

    wandb.init(project='inverse_kinematics')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    robot_dataset = RobotDataset(data_path=args.data_path, scale=args.output_scale)
    threashold = int(len(robot_dataset)*args.split_ratio)
    train_index = range(len(robot_dataset))[threashold:]
    val_index = range(len(robot_dataset))[:threashold]
    train_dataset = StandardScalerSubset(robot_dataset, train_index)
    val_dataset = StandardScalerSubset(robot_dataset, val_index, mean=train_dataset.mean, std=train_dataset.std)

    train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=2)
    val_dataloader = DataLoader(val_dataset, args.batch_size, shuffle=False, num_workers=2)

    model = InverseKinematics().to(device)

    criterion = nn.MSELoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # scheduler = StepLR(optimizer, step_size=20, gamma=0.1)
    scheduler = LambdaLR(optimizer, lr_lambda = lambda epoch: 0.999 ** epoch)
    """ scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_dataloader), eta_min=0,
                                                           last_epoch=-1) """

    print('Training start...')
    for epoch in range(args.epoch):  # loop over the dataset multiple times

        running_loss = 0.0
        model.train()
        for i, data in enumerate(train_dataloader, 0):

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # print(loss)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / (i + 1)
        
        model.eval()
        running_vloss = 0.0
        for i, vdata in enumerate(val_dataloader):
            vinputs, vlabels = vdata
            vinputs, vlabels = vinputs.to(device), vlabels.to(device)
            voutputs = model(vinputs)
            vloss = criterion(voutputs, vlabels)
            running_vloss += vloss
        avg_vloss = running_vloss / (i + 1)
        lr = scheduler.get_last_lr()[0]
        scheduler.step()
        wandb.log({'epoch': epoch, 'train_loss': avg_loss, 'val_loss': avg_vloss, 'lr': lr})
        print(f'Epoch: {epoch}, Loss train: {avg_loss}, valid: {avg_vloss}, lr: {lr}')
    print('Finished Training')