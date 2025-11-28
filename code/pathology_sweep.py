"""
File: pathology_sweep.py
Author: Charlie Kotula
Date: 2025-11-18
Description: Performs hyperparameter sweep with Wandb for Midnight-12k pathology foundation model
Args: Must specify dataset for tuning -- "python pathology_sweep.py [BreakHis|PCam]
"""

import os
import gc
import pickle
import torch
import wandb
import sys
import h5py
from PIL import Image
from transformers import AutoModel
from torchvision.transforms import v2
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW, Adam, SGD
from torch import nn
from tqdm import tqdm
from PathBinaryClassifier import PathBinaryClassifier


### Dataset for PatchCam images
class PatchCamDataset(Dataset):
    def __init__(self, X, y, transform):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (self.transform(self.X[idx]), self.y[idx])


def sweep_train():
    """
    Function for hyperparameter sweep using Wandb
    Must specify sweep configuration outside of function call
    """
    with wandb.init() as run:
        config = wandb.config
        opt_choice = config.optimizer
        init_lr = config.init_lr
        criterion = config.criterion

        # Define model based on HP choice
        model = PathBinaryClassifier(
            backbone=backbone, 
        ).to(device)

        # Freeze model params
        for name, block in model.named_children():
            if name != 'classifier':
                for param in block.parameters():
                    param.requires_grad = False

        # Loss Function
        loss_fun = nn.CrossEntropyLoss()

        # Setup optimizer based on HP choice
        if opt_choice == 'AdamW':
            opt = AdamW(filter(lambda p: p.requires_grad == True, model.parameters()), lr=init_lr)
        elif opt_choice == 'Adam':
            opt = Adam(filter(lambda p: p.requires_grad == True, model.parameters()), lr=init_lr)
        elif opt_choice == 'SGD':
            opt = SGD(filter(lambda p: p.requires_grad == True, model.parameters()), lr=init_lr)

        # Train model with HPs choice by sweep
        train(
            model,
            train_dataloader,
            val_dataloader,
            opt,
            loss_fun
        )


def train(model, train_dataloader, val_dataloader, opt, l, epochs=2, grad_accum=4):
    """
    Trains pathology model
    Returns trained model and best validation accuracy
    """
    
    # Use wandb for tracking
    # with wandb.init(project=project, config=config, name=name) as run:
    run = wandb.run # if initialized sweep
    
    epoch_train_loss = []
    epoch_train_acc = []
    epoch_val_acc = []
    best_val_acc = 0
    
    for i in range(epochs):
        running_loss = 0
        running_correct_train = 0
        running_correct_val = 0
        
        print(f'---- Epoch {i+1}/{epochs}----')
        opt.zero_grad()

        # training
        model.train()
        for X, y in tqdm(train_dataloader):
            X, y = X.to(device), y.to(device)

            # Get predictions
            output = model(X)

            # Compute loss and gradients
            loss = l(output, y)
            running_loss += loss
            loss.backward()

            # Count correct predictions
            preds = torch.argmax(output, dim=1)
            running_correct_train += sum(preds == y).item()
            
            # Update parameters after {grad_accum} batches
            if (i+1) % grad_accum == 0:
                opt.step()
                opt.zero_grad()

            opt.step()
            opt.zero_grad()

        # Metrics for training epoch
        cur_train_loss = running_loss / len(train_dataloader.dataset)
        cur_train_acc = running_correct_train / len(train_dataloader.dataset)
        epoch_train_loss.append(cur_train_loss)
        epoch_train_acc.append(cur_train_acc)

        print(f'---- Epoch {i+1}/{epochs} Train Loss: {cur_train_loss} --- Train Accuracy: {cur_train_acc} ----')

        # validation
        model.eval()
        with torch.no_grad():
            for X, y in tqdm(val_dataloader):
                X, y, = X.to(device), y.to(device)
    
                # Get predictions
                output = model(X)
                
                # Count correct predictions
                preds = torch.argmax(output, dim=1)
                
                running_correct_val += sum(preds == y).item()


        # Metrics for validation epoch
        cur_val_acc = running_correct_val / len(val_dataloader.dataset)
        epoch_val_acc.append(cur_val_acc)
        
        if cur_val_acc > best_val_acc:
            best_val_acc = cur_val_acc            
                                                
        print(f'---- Epoch {i+1}/{epochs} Val Accuracy: {cur_val_acc} ----')

        # Wandb logging
        run.log({
            'train_loss': cur_train_loss, 
            'train_acc': cur_train_acc,
            'val_acc': cur_val_acc
        })
    
    print('Best val acc: ', best_val_acc)
    
    return model, best_val_acc


### Parse Args to determine which dataset to use (BreakHis or PCam)
if len(sys.argv) != 2:
    raise Exception('Only expected 1 command line argument')
else:
    data = sys.argv[1] # BreakHis or PCam

# Specify how images should be transformed (same for any dataset)
transform = v2.Compose(
    [
        v2.Resize(224),
        v2.CenterCrop(224),
        v2.ToImage(), 
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ]
)  

# Generator used for random splitting
generator = torch.Generator().manual_seed(42)

# Making datasets
if data == 'BreakHis':
    def loader(path):
        img = Image.open(path)
        return img

    
        
    dataset = ImageFolder(
        '../images/40X/', 
        loader=loader,
        transform=transform
    )

    # Split data into train/val/test
    num_imgs = len(dataset.samples)
    train_size = int(num_imgs * 0.7)
    val_size = int(num_imgs * 0.15)
    test_size = num_imgs - train_size - val_size


    train_dataset, val_dataset, test_dataset = random_split(
        dataset=dataset,
        lengths=[train_size, val_size, test_size],
        generator=generator
    )
    print('exiting: BreakHis')
    exit(0)
elif data == 'PCam':
     # Get images/labels and convert to PyTorch tensors
    with h5py.File('../patch_cam/camelyonpatch_level_2_split_test_x.h5', 'r') as f:
        cam_imgs_X = torch.Tensor(f['x'][()] / 255) # scaling pixel values to be between 0 and 1

    with h5py.File('../patch_cam/camelyonpatch_level_2_split_test_y.h5', 'r') as f:
        cam_imgs_y = torch.Tensor(f['y'][()]).long()

    # Reshape y to be (samples, 1)
    cam_imgs_y = cam_imgs_y.reshape((cam_imgs_y.shape[0]))

    # Reshape data specify image transformations to suit Midnight model
    cam_imgs_X = cam_imgs_X.permute(0, 3, 1, 2)


    dataset = PatchCamDataset(cam_imgs_X, cam_imgs_y, transform)

    # Split data into train/val/test
    num_imgs = len(dataset) // 10 # Take subset of PCam
    remaining_imgs = len(dataset) - num_imgs

    train_size = int(num_imgs * 0.7)
    val_size = int(num_imgs * 0.15)
    test_size = num_imgs - train_size - val_size

    # Subset dataset 
    subset, _ = random_split(
        dataset=dataset,
        lengths=[num_imgs, remaining_imgs],
        generator=generator
    )

    # Split subset into train/val/test datasets
    train_dataset, val_dataset, test_dataset = random_split(
        dataset=subset,
        lengths=[train_size, val_size, test_size],
        generator=generator
    )

    print('train: ', train_size)
    print('val: ', val_size)
    print('test: ', test_size)
else:
    raise Exception('Data argument did not specify BreakHis of PCam dataset')

# Make dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

# Load model
with torch.no_grad():
    backbone = AutoModel.from_pretrained('kaiko-ai/midnight')

# NEED GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Specify Wandb config
project = 'PathologyFineTuning'
config = {
    'architecture': 'Midnight-12k',
    'dataset': 'BreakHis',
    'optimizer': 'AdamW',
    # 'classifier_hidden_dim': 512,
    'epochs': 5,
}

# Define config for Sweep
sweep_config = {
    'method' : 'random',
    'metric' : {'name': 'val_acc', 'goal': 'maximize'},
    'parameters' : {
        'init_lr' : {'min': 1e-7, 'max': 1e-3},
        #'optimizer' : {'values' : ['AdamW', 'Adam', 'SGD']},
        'optimizer' : {'values' : ['AdamW']},
        'criterion' : {'values' : ['CrossEntropy']}
    },
}

# Create sweep
sweep_id = wandb.sweep(sweep_config, project=project)

# Launch sweep
wandb.agent(sweep_id, function=sweep_train, count=2)

# Get best run from sweep and save config
api = wandb.Api()
sweep = api.from_path(f'team-chucklemunch/PathologyFineTuning/sweeps/{sweep_id}')

best_config = None
best_run = None
best_val_acc = 0

# Selects best run
for run in sweep.runs:
    if run.summary['val_acc'] > best_val_acc:
        best_run = run
        best_val_acc = run.summary['val_acc']

# Get/Save config from best run
best_config = best_run.config
best_config['val_acc'] = best_val_acc

with open(f'model_configs/best_hp_{sweep_id}.pickle', 'wb') as f:
    pickle.dump(best_config, f)

# Cleanup 
wandb.finish()
