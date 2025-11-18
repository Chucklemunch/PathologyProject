"""
File: pathology_sweep.py
Author: Charlie Kotula
Date: 2025-11-18
Description: Performs hyperparameter sweep with Wandb for Midnight-12k pathology foundation model
"""

import os
import gc
import torch
import wandb
from PIL import Image
from transformers import AutoModel
from torchvision.transforms import v2
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW, Adam, SGD
from torch import nn
from tqdm import tqdm
from PathBinaryClassifier import PathBinaryClassifier

def sweep_train():
    """
    Function for hyperparameter sweep using Wandb
    Must specify sweep configuration outside of function call
    """
    with wandb.init() as run:
        config = wandb.config
        hidden_dim = config.classifier_hidden_dim
        opt_choice = config.optimizer
        init_lr = config.init_lr
        dropout = config.dropout
        criterion = config.criterion

        # Define model based on HP choice
        model = PathBinaryClassifier(
            backbone=backbone, 
            hidden=hidden_dim, 
            dropout=dropout
        ).to(device)

        # Freeze model params
        for name, block in model.named_children():
            if name != 'classifier':
                for param in block.parameters():
                    param.requires_grad = False

        # Select Loss Function
        if criterion == 'CrossEntropy':
            loss_fun = nn.CrossEntropyLoss()

        # Setup optimizer based on HP choice
        if opt_choice == 'AdamW':
            opt = AdamW(filter(lambda p: p.requires_grad == True, model.parameters()), lr=2e-5)
        elif opt_choice == 'Adam':
            opt = Adam(filter(lambda p: p.requires_grad == True, model.parameters()), lr=2e-5)
        elif opt_choice == 'SGD':
            opt = SGD(filter(lambda p: p.requires_grad == True, model.parameters()), lr=2e-5)

        # Train model with HPs choice by sweep
        train(
            model,
            train_dataloader,
            val_dataloader,
            opt,
            loss_fun # defined externally
        )


def train(model, train_dataloader, val_dataloader, opt, l, epochs=5, grad_accum=4):
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

        # Wandb logging
        run.log({'train_loss': cur_train_loss, 'train_acc': cur_train_acc})
        
        # validation
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
        run.log({'val_acc': cur_val_acc})
    
    print('Best val acc: ', best_val_acc)
    
    return model, best_val_acc

    
# Making datasets
def loader(path):
    img = Image.open(path)
    return img

transform = v2.Compose(
    [
        v2.Resize(224),
        v2.CenterCrop(224),
        v2.ToImage(), 
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ]
)
    
dataset = ImageFolder(
    'images/40X/', 
    loader=loader,
    transform=transform
)

# Split data into train/val/test
num_imgs = len(dataset.samples)
train_size = int(num_imgs * 0.7)
val_size = int(num_imgs * 0.15)
test_size = num_imgs - train_size - val_size

generator = torch.Generator().manual_seed(42)

train_dataset, val_dataset, test_dataset = random_split(
    dataset=dataset,
    lengths=[train_size, val_size, test_size],
    generator=generator
)

# Make dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

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
    'classifier_hidden_dim': 512,
    'epochs': 2,
}
name = 'trial-run'

# Define config for Sweep
sweep_config = {
    'method' : 'random',
    'metric' : {'name': 'val_acc', 'goal': 'maximize'},
    'parameters' : {
        'classifier_hidden_dim' : {'min': 32, 'max': 1024},
        'init_lr' : {'min': 1e-7, 'max': 1e-3},
        'dropout' : {'min': 0.1, 'max': 0.5},
        'optimizer' : {'values' : ['AdamW', 'Adam', 'SGD']},
        'criterion' : {'values' : ['CrossEntropy']}
    },
}

# Create sweep
sweep_id = wandb.sweep(sweep_config, project=project)

# Launch sweep
wandb.agent(sweep_id, function=sweep_train, count=10)

# Cleanup 
wandb.finish()
