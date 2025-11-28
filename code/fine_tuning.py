"""
File: fine_tuning.py
Author: Charlie Kotula
Date: 2025-11-20
Description: Fine tunes Midnight-12k pathology foundation model with hyperparameters determined by Wandb sweep

Run script by specifying which the run name, the Wandb sweep (using sweep id) to use for hyperparameters, and how many training epochs to run e.g. "python fine_tuning.py [BreakHis | PCam] fine-tuning-run-abc n2rtz657 20"
"""

import os
import sys
import gc
import torch
import wandb
import h5py
from PIL import Image
from sklearn.metrics import roc_auc_score
from transformers import AutoModel
from torchvision.transforms import v2
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW, Adam, SGD
from torch import nn
from tqdm import tqdm
from PathBinaryClassifier import PathBinaryClassifier

### Process args and get setup HPs for training###
args = sys.argv[1:]

if len(args) != 4:
    raise Exception(f'Expected 4 arguments for fine_tuning.py, got {len(args)}')
    
# Specify Wandb config/model hyperparameters
wandb_project = 'PathologyFineTuning'
dataset_name = args[0] # Specifies name of dataset
wandb_run_name = args[1]

# Getting HPs from Wandb Sweep
sweep_id = args[2]
epochs = int(args[3])

print('Dataset: ', dataset_name)
print('Run name: ', wandb_run_name)
print('Sweep ID: ', sweep_id)
print('Epochs: ', epochs)

api = wandb.Api()
sweep = api.from_path(f'team-chucklemunch/PathologyFineTuning/sweeps/{sweep_id}')

wandb_run_config = None
best_run = None
best_val_acc = 0

# Selects best run
for run in sweep.runs:
    if run.summary['val_acc'] > best_val_acc:
        best_run = run
        best_val_acc = run.summary['val_acc']

# Get config from best run
wandb_run_config = best_run.config

init_lr = wandb_run_config['init_lr']
opt_choice = wandb_run_config['optimizer']

print('init_lr: ', init_lr)
print('opt_choice: ', opt_choice)
print('epochs: ', epochs)
print('wandb_run_name: ', wandb_run_name)

### Initialize model with optimimal HPs from Wandb sweep ###
# Load model
with torch.no_grad():
    backbone = AutoModel.from_pretrained('kaiko-ai/midnight')

# NEED GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Instantiate model
model = PathBinaryClassifier(backbone).to(device)

# Freeze all layers except for those in classification head
for name, block in model.named_children():
    if name != 'classifier':
        for param in block.parameters():
            param.requires_grad = False

# Setup optimizer based on HP choice
if opt_choice == 'AdamW':
    opt = AdamW(filter(lambda p: p.requires_grad == True, model.parameters()), lr=init_lr)
elif opt_choice == 'Adam':
    opt = Adam(filter(lambda p: p.requires_grad == True, model.parameters()), lr=init_lr)
elif opt_choice == 'SGD':
    opt = SGD(filter(lambda p: p.requires_grad == True, model.parameters()), lr=init_lr)

# Loss function
criterion = nn.CrossEntropyLoss()

# Initialize Wandb run
run = wandb.init(project=wandb_project, config=wandb_run_config, name=wandb_run_name)

##################################################
##### Functions used for training/evaluating #####
def save_model_with_artifact(model, run):
    """
    Saves model state-dict and logs model to Wandb
    """

    # Save model
    path = f'models/PathBinaryClassifier_{run.name}.pt'
    torch.save(model.state_dict(), path)

    # Save artifact to Wandb
    artifact = wandb.Artifact('model', type='model')
    artifact.add_file(path)
    run.log_artifact(artifact)
    
def train(model, 
          train_dataloader, 
          val_dataloader, 
          opt, 
          l,
          run,
          epochs=5, 
          grad_accum=4
        ):
    """
    Trains pathology model
    Returns trained model and best validation accuracy
    """

    # Items to track during training
    epoch_train_loss = []
    epoch_train_acc = []
    epoch_val_acc = []
    best_val_acc = 0

    # For early stopping -- stop after 5 epochs without improving validation accuracy by 0.05
    epochs_without_gain = 0
    
    for i in range(epochs):
        running_loss = 0
        running_correct_train = 0
        running_correct_val = 0
        
        print(f'---- Epoch {i+1}/{epochs}----')
        opt.zero_grad()

        # Training
        model.train() # ensure model is in train mode
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

        # Validation
        model.eval() # ensure model is in evaluation mode 
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

        print(f'---- Epoch {i+1}/{epochs} Val Accuracy: {cur_val_acc} ----')

        # Wandb logging
        run.log({
            'train_loss': cur_train_loss, 
            'train_acc': cur_train_acc,
            'val_acc': cur_val_acc
        })
        
        if cur_val_acc > best_val_acc:
            best_val_acc = cur_val_acc 
            epochs_without_gain = 0 # resets early stopping epoch count
        else:
            epochs_without_gain += 1
            # check if early stopping should occur
            print('epochs_without_gain: ', epochs_without_gain)
            if epochs_without_gain == 5:
                print('Stopping early!!!')
                break
                                                

    print('Best val acc: ', best_val_acc)

    # Save model locally and save artifact on Wandbcl
    save_model_with_artifact(model, run)
    
    return model, best_val_acc

def test(model, test_dataloader, run):
    """
    Evaluates trained model on test set and logs results to Wandb
    """
    running_correct = 0
    y_true = []
    y_score = []
    
    # Evaluating on test dataset
    model.eval() # ensure model is in evaluation mode
    with torch.no_grad():
        for X, y in tqdm(test_dataloader):
            X, y, = X.to(device), y.to(device)
    
            # Get predictions
            output = model(X)
            
            # Count correct predictions
            preds = torch.argmax(output, dim=1)
            running_correct += sum(preds == y).item()

            # For later AUROC computation
            y_probs = torch.softmax(output, dim=1)[:, 1] # Get probabilities of positive class
            y_score.extend(y_probs.cpu().numpy())
            y_true.extend(y.cpu().numpy())   
    
    # Metrics for validation epoch
    test_acc = running_correct / len(test_dataloader.dataset)
    test_auroc = roc_auc_score(y_true, y_score)
    
    print(f'---- Test Accuracy: {test_acc} ----')
    print(f'---- Test AUROC: {test_auroc} ----')


    # Wandb logging
    run.log({'test_acc': test_acc, 'test_auroc': test_auroc})
    
    return

##################################################

#########################
###### Make Datasets #####
#def loader(path):
#    img = Image.open(path)
#    return img
#
#transform = v2.Compose(
#    [
#        v2.Resize(224),
#        v2.CenterCrop(224),
#        v2.ToImage(), 
#        v2.ToDtype(torch.float32, scale=True),
#        v2.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
#    ]
#)
#    
#dataset = ImageFolder(
#    '../images/40X/', 
#    loader=loader,
#    transform=transform
#)
#
## Split data into train/val/test
#num_imgs = len(dataset.samples)
#train_size = int(num_imgs * 0.7)
#val_size = int(num_imgs * 0.15)
#test_size = num_imgs - train_size - val_size
#
#generator = torch.Generator().manual_seed(42)
#
#train_dataset, val_dataset, test_dataset = random_split(
#    dataset=dataset,
#    lengths=[train_size, val_size, test_size],
#    generator=generator
#)
#

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


### Making Datasets ###

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

if dataset_name == 'BreakHis':
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

    print('train: ', train_size)
    print('val: ', val_size)
    print('test: ', test_size)

elif dataset_name == 'PCam':
    print('making datasets')
     # Get images/labels and convert to PyTorch tensors
    with h5py.File('../patch_cam/camelyonpatch_level_2_split_test_x.h5', 'r') as f:
        cam_imgs_X = torch.Tensor(f['x'][()] / 255) # scaling pixel values to be between 0 and 1

    with h5py.File('../patch_cam/camelyonpatch_level_2_split_test_y.h5', 'r') as f:
        cam_imgs_y = torch.Tensor(f['y'][()]).long()
    print('data loaded')

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
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

#########################

#### Run training ####
print('pre train')
model, best_val_acc = train(model, 
      train_dataloader,
      val_dataloader, 
      opt, 
      criterion, 
      run,
      # wandb_project, 
      # wandb_run_config,
      # wandb_run_name, 
      epochs=epochs, 
      grad_accum=4
)

print('Training complete! Best Val accuracy: ', best_val_acc)

#### Run testing ####
test(model, test_dataloader, run)

print('Training and evaluation complete for run: ', run.name)

# Cleanup 
run.finish()
