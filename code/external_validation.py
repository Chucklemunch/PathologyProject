"""
File: external_validation.py
Author: Charlie Kotula
Date: 2025-11-26
Description: Externally validates fine-tuned Midnight-12k foundation model on PatchCamelyon's training dataset
"""

import h5py
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from transformers import AutoModel
from sklearn.metrics import roc_auc_score
from torch import nn
from tqdm import tqdm
from PathBinaryClassifier import PathBinaryClassifier
import pickle
from collections import Counter


### Extract and reshape images ###

# Transform taken from hugging-face documentation for Midnight model
transform = v2.Compose(
    [
        v2.Resize(224),
        v2.CenterCrop(224),
        v2.ToImage(), 
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ]
)

# Get images/labels and convert to PyTorch tensors
with h5py.File('../patch_cam/camelyonpatch_level_2_split_test_x.h5', 'r') as f:
    cam_imgs_X = torch.Tensor(f['x'][()] / 255) # scaling pixel values to be between 0 and 1

with h5py.File('../patch_cam/camelyonpatch_level_2_split_test_y.h5', 'r') as f:
    cam_imgs_y = torch.Tensor(f['y'][()])

# Reshape y to be (samples, 1)
cam_imgs_y = cam_imgs_y.reshape((cam_imgs_y.shape[0]))

# Reshape data specify image transformations to suit Midnight model
cam_imgs_X = cam_imgs_X.permute(0, 3, 1, 2)

########################################


### Make PatchCam Dataset/Dataloader ###
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

dataset = PatchCamDataset(cam_imgs_X, cam_imgs_y, transform)
dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=False, num_workers=4)

########################################

### Load model from fine-tuning run ###

# Load backbone
with torch.no_grad():
    backbone = AutoModel.from_pretrained('kaiko-ai/midnight')

# Load config
with open('model_configs/best_hp_gnssbjst.pickle', 'rb') as f:
    config = pickle.load(f)

model = PathBinaryClassifier(backbone)

# Load model state from fine-tuning
state_dict = torch.load('models/PathBinaryClassifier_fine-tuning-cls-patch-embed.pt')

model.load_state_dict(state_dict)

# Put model on GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if device == 'cuda':
    model.to(device)

########################################

### Define and run test function ###
def test(model, test_dataloader):
    """
    Evaluates trained model on test set and logs results to Wandb
    """
    running_correct = 0
    y_true = []
    y_score = []
    y_preds = []
    
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
            y_preds.extend(preds.cpu().numpy())
            y_score.extend(y_probs.cpu().numpy())
            y_true.extend(y.cpu().numpy())


    # Metrics for validation epoch
    test_acc = running_correct / len(test_dataloader.dataset)
    test_auroc = roc_auc_score(y_true, y_score)

    # Check distribution of positive/negative class predictions
    pred_count = Counter(y_preds)
    print('pred counts: ', pred_count)
                                            
    print(f'---- Test Accuracy: {test_acc} ----')
    print(f'---- Test AUROC: {test_auroc} ----')
    
    return test_acc, test_auroc

test_acc, test_auroc = test(model, dataloader)

print('Accuracy: ', test_acc)
print('AUROC: ', test_auroc)

with open('PatchCam-Test-Results.txt', 'w') as f:
    f.write(f'PatchCam Accuracy: {test_acc}\n')
    f.write(f'PatchCam AUROC: {test_auroc}\n')

print('----- External Validation Finished -----')
