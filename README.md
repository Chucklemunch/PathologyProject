# PathologyProject
**Objective: Fine-tune and evaluate a pathology foundation model**

### Data
- [BreakHis Dataset](https://www.kaggle.com/datasets/ambarish/breakhis)
- [Patch Camelyon (PatchCam) Dataset](https://github.com/basveeling/pcam?tab=readme-ov-file#details)

### Strategy
- Fine tune model from Hugging Face
- [Model: Kaiko-AI's Midnight Model](https://huggingface.co/kaiko-ai/midnight)
- Use BreakHis and PCam datasets (separately) for fine tuning
- [Specs for Kaiko-AI's leaderboard](https://kaiko-ai.github.io/eva/main/leaderboards/)

### AWS EC2 Setup
- AMI: Deep Learning OSS Nvidia Driver AMI GPU Pytorch 2.8 (Ubuntu 24.04)
- Instance Type: g4dn.xlarge (single GPU)

### Relevant Papers
- [Training state-of-the-art pathology foundation models with orders of magnitude less data](https://papers.miccai.org/miccai-2025/paper/4651_paper.pdf)
- [A foundation model for generalizable cancer diagnosis and survival prediction from histopathological images](https://www.nature.com/articles/s41467-025-57587-y#data-availability)

## BreakHis Fine-tuning and Cross-Magnification Evaluation
Following the documentation for the Kaiko-AI Midnight12k model, I added a classification head on top of the pre-trained backbone that sends embeddings through a single linear layer to make a binary prediction (0=benign; 1=malignant). The embedding extraction from the pre-trained backbone is as follows:

<img width="1352" height="230" alt="image" src="https://github.com/user-attachments/assets/cab77b72-7dd1-4f51-acab-d4a1d8fed307" />

Hyperparameters were chosen by running a [Wandb sweep](https://wandb.ai/team-chucklemunch/PathologyFineTuning/sweeps/gnssbjst?nw=nwusercharliekotula). The model was then fine-tuned by freezing the backbone and only updating weights of the classification head. Details of the training run can be found [here]([https://wandb.ai/team-chucklemunch/PathologyFineTuning/runs/z32zvigj?nw=nwusercharliekotula](https://wandb.ai/team-chucklemunch/PathologyFineTuning/runs/0jsvmoqp?nw=nwusercharliekotula)). I evaluated the fine-tuned model on images from each of the magnification levels, 40X, 100X, 200X, and 400X. The results are shown in the above graph.

<img width="1092" height="684" alt="image" src="https://github.com/user-attachments/assets/533abe79-9065-40de-ad15-e28fc587d18e" />


<img width="1092" height=706" alt="image" src="https://github.com/user-attachments/assets/5b66b29a-5152-4fb9-a94f-c8b18924c812" />


## External Validation on PCam Dataset
After fine-tuning, I assessed model performance on the PCam dataset. Specifically, the training subset of the dataset, as it has the most images of the train, validation, and test subsets. The results were very poor:

- Accuracy = 0.5168
- AUROC = 0.5456

## Training Directly on PCam (with Hyperparameter Tuning)
I also fine-tuned a separate classifier on the PatchCamelyon (PCam) dataset. Because the dataset is very large (327,680 images), I used a 10% (32,768 images) of the PCam training dataset to create my own train, validation, and testing sets. Details of the training run can be found [here](https://wandb.ai/team-chucklemunch/PathologyFineTuning/runs/zbx03t1f?nw=nwusercharliekotula). 

- Accuracy = 0.9833
- AUROC = 0.9994
