# PathologyProject
**Objective: Fine-tune and evaluate a pathology foundation model**

### Data
- [BreakHis Dataset](https://www.kaggle.com/datasets/ambarish/breakhis): Used for fine-tuning
- [Patch Camelyon (PatchCam) Dataset](https://github.com/basveeling/pcam?tab=readme-ov-file#details): Used for external validation

### Strategy
- Fine tune model from Hugging Face
- [Model: Kaiko-AI's Midnight Model](https://huggingface.co/kaiko-ai/midnight)
- Use BreakHis dataset for fine tuning
- [Specs for Kaiko-AI's leaderboard](https://kaiko-ai.github.io/eva/main/leaderboards/)

### AWS EC2 Setup
- AMI: Deep Learning OSS Nvidia Driver AMI GPU Pytorch 2.8 (Ubuntu 24.04)
- Instance Type: g4dn.xlarge (single GPU)

### Experiments to Run
- Tune/Train on 40x BreakHis, test on other magnifications
- External validation of BreakHis training on PCam (train) dataset
- Tune/Train on PCam
- Explore few-shot learning for different magnification levels?
- Add explainability methods to highlight important features in images?

### Relevant Papers
- [Training state-of-the-art pathology foundation models with orders of magnitude less data](https://papers.miccai.org/miccai-2025/paper/4651_paper.pdf)
- [A foundation model for generalizable cancer diagnosis and survival prediction from histopathological images](https://www.nature.com/articles/s41467-025-57587-y#data-availability)

## Cross-Magnification Evaluation using Classifification Token and Averaged Patch Embeddings for Predictions

Following the documentation for the Kaiko-AI Midnight12k model, I added a classification head on top of the pre-trained backbone that sends embeddings through a single linear layer to make a binary prediction (0=benign; 1=malignant). The embedding extraction from the pre-trained backbone is as follows:

<img width="1352" height="230" alt="image" src="https://github.com/user-attachments/assets/cab77b72-7dd1-4f51-acab-d4a1d8fed307" />

Hyperparameters were chosen by running a [Wandb sweep](https://wandb.ai/team-chucklemunch/PathologyFineTuning/sweeps/gnssbjst?nw=nwusercharliekotula). The model was then fine-tuned by freezing the backbone and only updating weights of the classification head. Details of the training run can be found [here](https://wandb.ai/team-chucklemunch/PathologyFineTuning/runs/z32zvigj?nw=nwusercharliekotula). I evaluated the fine-tuned model on images from each of the magnification levels, 40X, 100X, 200X, and 400X. The results are shown in the above graph. A reference of un-tuned model performance on the 40X images is also included.

<img width="1464" height="1106" alt="image" src="https://github.com/user-attachments/assets/1df3c913-9d1c-4c5a-b261-1921f35bd1cf" />

<img width="1466" height="1090" alt="image" src="https://github.com/user-attachments/assets/8bdf8eac-e7ef-4bd4-be95-d829d0b1a95a" />

Accuracy and AUROCs were all very high, and performance only decreased slightly when predicting on images at magnification levels other than the level the classification head was trained on (40x). This suggests high transfer between magnification tasks.

## External Validation on PCam Dataset

After fine-tuning, I assessed model performance on the PCam dataset. Specifically, the training subset of the dataset, as it has the most images of the train, validation, and test subsets. The results were very poor:

- Accuracy = 0.5179
- AUROC = 0.5493

## Training Directly on PCam (with Hyperparameter Tuning)
- TBD
