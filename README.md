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
- Train on 40x, test on other magnifications
- External validation on PCam (train) dataset
- Explore few-shot learning for different magnification levels?
- Add explainability methods to highlight important features in images?

### Relevant Papers
- [Training state-of-the-art pathology foundation models with orders of magnitude less data](https://papers.miccai.org/miccai-2025/paper/4651_paper.pdf)
- [A foundation model for generalizable cancer diagnosis and survival prediction from histopathological images](https://www.nature.com/articles/s41467-025-57587-y#data-availability)

## Cross-Magnification Evaluation
<img width="1508" height="1110" alt="image" src="https://github.com/user-attachments/assets/1beff89f-1dac-4088-b442-c0624bf42651" />


After conducting a [Wandb sweep](https://wandb.ai/team-chucklemunch/PathologyFineTuning/sweeps/3bh8rw9f?nw=nwusercharliekotula) to determine model hyperparameters, I [fine-tuned](https://wandb.ai/team-chucklemunch/PathologyFineTuning/runs/y5a1r5zw?nw=nwusercharliekotula) the the classification head using the 40X magnification images. Subsequently, I evaluated the fine-tuned model on images from each of the magnification levels, 40X, 100X, 200X, and 400X. The results are shown in the above graph. A reference of un-tuned model performance on the 40X images is also included.

<img width="1478" height="1092" alt="image" src="https://github.com/user-attachments/assets/f557380e-e868-43f1-bf04-305fa5f4db04" />


Additionally, AUROCs were computed across each magnification level. As expected, model performed the best on the 40X magnification images, as this was the magnification level on which it was trained. The low AUROCs on the 100X, 200X, and 400X images suggest poor transfer between the 40X and other magnifcation levels.

## External Validation on PCam Dataset

After fine-tuning, I assessed model performance on the PCam dataset. Specifically, the training subset of the dataset, as it has the most images of the train, validation, and test subsets.
