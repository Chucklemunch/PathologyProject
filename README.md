# PathologyProject
Exploring pathology models
**Objective: Train a pathology model and fine-tune a pathology foundation model**

### Data
- [LC25000 Dataset](https://academictorrents.com/details/7a638ed187a6180fd6e464b3666a6ea0499af4af)
- [BreakHis Dataset](https://www.kaggle.com/datasets/ambarish/breakhis)

### Strategy
- Fine tune model from Hugging Face
- [Model: Kaiko-AI's Midnight Model](https://huggingface.co/kaiko-ai/midnight)
- Use BreakHis dataset for fine tuning
- Fine tune separate model for each organ in BreakHis?
- Distributed training on AWS GPUs

### AWS EC2 Setup
- AMI: Deep Learning OSS Nvidia Driver AMI GPU Pytorch 2.8 (Ubuntu 24.04)
- Instance Type: TBD

### Relevant Papers
- [Training state-of-the-art pathology foundation models with orders of magnitude less data](https://papers.miccai.org/miccai-2025/paper/4651_paper.pdf)
- [A foundation model for generalizable cancer diagnosis and survival prediction from histopathological images](https://www.nature.com/articles/s41467-025-57587-y#data-availability)
