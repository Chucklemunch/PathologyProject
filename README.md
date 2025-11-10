# PathologyProject
Exploring pathology models
**Objective: Train a pathology model and fine-tune a pathology foundation model**

### Data
- [LC25000 Dataset](https://academictorrents.com/details/7a638ed187a6180fd6e464b3666a6ea0499af4af)
- [BreakHis Dataset](https://www.kaggle.com/datasets/ambarish/breakhis)

### Strategy
- Fine Tune Model from Nature Paper: [A foundation model for generalizable cancer diagnosis and survival prediction from histopathological images
](https://www.nature.com/articles/s41467-025-57587-y#data-availability)
- [Model Repo](https://github.com/Zhcyoung/BEPH)
- Going to try to replicate the fine-tuning example provided in the GitHub repository on the BreakHis dataset. This will require GPUs for distributed training.

### AWS EC2 Setup
- AMI: Deep Learning OSS Nvidia Driver AMI GPU Pytorch 2.8 (Ubuntu 24.04)
- Instance Type: TBD
