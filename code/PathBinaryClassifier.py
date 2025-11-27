import torch
from torch import nn

class PathBinaryClassifier(nn.Module):
    def __init__(self, backbone):
        super().__init__()

        self.backbone = backbone
        self.dim = backbone.config.hidden_size * 2 # 3072 for Midnight Model

        # Classifier block
        self.classifier = nn.Sequential(
            nn.Linear(self.dim, 2)
        )

    def extract_classification_embedding(self, tensor):
        cls_embedding, patch_embeddings = tensor[:, 0, :], tensor[:, 1:, :]
        return torch.cat([cls_embedding, patch_embeddings.mean(1)], dim=-1)

    def forward(self, X):
        out = self.extract_classification_embedding(self.backbone(X).last_hidden_state)
        logit = self.classifier(out)
        return logit