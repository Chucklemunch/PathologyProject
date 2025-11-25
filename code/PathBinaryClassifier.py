from torch import nn

class PathBinaryClassifier(nn.Module):
    def __init__(self, backbone, hidden=512, dropout=0.2):
        super().__init__()

        self.backbone = backbone
        self.dim = backbone.config.hidden_size # 1536 for Midnight Model
        self.dropout = dropout

        # Classifier block
        self.classifier = nn.Sequential(
            # nn.Linear(self.dim, hidden),
            # nn.ReLU(),
            # nn.Dropout(dropout),
            # nn.Linear(hidden, 2), # 1 for BCELoss, 2 for CrossEntropyLoss
            nn.Linear(self.dim, 2)
        )

    def forward(self, X):
        out = self.backbone(X)['pooler_output']
        logit = self.classifier(out)
        return logit