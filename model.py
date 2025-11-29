import torch
import torch.nn as nn
import math
import timm


# ============================================================
# ArcFace (Additive Angular Margin Softmax)
# ============================================================
class ArcMarginProduct(nn.Module):
    """
    ArcFace layer that applies:
    cos(theta + m) margin penalty
    """
    def __init__(self, in_features, out_features, s=30.0, m=0.5, easy_margin=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m

        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label=None):
        # Embedding normalization
        cosine = nn.functional.linear(
            nn.functional.normalize(input),
            nn.functional.normalize(self.weight)
        )

        sine = torch.sqrt(1.0 - torch.clamp(cosine ** 2, 0.0, 1.0))
        phi = cosine * self.cos_m - sine * self.sin_m

        if label is None:
            # In inference mode, just return cosine similarities
            return cosine * self.s

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return output * self.s


# ============================================================
# Backbone + Embedding Network
# ============================================================
class FaceModel(nn.Module):
    """
    Backbone (EfficientNet / ViT / Swin) → 512-dim Embedding → ArcFace
    """
    def __init__(self, backbone_name, embedding_size, num_classes, pretrained=True):
        super().__init__()

        # Load backbone model from TIMM
        backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,       # remove classification head
            global_pool="avg"    # use average pooling
        )

        self.backbone = backbone
        feat_dim = backbone.num_features

        # Embedding projection (feat_dim → embedding_size)
        self.embedding = nn.Sequential(
            nn.Linear(feat_dim, embedding_size),
            nn.BatchNorm1d(embedding_size),
            nn.ReLU(inplace=True)
        )

        # ArcFace head
        self.arcface = ArcMarginProduct(
            embedding_size,
            num_classes,
            s=30.0,
            m=0.5
        )

    def forward(self, x, label=None):
        feats = self.backbone(x)              # Features from backbone
        emb = self.embedding(feats)           # 512-dim embedding

        if label is None:
            # Inference → return embedding only
            return emb

        # Training → return ArcFace logits
        logits = self.arcface(emb, label)
        return logits
