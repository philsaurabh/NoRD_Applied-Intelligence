import torch
import torch.nn as nn
import torch.nn.functional as F

class wrapper(nn.Module):

    def __init__(self, module):

        super(wrapper, self).__init__()

        self.backbone = module
        feat_dim = list(module.children())[-1].in_features
        self.proj_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim)
            )

    def forward(self, x, bb_grad=True):
       # print(self.backbone(x))
        feats, out = self.backbone(x, is_feat=True)
        feat = feats[-1].view(feats[-1].size(0), -1)
        if not bb_grad:
            feat = feat.detach()

        return out, self.proj_head(feat), feat
        
#class wrapper(nn.Module):
#    def __init__(self, backbone, proj_dim=64):
#        super(wrapper, self).__init__()
#        self.backbone = module
#        self.proj_head = nn.Linear(backbone.embed_dim, proj_dim)  # Match dimensions
#
#    def forward(self, x):
#        feat, logits = self.backbone(x, is_feat=True)
#        feat = feat[-1]  # Use the last feature map
#        feat = torch.mean(feat, dim=1)  # Global average pooling
#        proj = self.proj_head(feat)
#        return logits, proj, feat
