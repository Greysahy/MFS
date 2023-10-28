from functools import partial
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import timm.models.vision_transformer
import copy

class GCNAggregator(nn.Module):

    def __init__(self,
                 num_classes: int,
                 input_dim: Union[int, None] = None,
                 embed_dim: Union[int, None] = None,):
        super(GCNAggregator, self).__init__()

        ### auto-proj
        self.embed_dim = embed_dim

        ### build one layer structure (with adaptive module)
        self.projector = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        num_joints = 128
        
        A = torch.eye(num_joints) / 100 + 1 / 100
        self.adj1 = nn.Parameter(copy.deepcopy(A))
        self.conv1 = nn.Conv1d(self.embed_dim, self.embed_dim, 1)
        self.batch_norm1 = nn.BatchNorm1d(self.embed_dim)

        self.conv_q1 = nn.Conv1d(self.embed_dim, self.embed_dim // 4, 1)
        self.conv_k1 = nn.Conv1d(self.embed_dim, self.embed_dim // 4, 1)
        self.alpha1 = nn.Parameter(torch.zeros(1))

        ### merge information
        self.param_pool1 = nn.Linear(num_joints, 1)

        #### class predict
        self.dropout = nn.Dropout(p=0.1)
        self.classifier = nn.Linear(self.embed_dim, num_classes)

        self.tanh = nn.Tanh()

    def forward(self, x):
        #x =  self.projector(x)
        hs = x.transpose(1, 2).contiguous()  # B, S', C --> B, C, S
        ### adaptive adjacency
        q1 = self.conv_q1(hs).mean(1)
        k1 = self.conv_k1(hs).mean(1)
        A1 = self.tanh(q1.unsqueeze(-1) - k1.unsqueeze(1))
        A1 = self.adj1 + A1 * self.alpha1
        ### graph convolution
        hs = self.conv1(hs)
        hs = torch.matmul(hs, A1)
        hs = self.batch_norm1(hs)
        ### predict
        hs = self.param_pool1(hs)
        hs = self.dropout(hs)
        hs = hs.flatten(1)
        hs = self.classifier(hs)

        return hs

class FeatureSelector(nn.Module):
    """
    x: (B, N, D) , in VisionTransformer-base16, N = 197, D = 768
    """
    def __init__(self, input_dim: int, num_classes: int, num_select: int):
        super(FeatureSelector, self).__init__()
        self.num_select = num_select

        ### build classifier
        self.num_classes = num_classes
        self.projector = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        logits = self.projector(x)  # (B, C)
        probs = torch.softmax(logits, dim=-1)  # (B, N, C)

        selections = []
        preds_1 = []
        preds_0 = []
        num_select = self.num_select

        for batch in range(logits.shape[0]):
            max_ids, _ = torch.max(probs[batch], dim=-1)  # 每一个token对应的概率最大值
            confs, idxs = torch.sort(max_ids, descending=True)  # 对
            sf = x[batch][idxs[:num_select]]  # (num_select, D)
            nf = x[batch][idxs[num_select:]]  # calculate
            
            preds_1.append(logits[batch][idxs[:num_select]])
            preds_0.append(logits[batch][idxs[num_select:]])
            selections.append(sf)  # (num_selected, D)

        selections = torch.stack(selections) # (B, num_select, D)
        preds_1 = torch.stack(preds_1)
        preds_0 = torch.stack(preds_0)
        return selections, preds_1, preds_0


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., 
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, init_values=0):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        if self.gamma_1 is None:
            x = x + self.drop_path(y)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * y)
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x, attn

class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, out_indices: Union[Sequence, int] = -1,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        self.out_indices = out_indices
        self.fc_norm = norm_layer(embed_dim)
        self.depth = depth
        
        del self.norm  # remove the original norm
            
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        
        proj_layers = [
            torch.nn.Linear(self.embed_dim, self.embed_dim)
            for _ in range(len(self.out_indices) - 1)
        ]
        self.proj_layers = torch.nn.ModuleList(proj_layers)
        self.proj_weights = torch.nn.Parameter(
            torch.ones(len(self.out_indices)).view(-1, 1, 1, 1))
        if len(self.out_indices) == 1:
            self.proj_weights.requires_grad = False
            
        self.feature_selector = FeatureSelector(input_dim=768, num_classes=7, num_select=128)
        self.combiner = GCNAggregator(num_classes=7, input_dim=768, embed_dim=768)

    def forward(self, x, return_attention=False):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        res = []
        for i, blk in enumerate(self.blocks):
            if i == self.depth - 1:
                x, att = blk(x)
            else:
                x, _ = blk(x)
            if i in self.out_indices:
                if i != self.out_indices[-1]:
                    proj_x = self.proj_layers[self.out_indices.index(i)](x)
                else:
                    proj_x = x
#                 res.append(proj_x)
#         res = torch.stack(res)
#         proj_weights = F.softmax(self.proj_weights, dim=0)
#         res = res * proj_weights
#         res = res.sum(dim=0)
        
        # output the mean feature as cls token
#         x_ = res[:, 1:, :].mean(dim=1)  # global pool without cls token
#         x_ = self.fc_norm(x_)
#         outcome_mean = self.head(x_)
        
        outcome_mean = 1
        # select fusion feature by selector and gcn
        x, preds_1, preds_0 = self.feature_selector(x)
        outcome_select = self.combiner(x)
        
        cls_att = att[:, :, 0, 1:].mean(dim=1).detach().clone()
        if return_attention:
            return outcome, outcome_mean, preds_1, preds_0, cls_att
        else:
            return outcome_select, outcome_mean, preds_1, preds_0
        

def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, out_indices=[0, 2, 4, 6, 11], embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


if __name__ == '__main__':
    model = vit_base_patch16()
    x = torch.rand((2, 3, 224, 224))
    out = model(x)
    print(model)
