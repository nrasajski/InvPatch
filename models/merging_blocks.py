import torch
from transformers import GPT2LMHeadModel
import torch.nn.functional as F
import pickle
import torch.nn as nn
from typing import Tuple, Any
from string import punctuation
from timm.layers.weight_init import trunc_normal_
from typing import OrderedDict
from models.merging_vit import CustomViT

"""
    Code for patch merging transformer is based on https://github.com/microsoft/VideoX/blob/master/X-CLIP/models/xclip.py
"""

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        # orig_type = x.dtype
        # ret = super().forward(x.type(torch.float32))
        # return ret.type(orig_type)
        return super().forward(x)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class StackedBiLSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(StackedBiLSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=True)

        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.projection = nn.Linear(input_dim, input_dim)


    def forward(self, x):
        x = self.projection(x)
        out, (h_n, c_n) = self.lstm(x)

        h_forward = h_n[-2, :, :]
        h_backward = h_n[-1, :, :]

        h_combined = torch.cat((h_forward, h_backward), dim=1)
        logits = self.fc(h_combined)
        return logits


class WordAttention(nn.Module):
    def __init__(self, embed_size, hidden_size):
        super().__init__()
        self.word_gru = nn.GRU(embed_size, hidden_size, bidirectional=True, batch_first=True)
        self.word_attn = nn.Linear(2 * hidden_size, 2 * hidden_size)
        self.context_vector = nn.Parameter(torch.randn(2 * hidden_size))

    def forward(self, x):
        h, _ = self.word_gru(x)
        u = torch.tanh(self.word_attn(h))
        attn_scores = torch.matmul(u, self.context_vector)
        attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(-1)
        s = torch.sum(h * attn_weights, dim=1)
        return s

def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype=torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


class HANClassifier(nn.Module):
    def __init__(self, embed_size, hidden_size, output_dim):
        super().__init__()
        self.pos_embedding = posemb_sincos_2d(
            h=4,
            w=4,
            dim=1536,
        )
        self.word_attn = WordAttention(embed_size, hidden_size)
        self.fc = nn.Linear(hidden_size * 2, output_dim)

    def forward(self, x):
        x = x + self.pos_embedding.to(x.device, dtype=x.dtype)
        s = self.word_attn(x)  
        out = self.fc(s)  
        return out


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class MultiframeIntegrationTransformer(nn.Module):
    def __init__(self, T, embed_dim=512, out_dim=768, layers=1,):
        super().__init__()
        self.T = T
        transformer_heads = embed_dim // 64
        self.positional_embedding = nn.Parameter(torch.empty(1, T, embed_dim))
        trunc_normal_(self.positional_embedding, std=0.02)
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(d_model=embed_dim, n_head=transformer_heads) for _ in range(layers)])

        self.apply(self._init_weights)
        self.out_ln = nn.Linear(embed_dim, out_dim)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Linear,)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward(self, x):
        ori_x = x
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)
        x = self.resblocks(x)
        x = x.permute(1, 0, 2)  
        x = x.type(ori_x.dtype) + ori_x
        
        return self.out_ln(x.mean(dim=1, keepdim=False))

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class PatchMerging(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.reduction = nn.Linear(4 * input_dim, 2 * input_dim, bias=False)
        self.norm = nn.LayerNorm(4 * input_dim)

    def forward(self, x, h, w):

        B, T, L, C = x.shape
        x = x.view(B, T, h, w, C)
        x0 = x[:, :, 0::2, 0::2, :] 
        x1 = x[:, :, 0::2, 1::2, :] 
        x2 = x[:, :, 1::2, 0::2, :] 
        x3 = x[:, :, 1::2, 1::2, :]  
        x = torch.cat([x0, x1, x2, x3], dim=-1)
        H_new, W_new = h // 2, w // 2
        x = x.view(B, T, H_new * W_new, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)

        return x


class AttentionPatchMergingVideo(nn.Module):
    def __init__(self, input_dim, output_dim=None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim or input_dim * 2
        self.attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=3, batch_first=True)
        self.output_proj = nn.Linear(input_dim, self.output_dim)
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x, h, w):
        B, T, L, C = x.shape
        x = x.view(B, T, h, w, C)
        x0 = x[:, :, 0::2, 0::2, :] 
        x1 = x[:, :, 0::2, 1::2, :] 
        x2 = x[:, :, 1::2, 0::2, :] 
        x3 = x[:, :, 1::2, 1::2, :] 
        x_cat = torch.stack([x0, x1, x2, x3], dim=4)
        B_, T_, H_, W_, _, _ = x_cat.shape
        x_cat = x_cat.reshape(B * T * H_ * W_, 4, C)        
        x_cat = self.norm(x_cat)        
        attn_out, _ = self.attn(x_cat, x_cat, x_cat)
        merged = attn_out.mean(dim=1)
        merged = self.output_proj(merged)
        merged = merged.view(B, T, H_ * W_, self.output_dim)
        return merged


class CrossFramelAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, droppath = 0., T=0, ):
        super().__init__()
        self.T = T

        self.message_fc = nn.Linear(d_model, d_model)
        self.message_ln = LayerNorm(d_model)
        self.message_attn = nn.MultiheadAttention(d_model, n_head,)
           
        self.attn = nn.MultiheadAttention(d_model, n_head,)
        self.ln_1 = LayerNorm(d_model)
        
        self.drop_path = DropPath(droppath) if droppath > 0. else nn.Identity()
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]


    def forward(self, x):
        l, bt, d = x.size()
        b = bt // self.T
        x = x.view(l, b, self.T, d) 

        msg_token = self.message_fc(x[0,:,:,:]) 
        msg_token = msg_token.view(b, self.T, 1, d) 
        
        msg_token = msg_token.permute(1,2,0,3).view(self.T, b, d) 
        msg_token = msg_token + self.drop_path(self.message_attn(self.message_ln(msg_token),self.message_ln(msg_token),self.message_ln(msg_token),need_weights=False)[0])
        msg_token = msg_token.view(self.T, 1, b, d).permute(1,2,0,3)
        
        x = torch.cat([x, msg_token], dim=0)
        
        x = x.view(l+1, -1, d)
        x = x + self.drop_path(self.attention(self.ln_1(x)))
        x = x[:l,:,:]
        x = x + self.drop_path(self.mlp(self.ln_2(x)))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, droppath=None, use_checkpoint=False, T=8):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        if droppath is None:
            droppath = [0.0 for i in range(layers)] 
        self.width = width
        self.layers = layers
        
        self.resblocks = nn.Sequential(*[CrossFramelAttentionBlock(width, heads, attn_mask, droppath[i], T) for i in range(layers)])
       
    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class CrossFrameCommunicationTransformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, output_dim: int, total_patches: int,
                 droppath = None, T = 16):
        super().__init__()
        self.output_dim = output_dim

        scale = width ** -0.5
        self.width = width
        self.total_patches = total_patches
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(self.total_patches + 1, width))
        self.ln_pre = LayerNorm(width)

        ## Attention Blocks
        self.transformer = Transformer(width, layers, heads, droppath=droppath, T=T,)
        self.ln_post = LayerNorm(width)
        self.ln_video_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))


    def init_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor):
        x = x.reshape(-1, self.total_patches, self.width)
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)

        cls_x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            cls_x = cls_x @ self.proj
        
        return cls_x, x[:,1:,:]


class Xclip(nn.Module):
    def __init__(self):
        super(Xclip, self).__init__()
        # self.mit = MultiframeIntegrationTransformer(T=16, embed_dim=768, out_dim=768, layers=4)
        self.mit = CustomViT(
                            dim=384,
                            depth=3,
                            heads=12,
                            dim_head=128,
                            mlp_dim=384,
                            out_dim=768
                        )
        self.visual = CrossFrameCommunicationTransformer(
            width=384,
            layers=3,
            heads=12,
            output_dim=384,
            droppath=[x.item() for x in torch.linspace(0, 0.1, 3)],
            T=16,
            total_patches=256
        )
        self.patch_merging = AttentionPatchMergingVideo(input_dim=384, output_dim=384)

    def forward(self, x):

        cls_features, video_features = self.visual(x)
        video_features = self.mit(cls_features.view(x.shape[0], 16, 384))

        return video_features

# Our code is based on https://github.com/microsoft/VideoX/blob/master/X-CLIP/models/xclip.py
# We made changes to allow the use of pre-trained patch embeddings, and added attention based patch merging to reduce computational burden
class InvPatchAggregator(nn.Module):
    def __init__(self, patch_merge_input_dim=768, patch_merge_output_dim=768, T=16, transformer_blocks_p=3,
                 patches_input_dim=768, patches_output_dim=768, heads_patches=4, total_patches=64,
                 transformer_blocks_f=3, heads_frames=4, dim_head_frames=192, in_dim_frames=768, mlp_dim_frames=768,
                 out_dim_frames=768, pos_h=4, pos_w=4):
        """

        :param patch_merge_input_dim: input_dimension of patch embeddings before merging
        :param patch_merge_output_dim: output dimension of patch embeddings after merging
        :param T: number of frames
        :param transformer_blocks_p: number of transformer blocks for cross frame patch attention
        :param patches_input_dim: input size of patches in cross frame attention
        :param patches_output_dim: output size of embeddings after cross frame attention
        :param heads_patches: number of attention heads for cross frame attention
        :param total_patches: number of input patches
        :param transformer_blocks_f: number of transformer blocks for multi-frame integration
        :param heads_frames: number of attention heads for multi-frame integration
        :param dim_head_frames: dimension of attention heads for multi-frame integration
        :param in_dim_frames: input dimension for multi-frame integration
        :param mlp_dim_frames: intermediate dimension for multi-frame integration
        :param out_dim_frames: output dimension for multi-frame integration
        :param pos_h: height of grid for 2D position encodings for multi-frame integration
        :param pos_w: width of grid for 2D position encodings for multi-frame integration
        """
        super(InvPatchAggregator, self).__init__()
        self.cross_frame_dim = patches_output_dim
        self.T = T
        self.mit = CustomViT(
                            dim=in_dim_frames,
                            depth=transformer_blocks_f,
                            heads=heads_frames,
                            dim_head=dim_head_frames,
                            mlp_dim=mlp_dim_frames,
                            out_dim=out_dim_frames,
                            w=pos_w,
                            h=pos_h
                        )
        
        self.visual = CrossFrameCommunicationTransformer(
            width=patches_input_dim,
            layers=transformer_blocks_p,
            heads=heads_patches,
            output_dim=patches_output_dim,
            droppath=[x.item() for x in torch.linspace(0, 0.1, transformer_blocks_p)],
            T=T,
            total_patches=total_patches
        )
        self.patch_merging = AttentionPatchMergingVideo(input_dim=patch_merge_input_dim, output_dim=patch_merge_output_dim)

    def forward(self, x):
        _, num_frames, patch_seq, _ = x.shape
        h, w = int(patch_seq ** 0.5), int(patch_seq ** 0.5)
        x = self.patch_merging(x, h, w)
        cls_features, video_features = self.visual(x)
        video_features = self.mit(cls_features.view(x.shape[0], self.T, self.cross_frame_dim))
        return video_features
