from utils import *
from torch import nn

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

class AdaLN(nn.Module):

    def __init__(self, latent_dim, embed_dim=None):
        super().__init__()
        if embed_dim is None:
            embed_dim = latent_dim
        self.emb_layers = nn.Sequential(
            # nn.Linear(embed_dim, latent_dim, bias=True),
            nn.SiLU(),
            zero_module(nn.Linear(embed_dim, 2 * latent_dim, bias=True)),
        )
        self.norm = nn.LayerNorm(latent_dim, elementwise_affine=False, eps=1e-6)

    def forward(self, h, emb):
        """
        h: B, T, D
        emb: B, D
        """
        # B, 1, 2D
        emb_out = self.emb_layers(emb)
        # scale: B, 1, D / shift: B, 1, D
        scale, shift = torch.chunk(emb_out, 2, dim=-1)
        h = self.norm(h) * (1 + scale[:, None]) + shift[:, None]
        return h

class FlashSelfAttention(nn.Module):
    """Self-attention layer using Flash Attention algorithm"""
    def __init__(self, latent_dim, num_head, dropout, embed_dim=None):
        super().__init__()
        self.num_head = num_head
        self.head_dim = latent_dim // num_head
        self.scale = self.head_dim ** -0.5
        
        self.norm = AdaLN(latent_dim, embed_dim)
        self.q_proj = nn.Linear(latent_dim, latent_dim)
        self.k_proj = nn.Linear(latent_dim, latent_dim)
        self.v_proj = nn.Linear(latent_dim, latent_dim)
        self.out_proj = nn.Linear(latent_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, emb, key_padding_mask=None):
        """
        x: B, T, D
        emb: B, D
        key_padding_mask: B, T (True for padding positions)
        """
        batch_size, seq_len, _ = x.shape
        x_norm = self.norm(x, emb)
        
        # Project to queries, keys, values
        q = self.q_proj(x_norm)
        k = self.k_proj(x_norm)
        v = self.v_proj(x_norm)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_head, self.head_dim).transpose(1, 2)
        
        # Handle padding mask if provided
        attn_mask = None
        if key_padding_mask is not None:
            # Convert boolean mask to float and reshape for broadcasting
            padding_mask = key_padding_mask.float()
            padding_mask = padding_mask.unsqueeze(1).unsqueeze(2) * -1e9
            attn_mask = padding_mask
        
        # Use PyTorch's scaled_dot_product_attention which can use Flash Attention
        # when available (PyTorch 2.0+)
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            # Will automatically use Flash Attention on supporting hardware
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
            )
        else:
            # Fallback implementation for older PyTorch versions
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            if attn_mask is not None:
                attn_weights = attn_weights + attn_mask
            attn_weights = torch.softmax(attn_weights, dim=-1)
            attn_weights = self.dropout(attn_weights)
            attn_output = torch.matmul(attn_weights, v)
        
        # Reshape back to original dimensions
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, -1
        )
        
        output = self.out_proj(attn_output)
        return output


class FlashCrossAttention(nn.Module):
    """Cross-attention layer using Flash Attention algorithm"""
    def __init__(self, latent_dim, xf_latent_dim, num_head, dropout, embed_dim=None):
        super().__init__()
        self.num_head = num_head
        self.head_dim = latent_dim // num_head
        self.scale = self.head_dim ** -0.5
        
        self.norm = AdaLN(latent_dim, embed_dim)
        self.xf_norm = AdaLN(xf_latent_dim, embed_dim)
        
        self.q_proj = nn.Linear(latent_dim, latent_dim)
        self.k_proj = nn.Linear(xf_latent_dim, latent_dim)
        self.v_proj = nn.Linear(xf_latent_dim, latent_dim)
        self.out_proj = nn.Linear(latent_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, xf, emb, key_padding_mask=None):
        """
        x: B, T, D (query)
        xf: B, N, L (key/value)
        emb: B, D
        key_padding_mask: B, N (True for padding positions)
        """
        batch_size, seq_len, _ = x.shape
        _, kv_seq_len, _ = xf.shape
        
        x_norm = self.norm(x, emb)
        xf_norm = self.xf_norm(xf, emb)
        
        # Project to queries, keys, values
        q = self.q_proj(x_norm)
        k = self.k_proj(xf_norm)
        v = self.v_proj(xf_norm)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, kv_seq_len, self.num_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, kv_seq_len, self.num_head, self.head_dim).transpose(1, 2)
        
        # Handle padding mask if provided
        attn_mask = None
        if key_padding_mask is not None:
            # Convert boolean mask to float and reshape for broadcasting
            padding_mask = key_padding_mask.float()
            padding_mask = padding_mask.unsqueeze(1).unsqueeze(2) * -1e9
            attn_mask = padding_mask
        
        # Use PyTorch's scaled_dot_product_attention which can use Flash Attention
        # when available (PyTorch 2.0+)
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            # Will automatically use Flash Attention on supporting hardware
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
            )
        else:
            # Fallback implementation for older PyTorch versions
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            if attn_mask is not None:
                attn_weights = attn_weights + attn_mask
            attn_weights = torch.softmax(attn_weights, dim=-1)
            attn_weights = self.dropout(attn_weights)
            attn_output = torch.matmul(attn_weights, v)
        
        # Reshape back to original dimensions
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, -1
        )
        
        output = self.out_proj(attn_output)
        return output