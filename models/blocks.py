from .layers import *
from .flash_layers import *

class TransformerBlock(nn.Module):
    def __init__(self,
                 latent_dim=512,
                 num_heads=8,
                 ff_size=1024,
                 dropout=0.,
                 cond_abl=False,
                 **kargs):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.cond_abl = cond_abl

        self.sa_block = VanillaSelfAttention(latent_dim, num_heads, dropout)
        self.ca_block = VanillaCrossAttention(latent_dim, latent_dim, num_heads, dropout, latent_dim)
        self.ffn = FFN(latent_dim, ff_size, dropout, latent_dim)

    def forward(self, x, y, emb=None, key_padding_mask=None):
        h1 = self.sa_block(x, emb, key_padding_mask)
        h1 = h1 + x
        h2 = self.ca_block(h1, y, emb, key_padding_mask)
        h2 = h2 + h1
        out = self.ffn(h2, emb)
        out = out + h2
        return out

class DoubleTransformerBlock(nn.Module):
    def __init__(self,
                 latent_dim=512,
                 num_heads=8,
                 ff_size=1024,
                 dropout=0.,
                 cond_abl=False,
                 **kargs):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.cond_abl = cond_abl

        self.sa_block = VanillaSelfAttention(latent_dim, num_heads, dropout)
        self.ca_block_1 = VanillaCrossAttention(latent_dim, latent_dim, num_heads, dropout, latent_dim)
        self.ca_block_2 = VanillaCrossAttention(latent_dim, latent_dim, num_heads, dropout, latent_dim)
        self.ffn = FFN(latent_dim, ff_size, dropout, latent_dim)

    def forward(self, x, y, music, emb=None, key_padding_mask=None):
        h1 = self.sa_block(x, emb, key_padding_mask)
        h1 = h1 + x
        h2 = self.ca_block_1(h1, music, emb, key_padding_mask)
        h2 = h2 + h1
        h3 = self.ca_block_2(h1, y, emb, key_padding_mask)
        h3 = h3 + h2
        out = self.ffn(h3, emb)
        out = out + h3
        return out

class VanillaCustomBlock(nn.Module):
    def __init__(
        self,
        latent_dim=512,
        num_heads=8,
        ff_size=1024,
        dropout=0.1,
        **kwargs
    ):
        super().__init__()
        
        # Dancer self-attention
        self.dancer_a_self_attn = VanillaSelfAttention(latent_dim, num_heads, dropout)
        self.dancer_b_self_attn = VanillaSelfAttention(latent_dim, num_heads, dropout)
        self.dancer_a_norm1 = nn.LayerNorm(latent_dim)
        self.dancer_b_norm1 = nn.LayerNorm(latent_dim)
        
        # Cross-attention: music → dancers (one-way)
        self.music_to_a_attn = VanillaCrossAttention(latent_dim, latent_dim, num_heads, dropout, latent_dim)
        self.music_to_b_attn = VanillaCrossAttention(latent_dim, latent_dim, num_heads, dropout, latent_dim)
        self.dancer_a_norm2 = nn.LayerNorm(latent_dim)
        self.dancer_b_norm2 = nn.LayerNorm(latent_dim)
        
        # Cross-attention: dancers interact
        self.a_to_b_attn = VanillaCrossAttention(latent_dim, latent_dim, num_heads, dropout, latent_dim)
        self.b_to_a_attn = VanillaCrossAttention(latent_dim, latent_dim, num_heads, dropout, latent_dim)
        self.dancer_a_norm3 = nn.LayerNorm(latent_dim)
        self.dancer_b_norm3 = nn.LayerNorm(latent_dim)
        
        # Feedforward networks for dancers
        self.dancer_a_ffn = FFN(latent_dim, ff_size, dropout, latent_dim)
        self.dancer_b_ffn = FFN(latent_dim, ff_size, dropout, latent_dim)
        self.dancer_a_norm4 = nn.LayerNorm(latent_dim)
        self.dancer_b_norm4 = nn.LayerNorm(latent_dim)
        
    def forward(self, x, y, music, emb=None, key_padding_mask=None):
        
        # Process dancers with self-attention
        x_norm1 = self.dancer_a_norm1(x)
        y_norm1 = self.dancer_b_norm1(y)
        x_self = x + self.dancer_a_self_attn(x_norm1, emb, key_padding_mask)
        y_self = y + self.dancer_b_self_attn(y_norm1, emb, key_padding_mask)
        
        # Apply music conditioning to dancers (one-way)
        x_norm2 = self.dancer_a_norm2(x_self)
        y_norm2 = self.dancer_b_norm2(y_self)
        x_music = x_self + self.music_to_a_attn(x_norm2, music, emb, key_padding_mask)
        y_music = y_self + self.music_to_b_attn(y_norm2, music, emb, key_padding_mask)
        
        # Dancers interact with each other
        x_norm3 = self.dancer_a_norm3(x_music)
        y_norm3 = self.dancer_b_norm3(y_music)
        x_interact = x_music + self.b_to_a_attn(x_norm3, y_music, emb, key_padding_mask)
        y_interact = y_music + self.a_to_b_attn(y_norm3, x_music, emb, key_padding_mask)
        
        # Apply feedforward networks to dancers
        x_norm4 = self.dancer_a_norm4(x_interact)
        y_norm4 = self.dancer_b_norm4(y_interact)
        x_final = x_interact + self.dancer_a_ffn(x_norm4, emb)
        y_final = y_interact + self.dancer_b_ffn(y_norm4, emb)
        
        return x_final, y_final, music
    
class VanillaCustomizedBlock(nn.Module):
    def __init__(
        self, 
        latent_dim=512, 
        num_heads=8,
        ff_size=1024,
        dropout=0.1,
        **kwargs
    ):
        super().__init__()
        self.custom_block = VanillaCustomBlock(
            latent_dim=latent_dim,
            num_heads=num_heads,
            ff_size=ff_size,
            dropout=dropout,
            **kwargs
        )
        
    def forward(self, x, y, music, emb=None, key_padding_mask=None):
        return self.custom_block(x, y, music, emb, key_padding_mask)

class FlashCustomBlock(nn.Module):
    """Enhanced version of CustomBlock using Flash Attention"""
    def __init__(
        self,
        latent_dim=512,
        num_heads=8,
        ff_size=1024,
        dropout=0.1,
        **kwargs
    ):
        super().__init__()
       
        # Dancer self-attention with Flash Attention
        self.dancer_a_self_attn = FlashSelfAttention(latent_dim, num_heads, dropout)
        self.dancer_b_self_attn = FlashSelfAttention(latent_dim, num_heads, dropout)
        self.dancer_a_norm1 = nn.LayerNorm(latent_dim)
        self.dancer_b_norm1 = nn.LayerNorm(latent_dim)
       
        # Cross-attention: music → dancers (one-way) with Flash Attention
        self.music_to_a_attn = FlashCrossAttention(latent_dim, latent_dim, num_heads, dropout, latent_dim)
        self.music_to_b_attn = FlashCrossAttention(latent_dim, latent_dim, num_heads, dropout, latent_dim)
        self.dancer_a_norm2 = nn.LayerNorm(latent_dim)
        self.dancer_b_norm2 = nn.LayerNorm(latent_dim)
       
        # Cross-attention: dancers interact with Flash Attention
        self.a_to_b_attn = FlashCrossAttention(latent_dim, latent_dim, num_heads, dropout, latent_dim)
        self.b_to_a_attn = FlashCrossAttention(latent_dim, latent_dim, num_heads, dropout, latent_dim)
        self.dancer_a_norm3 = nn.LayerNorm(latent_dim)
        self.dancer_b_norm3 = nn.LayerNorm(latent_dim)
       
        # Feedforward networks for dancers
        self.dancer_a_ffn = FFN(latent_dim, ff_size, dropout, latent_dim)
        self.dancer_b_ffn = FFN(latent_dim, ff_size, dropout, latent_dim)
        self.dancer_a_norm4 = nn.LayerNorm(latent_dim)
        self.dancer_b_norm4 = nn.LayerNorm(latent_dim)
       
    def forward(self, x, y, music, emb=None, key_padding_mask=None):
        # Process dancers with self-attention
        x_norm1 = self.dancer_a_norm1(x)
        y_norm1 = self.dancer_b_norm1(y)
        x_self = x + self.dancer_a_self_attn(x_norm1, emb, key_padding_mask)
        y_self = y + self.dancer_b_self_attn(y_norm1, emb, key_padding_mask)
       
        # Apply music conditioning to dancers (one-way)
        x_norm2 = self.dancer_a_norm2(x_self)
        y_norm2 = self.dancer_b_norm2(y_self)
        x_music = x_self + self.music_to_a_attn(x_norm2, music, emb, key_padding_mask)
        y_music = y_self + self.music_to_b_attn(y_norm2, music, emb, key_padding_mask)
       
        # Dancers interact with each other
        x_norm3 = self.dancer_a_norm3(x_music)
        y_norm3 = self.dancer_b_norm3(y_music)
        x_interact = x_music + self.b_to_a_attn(x_norm3, y_music, emb, key_padding_mask)
        y_interact = y_music + self.a_to_b_attn(y_norm3, x_music, emb, key_padding_mask)
       
        # Apply feedforward networks to dancers
        x_norm4 = self.dancer_a_norm4(x_interact)
        y_norm4 = self.dancer_b_norm4(y_interact)
        x_final = x_interact + self.dancer_a_ffn(x_norm4, emb)
        y_final = y_interact + self.dancer_b_ffn(y_norm4, emb)
       
        return x_final, y_final, music

class FlashCustomizedBlock(nn.Module):
    """Wrapper for FlashCustomBlock"""
    def __init__(
        self,
        latent_dim=512,
        num_heads=8,
        ff_size=1024,
        dropout=0.1,
        **kwargs
    ):
        super().__init__()
        self.custom_block = FlashCustomBlock(
            latent_dim=latent_dim,
            num_heads=num_heads,
            ff_size=ff_size,
            dropout=dropout,
            **kwargs
        )
       
    def forward(self, x, y, music, emb=None, key_padding_mask=None):
        return self.custom_block(x, y, music, emb, key_padding_mask)