class CustomBlock(nn.Module):
    def __init__(
        self,
        latent_dim=512,
        num_heads=8,
        ff_size=1024,
        dropout=0.1,
        **kwargs
    ):
        super().__init__()
        
        # Music self-attention and processing
        # self.music_self_attn = VanillaSelfAttention(latent_dim, num_heads, dropout)
        # self.music_norm1 = nn.LayerNorm(latent_dim)
        # self.music_ffn = FFN(latent_dim, ff_size, dropout, latent_dim)
        # self.music_norm2 = nn.LayerNorm(latent_dim)
        
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
        # Process music with self-attention only (affected by text conditioning but not by motion)
        # music_norm = self.music_norm1(music)
        # music_attn = self.music_self_attn(music_norm, emb, key_padding_mask)
        # music_res1 = music + music_attn
        # music_norm2 = self.music_norm2(music_res1)
        # music_out = music_res1 + self.music_ffn(music_norm2, emb)
        
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
    
class CustomizedBlock(nn.Module):
    def __init__(
        self, 
        latent_dim=512, 
        num_heads=8,
        ff_size=1024,
        dropout=0.1,
        **kwargs
    ):
        super().__init__()
        self.custom_block = CustomBlock(
            latent_dim=latent_dim,
            num_heads=num_heads,
            ff_size=ff_size,
            dropout=dropout,
            **kwargs
        )
        
    def forward(self, x, y, music, emb=None, key_padding_mask=None):
        return self.custom_block(x, y, music, emb, key_padding_mask)