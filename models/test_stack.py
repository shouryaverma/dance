import torch
import torch.nn as nn
from mmdit.mmdit_generalized_pytorch import MMDiT


class MMDiTStack(nn.Module):
    def __init__(self, n_blocks, depth, dim_modalities, dim_cond, qk_rmsnorm=True):
        super(MMDiTStack, self).__init__()
        self.blocks = nn.ModuleList([
            MMDiT(
                depth=depth,
                dim_modalities=dim_modalities,
                dim_cond=dim_cond,
                qk_rmsnorm=qk_rmsnorm
            )
            for _ in range(n_blocks)
        ])

    def forward(self, modality_tokens, modality_masks, time_cond):
        for block in self.blocks:
            modality_tokens = block(
                modality_tokens=modality_tokens,
                modality_masks=modality_masks,
                time_cond=time_cond
            )
        return modality_tokens


# Parameters
n_blocks = 4  # Number of stacked MMDiT blocks
depth = 2
dim_modalities = (768, 512, 384)
dim_cond = 256

# Initialize stacked MMDiT
mmdit_stack = MMDiTStack(
    n_blocks=n_blocks,
    depth=depth,
    dim_modalities=dim_modalities,
    dim_cond=dim_cond,
    qk_rmsnorm=True
)

# Mock inputs
time_cond = torch.randn(2, dim_cond)
text_tokens = torch.randn(2, 512, 768)
text_mask = torch.ones((2, 512)).bool()
lead_tokens = torch.randn(2, 1024, 512)
audio_tokens = torch.randn(2, 256, 384)

# Forward pass through the stacked blocks
text_tokens, lead_tokens, audio_tokens = mmdit_stack(
    modality_tokens=(text_tokens, lead_tokens, audio_tokens),
    modality_masks=(text_mask, None, None),
    time_cond=time_cond,
)

print(text_tokens.shape, lead_tokens.shape, audio_tokens.shape)