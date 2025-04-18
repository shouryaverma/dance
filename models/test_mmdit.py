import torch
from mmdit.mmdit_generalized_pytorch import MMDiT

num_blocks = 2

mmdit = MMDiT(
    depth = num_blocks, 
    dim_modalities = (768, 512, 384),
    dim_cond = 256,
    qk_rmsnorm = True
)

# mock inputs

time_cond = torch.randn(2, 256)

text_tokens = torch.randn(2, 512, 768)
text_mask = torch.ones((2, 512)).bool()

lead_tokens = torch.randn(2, 1024, 512)

audio_tokens = torch.randn(2, 256, 384)

# forward

text_tokens, lead_tokens, audio_tokens = mmdit(
    modality_tokens = (text_tokens, lead_tokens, audio_tokens),
    modality_masks = (text_mask, None, None),
    time_cond = time_cond,
)