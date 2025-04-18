import torch
import clip
from torch import nn
from utils.utils import *
from models.utils import *
from models.blocks import *
from models.flow_matching import *
from models.flow_nets import *
from models.nets import *

class DuetModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.decoder = InterFlowMatching_Duet(cfg)
        # self.decoder = InterDiffusion_Duet(cfg, sampling_strategy=cfg.STRATEGY)
        
        # Load CLIP model more efficiently
        try:
            #clip_model, _ = clip.load("ViT-L/14@336px", device="cpu", jit=False)
            clip_model, _ = clip.load("ViT-L/14@336px", device="cpu", jit=False, download_root='/scratch/gilbreth/gupta596/MotionGen/Text2DanceAcc/dance/checkpoints')
            
            # Extract required components
            self.token_embedding = clip_model.token_embedding
            self.clip_transformer = clip_model.transformer
            self.positional_embedding = clip_model.positional_embedding
            self.ln_final = clip_model.ln_final
            self.dtype = clip_model.dtype
            
            # Freeze CLIP components
            set_requires_grad(self.clip_transformer, False)
            set_requires_grad(self.token_embedding, False)
            set_requires_grad(self.ln_final, False)
            
            # Free up memory
            del clip_model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error loading CLIP model: {e}")
            raise RuntimeError("Failed to initialize CLIP model")
        
        # Create transformer encoder for CLIP features with improved stability
        clipTransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=768,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True)  # Pre-norm for better stability
        
        self.clipTransEncoder = nn.TransformerEncoder(
            clipTransEncoderLayer,
            num_layers=2)
        
        self.clip_ln = nn.LayerNorm(768)
        
        # Token cache for efficiency
        self._token_cache = {}
    
    def compute_loss(self, batch):
        """Compute training loss"""
        batch = self.text_process(batch)
        losses = self.decoder.compute_loss(batch)
        return losses["total"], losses
    
    def decode_motion(self, batch):
        """Generate motion sequence"""
        batch.update(self.decoder(batch))
        return batch
    
    def forward(self, batch):
        """Forward pass during training"""
        return self.compute_loss(batch)
    
    def forward_test(self, batch):
        """Forward pass during inference"""
        batch = self.text_process(batch)
        batch.update(self.decode_motion(batch))
        return batch
    
    @torch.no_grad()
    def text_process(self, batch):
        """Process text inputs through CLIP and transformer encoder"""
        device = next(self.clip_transformer.parameters()).device
        raw_text = batch["text"]
        
        # Process texts in batch for efficiency
        text_embeddings = []
        batch_size = len(raw_text)
        
        # Check cache for common prompts
        cache_hits = 0
        for i, text in enumerate(raw_text):
            if text in self._token_cache and self._token_cache[text].device == device:
                text_embeddings.append(self._token_cache[text])
                cache_hits += 1
        
        # If all texts were in cache, skip CLIP processing
        if cache_hits == batch_size:
            batch["cond"] = torch.stack(text_embeddings)
            return batch
        
        # Process remaining texts through CLIP
        with torch.no_grad():
            # Tokenize all texts at once
            text_tokens = clip.tokenize(raw_text, truncate=True).to(device)
            
            # Embed tokens
            token_embeds = self.token_embedding(text_tokens).type(self.dtype)
            pe_tokens = token_embeds + self.positional_embedding.type(self.dtype)
            
            # Process through transformer
            transformer_input = pe_tokens.permute(1, 0, 2)
            transformer_output = self.clip_transformer(transformer_input)
            transformer_output = transformer_output.permute(1, 0, 2)
            
            # Apply final layer norm
            clip_features = self.ln_final(transformer_output).type(self.dtype)
        
        # Process through our additional transformer
        enhanced_features = self.clipTransEncoder(clip_features)
        normalized_features = self.clip_ln(enhanced_features)
        
        # Extract conditioning vectors
        argmax_indices = text_tokens.argmax(dim=-1)
        batch_indices = torch.arange(enhanced_features.shape[0], device=device)
        cond_vectors = normalized_features[batch_indices, argmax_indices]
        
        # Update cache with new embeddings (limited to conserve memory)
        if len(self._token_cache) < 100:  # Limit cache size
            for i, text in enumerate(raw_text):
                if text not in self._token_cache:
                    self._token_cache[text] = cond_vectors[i].detach().clone()
        
        # Store results
        batch["cond"] = cond_vectors
        
        return batch
    
    def clear_cache(self):
        """Clear the token cache to free memory"""
        self._token_cache.clear()
        torch.cuda.empty_cache()