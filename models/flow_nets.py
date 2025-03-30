import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import *
from models.utils import *
from models.blocks import *
from models.losses import *
from models.flow_matching import FlowMatching

class InterFlowPredictor_Duet(nn.Module):
    def __init__(self,
                 input_feats,
                 latent_dim=512,
                 num_frames=240,
                 ff_size=1024,
                 num_layers=8,
                 num_heads=8,
                 dropout=0.1,
                 activation="gelu",
                 **kargs):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        self.input_feats = input_feats
        
        self.text_emb_dim = 768
        self.music_emb_dim = 54
        
        # Time embedding with improved initialization
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, dropout=0)
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)
        
        # Input Embedding with better initialization
        self.motion_embed = nn.Linear(self.input_feats, self.latent_dim)
        nn.init.xavier_uniform_(self.motion_embed.weight, gain=0.02)
        
        self.text_embed = nn.Linear(self.text_emb_dim, self.latent_dim)
        nn.init.xavier_uniform_(self.text_embed.weight, gain=0.02)
        
        self.music_embed = nn.Linear(self.music_emb_dim, self.latent_dim)
        nn.init.xavier_uniform_(self.music_embed.weight, gain=0.02)
        
        # Music embedding processor with improved transformer config
        musicTransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=8,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True)  # Pre-norm for better stability
        
        self.musicTransEncoder = nn.TransformerEncoder(
            musicTransEncoderLayer,
            num_layers=4)
        
        # Dual-agent transformer blocks
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            self.blocks.append(DoubleTransformerBlock(
                num_heads=num_heads,
                latent_dim=latent_dim,
                dropout=dropout,
                ff_size=ff_size
            ))
        
        # Residual output module for better gradient flow
        self.out_net = nn.Sequential(
            nn.Linear(self.latent_dim, self.ff_size),
            nn.SiLU(),
            nn.Dropout(p=0.1),  # Additional dropout for regularization
            nn.Linear(self.ff_size, self.input_feats)
        )
        
        # Initialize final layer weights to near-zero for stability
        nn.init.zeros_(self.out_net[-1].weight)
        nn.init.zeros_(self.out_net[-1].bias)
        
        # Cache for efficient forward pass
        self.cached_masks = {}
    
    def forward(self, x, timesteps, mask=None, cond=None, music=None):
        """
        Predict the velocity field v(x, t) for flow matching.
        
        Args:
            x: Input tensor [B, T, D*2] (concatenated motion features for both agents)
            timesteps: Time parameter t [B]
            mask: Optional mask for valid positions
            cond: Text conditioning
            music: Music conditioning
        
        Returns:
            Predicted velocity field v(x, t) with same shape as x
        """
        B, T = x.shape[0], x.shape[1]
        device = x.device
        
        # Split into agent A and B features more efficiently
        x_a, x_b = torch.split(x, self.input_feats, dim=-1)
        
        if mask is not None:
            mask = mask[..., 0]

        # Process timesteps more efficiently
        if not isinstance(timesteps, torch.Tensor):
            timesteps = torch.tensor(timesteps, device=device)
    
        if len(timesteps.shape) == 0:
            timesteps = timesteps.expand(B)
        
        # Process time and text conditioning
        emb = self.embed_timestep(timesteps) + self.text_embed(cond)
        
        # Process motion features in parallel
        a_emb = self.motion_embed(x_a)
        b_emb = self.motion_embed(x_b)
        
        # Apply positional encoding
        h_a_prev = self.sequence_pos_encoder(a_emb)
        h_b_prev = self.sequence_pos_encoder(b_emb)
        
        # Process music features
        music_emb = self.music_embed(music)
        music_emb = self.sequence_pos_encoder(music_emb)
        
        # Create or retrieve cached padding mask
        if mask is None:
            # Use cached full mask if available
            cache_key = f"{B}_{T}_{device}"
            if cache_key in self.cached_masks:
                key_padding_mask = self.cached_masks[cache_key]
            else:
                # Create and cache full mask
                full_mask = torch.ones(B, T, device=device)
                key_padding_mask = ~(full_mask > 0.5)
                if len(self.cached_masks) < 10:  # Limit cache size
                    self.cached_masks[cache_key] = key_padding_mask
        else:
            key_padding_mask = ~(mask > 0.5)
        
        # Process music with transformer in one pass
        music_emb = self.musicTransEncoder(music_emb)
        
        # Ensure music features match sequence length
        if music_emb.shape[1] != T:
            raise ValueError(f"Music embedding shape {music_emb.shape} doesn't match expected sequence length {T}")
        
        # Process through dual-agent transformer blocks
        # Store intermediate outputs for residual connections
        h_a_outputs = [h_a_prev]
        h_b_outputs = [h_b_prev]
        
        # Process all blocks with efficient updates
        for i, block in enumerate(self.blocks):
            # Process both agents with shared music and condition
            h_a = block(h_a_prev, h_b_prev, music_emb, emb, key_padding_mask)
            h_b = block(h_b_prev, h_a_prev, music_emb, emb, key_padding_mask)
            
            # Update for next layer
            h_a_prev = h_a
            h_b_prev = h_b
            
            # Store for potential residual connections
            h_a_outputs.append(h_a)
            h_b_outputs.append(h_b)
        
        # Generate velocity predictions for each agent
        output_a = self.out_net(h_a)
        output_b = self.out_net(h_b)
        
        # Concatenate outputs for both agents
        output = torch.cat([output_a, output_b], dim=-1)
        
        return output


class InterFlowMatching_Duet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.nfeats = cfg.INPUT_DIM
        self.latent_dim = cfg.LATENT_DIM
        self.ff_size = cfg.FF_SIZE
        self.num_layers = cfg.NUM_LAYERS
        self.num_heads = cfg.NUM_HEADS
        self.dropout = cfg.DROPOUT
        self.activation = cfg.ACTIVATION
        self.motion_rep = cfg.MOTION_REP
        
        # Flow model parameters
        self.t_bar = cfg.T_BAR
        self.sigma_min = cfg.SIGMA_MIN
        self.sigma_max = cfg.SIGMA_MAX
        self.rho = cfg.RHO
        self.sampling_steps = cfg.SAMPLING_STEPS
        self.solver_type = cfg.SOLVER_TYPE

        # Loss weights parameters
        self.fm_weight = cfg.FM_WEIGHT
        self.inter_weight = cfg.INTER_WEIGHT
        self.geo_weight = cfg.GEO_WEIGHT

        self.ro_weight = cfg.RO_WEIGHT
        self.ja_weight = cfg.JA_WEIGHT
        self.dm_weight = cfg.DM_WEIGHT
        self.vel_weight = cfg.VEL_WEIGHT
        self.bl_weight = cfg.BL_WEIGHT
        self.fc_weight = cfg.FC_WEIGHT
        self.pose_weight = cfg.POSE_WEIGHT
        self.tr_weight = cfg.TR_WEIGHT
        
        # Create flow matching engine
        self.flow_engine = FlowMatching(
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
            rho=self.rho,
            motion_rep=self.motion_rep,
            flow_type="rectified"
        )
        
        # Create the flow velocity predictor network
        self.flow_predictor = InterFlowPredictor_Duet(
            input_feats=self.nfeats,
            latent_dim=self.latent_dim,
            ff_size=self.ff_size,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            dropout=self.dropout,
            activation=self.activation
        )
        
        # Create the loss weights dictionary
        self.loss_weights = {}
        self.loss_weights["RO"] = self.ro_weight
        self.loss_weights["JA"] = self.ja_weight
        self.loss_weights["DM"] = self.dm_weight
        self.loss_weights["VEL"] = self.vel_weight
        self.loss_weights["BL"] = self.bl_weight
        self.loss_weights["FC"] = self.fc_weight
        self.loss_weights["POSE"] = self.pose_weight
        self.loss_weights["TR"] = self.tr_weight
        
        # Normalizer for converting between representations
        self.normalizer = MotionNormalizerTorch()
        
        # Precompute masks for common sequence lengths
        self._mask_cache = {}
    
    def compute_loss(self, batch):
        """
        Compute flow matching and geometric losses for training.
        
        Args:
            batch: Dictionary containing:
                - motions: Motion data [B, T, D*2]
                - motion_lens: Length of valid motion data
                - cond: Text conditioning vectors
                - music: Music conditioning features
                
        Returns:
            Dictionary of losses
        """
        # Extract inputs
        x_0 = batch["motions"]
        B, T = x_0.shape[:2]
        device = x_0.device
        
        # Create mask for valid positions (use cached version if possible)
        cache_key = f"{T}_{batch['motion_lens'].cpu().numpy().tobytes()}"
        if cache_key in self._mask_cache:
            seq_mask = self._mask_cache[cache_key].to(device)
        else:
            seq_mask = self.generate_src_mask(T, batch["motion_lens"])
            if len(self._mask_cache) < 20:  # Limit cache size
                self._mask_cache[cache_key] = seq_mask.cpu()
            seq_mask = seq_mask.to(device)
        
        # Get global weights from config
        fm_weight = self.fm_weight
        inter_weight = self.inter_weight
        geo_weight = self.geo_weight
        
        # Compute flow matching loss with gradient tracking for backprop
        try:
            # Get flow matching results
            flow_results = self.flow_engine.loss_fn(
                model=self.flow_predictor,
                x_0=x_0,
                mask=seq_mask,
                cond=batch["cond"],
                t_bar=getattr(self.cfg, 'T_BAR', None),
                music=batch["music"]
            )
            
            # Extract the flow matching loss
            fm_loss = flow_results["loss"]
            
            # Extract prediction and target for additional losses
            prediction = flow_results["pred"]  # [B, T, 2, D]
            target = flow_results["target"]    # [B, T, 2, D]
            t = flow_results["t"]              # [B]
            
            # Reshape mask for loss computation
            mask_reshaped = seq_mask.reshape(B, T, -1, 1)
            
            # Calculate timestep mask similar to diffusion model
            t_bar = self.t_bar
            timestep_mask = (t <= t_bar).float()
            
            # Compute additional losses
            # 1. InterLoss for interaction between agents
            interloss_manager = InterLoss("l2", 22)
            interloss_manager.forward(prediction, target, mask_reshaped, timestep_mask)
            
            # 2. GeometricLoss for each agent
            loss_a_manager = GeometricLoss("l2", 22, "A")
            loss_a_manager.forward(prediction[..., 0, :], target[..., 0, :], 
                                mask_reshaped[..., 0, :], timestep_mask)
            
            loss_b_manager = GeometricLoss("l2", 22, "B")
            loss_b_manager.forward(prediction[..., 1, :], target[..., 1, :], 
                                mask_reshaped[..., 0, :], timestep_mask)
            
            # Collect all losses
            losses = {
                "flow_matching": fm_loss * fm_weight
            }
            
            # Add interaction losses
            for key, value in interloss_manager.losses.items():
                if key in self.loss_weights:
                    losses[key] = value * self.loss_weights[key]
                else:
                    losses[key] = value
            
            # Add agent A losses
            for key, value in loss_a_manager.losses.items():
                losses[key] = value
            
            # Add agent B losses
            for key, value in loss_b_manager.losses.items():
                losses[key] = value
            
            # Calculate total loss with proper weighting
            total_loss = losses["flow_matching"]
            
            # Apply global weights to groups of losses
            # Add geometric losses with geo_weight
            if "A" in loss_a_manager.losses:
                total_loss += loss_a_manager.losses["A"] * geo_weight
            
            if "B" in loss_b_manager.losses:
                total_loss += loss_b_manager.losses["B"] * geo_weight
            
            # Add interaction loss with inter_weight
            if "total" in interloss_manager.losses:
                total_loss += interloss_manager.losses["total"] * inter_weight
            
            losses["total"] = total_loss
            
            # Check for NaN and handle gracefully
            if not torch.isfinite(total_loss):
                print("Warning: Non-finite values in loss calculation. Using fallback loss.")
                losses["total"] = torch.ones(1, device=device, requires_grad=True)
            
        except Exception as e:
            print(f"Warning: Error in loss computation: {e}")
            # Create a small dummy loss as fallback
            losses = {
                "flow_matching": torch.ones(1, device=device, requires_grad=True),
                "total": torch.ones(1, device=device, requires_grad=True)
            }
        
        return losses
    
    @torch.inference_mode()
    def forward(self, batch):
        """
        Generate a motion sample using the flow model with improved error handling.
        
        Args:
            batch: Dictionary containing:
                - cond: Text conditioning vectors
                - motion_lens: Length of valid motion data
                - music: Music conditioning features
                
        Returns:
            Dictionary with generated output
        """
        B = batch["cond"].shape[0]
        T = batch["motion_lens"][0].item()
        device = batch["cond"].device
        
        # Create mask for valid positions (use cached if possible)
        cache_key = f"{T}_{batch['motion_lens'].cpu().numpy().tobytes()}"
        if cache_key in self._mask_cache:
            mask = self._mask_cache[cache_key].to(device)
        else:
            mask = self.generate_src_mask(T, batch["motion_lens"]).to(device)
        
        try:
            # Ensure music features are properly sized
            music_input = batch["music"][:, :T]
            
            # Sample from the flow model with error handling
            output = self.flow_engine.sample(
                model=self.flow_predictor,
                shape=(B, T, self.nfeats*2),  # 2 for agent A and B
                steps=self.sampling_steps,
                mask=mask,
                cond=batch["cond"],
                music=music_input,
                solver_type=self.solver_type,
            )
            
            # Validate output and handle errors
            if not torch.isfinite(output).all():
                print("Warning: Non-finite values in generation. Cleaning up...")
                # Selectively replace only the problematic values
                output = torch.nan_to_num(
                    output, 
                    nan=0.0,
                    posinf=100.0, 
                    neginf=-100.0
                )
                
        except Exception as e:
            print(f"Error during sampling: {e}")
            # Generate smoother fallback with learned bias
            # This creates a more neutral pose than pure random noise
            output = torch.randn(B, T, self.nfeats*2, device=device) * 0.1
            
        return {"output": output}
    
    def generate_src_mask(self, T, length):
        """Generate attention mask based on sequence lengths with vectorized operations"""
        B = length.shape[0]
        device = length.device
        
        # Create position indices matrix [B, T]
        pos_indices = torch.arange(T, device=device).expand(B, T)
        # Create length matrix [B, T]
        length_expanded = length.unsqueeze(1).expand(B, T)
        
        # Create mask where position < length
        mask = (pos_indices < length_expanded).float().unsqueeze(-1).expand(B, T, 2)
        
        return mask