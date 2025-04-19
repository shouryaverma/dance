import torch
import torch.nn as nn
from models.utils import *
from models.blocks import *
from models.flow_blocks import *
from models.flow_matching import RectifiedFlow, FlowType

class FlowNet_React(nn.Module):
    """
    Flow network for reactive follower motion generation.
    Predicts the velocity fields for rectified flow.
    """
    def __init__(
        self,
        input_feats,
        latent_dim=512,
        num_frames=240,
        ff_size=1024,
        num_layers=8,
        num_heads=8,
        dropout=0.1,
        attention_type="flash",  # Default to flash attention
        **kwargs
    ):
        super().__init__()
        
        self.num_frames = num_frames
        self.latent_dim = latent_dim
        self.input_feats = input_feats
        self.time_embed_dim = latent_dim
        
        # Define embedding dimensions
        self.music_emb_dim = 54
        self.text_emb_dim = 768
        
        # Time and position embeddings
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, dropout=0)
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)
        
        # Input embeddings
        self.motion_embed = nn.Linear(self.input_feats, self.latent_dim)
        self.text_embed = nn.Linear(self.text_emb_dim, self.latent_dim)
        self.music_embed = nn.Linear(self.music_emb_dim, self.latent_dim)
        
        # Music transformer
        musicTransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=8,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.musicTransEncoder = nn.TransformerEncoder(
            musicTransEncoderLayer,
            num_layers=4
        )
        
        # Interactive modeling blocks
        self.blocks = nn.ModuleList()
        
        # Choose block type based on attention_type
        if attention_type == "vanilla":
            for i in range(num_layers):
                self.blocks.append(
                    VanillaReactBlock(
                        latent_dim=latent_dim,
                        num_heads=num_heads,
                        dropout=dropout,
                        ff_size=ff_size
                    )
                )
        elif attention_type == "flash":
            for i in range(num_layers):
                self.blocks.append(
                    FlashReactBlock(
                        latent_dim=latent_dim,
                        num_heads=num_heads,
                        dropout=dropout,
                        ff_size=ff_size
                    )
                )
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")
          
        # Output layer - zero initialization for better stability
        self.out = zero_module(FinalLayer(self.latent_dim, self.input_feats))
    
    def forward(self, x, timesteps, mask=None, cond=None, music=None):
        """
        Forward pass to predict velocity field for reactive dancing.
        Only the follower's motion will be updated.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, T, 2*D]
            timesteps (torch.Tensor): Timesteps tensor of shape [B]
            mask (torch.Tensor, optional): Mask tensor
            cond (torch.Tensor, optional): Text conditioning tensor
            music (torch.Tensor, optional): Music conditioning tensor
        """
        B, T = x.shape[0], x.shape[1]
        
        # Split input into dancer A lead and dancer B follower
        x_a, x_b = x[..., :self.input_feats], x[..., self.input_feats:]
        
        if mask is not None:
            mask = mask[..., 0]
        
        # Embed timesteps and conditioning
        emb = self.embed_timestep(timesteps) + self.text_embed(cond)
        
        # Embed motions for both dancers
        a_emb = self.motion_embed(x_a)
        b_emb = self.motion_embed(x_b)
        h_a_prev = self.sequence_pos_encoder(a_emb)
        h_b_prev = self.sequence_pos_encoder(b_emb)
        
        # Process music embedding
        music_emb = self.music_embed(music)
        music_emb = self.sequence_pos_encoder(music_emb)
        music_emb = self.musicTransEncoder(music_emb)
        
        # Create mask if not provided
        if mask is None:
            mask = torch.ones(B, T).to(x_a.device)
        key_padding_mask = ~(mask > 0.5)
        
        # Verify music and motion have compatible shapes
        assert music_emb.shape[1] == T, (music_emb.shape, x_a.shape)
        
        # Process through custom blocks
        for i, block in enumerate(self.blocks):
            h_a_all, h_b_all, music_emb_all = block(h_a_prev, h_b_prev, music_emb, emb, key_padding_mask)

            # update previous hidden states
            h_a_prev = h_a_all
            h_b_prev = h_b_all
            music_emb = music_emb_all
        
        # Generate output velocities for both dancers
        output_a = self.out(h_a_prev)
        output_b = self.out(h_b_prev)
        
        # Combine outputs
        output = torch.cat([output_a, output_b], dim=-1)
        
        return output

class FlowMatching_React(nn.Module):
    """
    Rectified Flow Matching model for duet motion generation.
    This is the main class that integrates the flow model with the denoising network.
    """
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
        
        # Create the velocity field prediction network
        self.net = FlowNet_React(
            self.nfeats,
            self.latent_dim,
            ff_size=self.ff_size,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            dropout=self.dropout,
            activation=self.activation
        )
        
        # Create the rectified flow model
        self.flow = RectifiedFlow(
            num_timesteps=cfg.DIFFUSION_STEPS,
            flow_type=FlowType.RECTIFIED,
            rescale_timesteps=False,
            motion_rep=self.motion_rep
        )
    
    def compute_loss(self, batch):
        """
        Compute training loss
        """
        x_start = batch["motions"]
        
        B = x_start.shape[0]
        cond = batch.get("cond", None)
        music = batch.get("music", None)
        
        # Generate mask based on motion lengths
        mask = self.generate_src_mask(x_start.shape[1], batch["motion_lens"]).to(x_start.device)
        
        # Sample random timesteps for training
        t = torch.randint(0, self.flow.num_timesteps, (B,), device=x_start.device)
        
        # Generate timestep mask for conditional loss weighting
        timestep_mask = (t <= self.cfg.T_BAR).float()
        
        # Compute loss
        losses = self.flow.compute_loss(
            model=self.net,
            x_start=x_start,
            t=t,
            mask=mask,
            timestep_mask=timestep_mask,
            t_bar=self.cfg.T_BAR,
            model_kwargs={
                "mask": mask,
                "cond": cond,
                "music": music
            }
        )
        
        return losses
    
    def generate_src_mask(self, T, length):
        """
        Generate source mask for transformer based on sequence lengths.
        """
        B = length.shape[0]
        src_mask = torch.ones(B, T, 2)
        for p in range(2):
            for i in range(B):
                for j in range(length[i], T):
                    src_mask[i, j, p] = 0
        return src_mask
    
    def forward(self, batch):
        """
        Generate motion sequence using flow matching.
        This is used during inference.
        """
        cond = batch["cond"]
        B = cond.shape[0]
        T = batch["motion_lens"][0]
        
        # For reactive dancing, we need to use the lead dancer's motion
        if "lead_motion" in batch:
            # Extract lead dancer's motion and ensure float32 dtype
            lead_motion = batch["lead_motion"].to(torch.float32)
            
            # Ensure lead_motion has the right sequence length
            if lead_motion.shape[1] != T:
                lead_motion = lead_motion[:, :T]
            
            # Get feature dimension explicitly
            D = lead_motion.shape[2]
            
            # Reshape and normalize if needed
            if self.motion_rep == "global":
                # Instead of using -1, explicitly provide the feature dimension
                lead_motion_reshaped = lead_motion.reshape(B, T, 1, D)
                lead_motion_norm = self.flow.normalizer.forward(lead_motion_reshaped)
                lead_motion_flat = lead_motion_norm.reshape(B, T, -1)
            else:
                lead_motion_flat = lead_motion
            
            # Generate follower motion with half the feature dimension
            follower_shape = (B, T, self.nfeats)
            
            # Get initial noise for follower (with explicit float32 dtype)
            device = cond.device
            follower_noise = self.flow.sample_noise(follower_shape, device).to(torch.float32)
            
            # Construct initial combined state with real lead motion and noisy follower
            initial_x = torch.cat([lead_motion_flat, follower_noise], dim=-1)
            
            # Sample from the flow model
            output = self.flow.sample(
                model=self.net,
                shape=(B, T, self.nfeats*2),
                noise=initial_x,  # Use our custom initial state
                model_kwargs={
                    "mask": None,
                    "cond": cond,
                    "music": batch["music"][:, :T]
                },
                progress=True
            )
            
            # Extract only the follower part from the result and recombine with original lead
            if self.motion_rep == "global":
                # Re-normalize lead motion to match output format
                output_reshaped = output.reshape(B, T, 2, -1)
                # Replace lead dancer part with the original lead
                output_reshaped[:, :, 0, :] = lead_motion_norm.squeeze(2)
                # Reshape back to flat format
                output = output_reshaped.reshape(B, T, -1)
        else:
            # Original duet generation case
            output = self.flow.sample(
                model=self.net,
                shape=(B, T, self.nfeats*2),
                model_kwargs={
                    "mask": None,
                    "cond": cond,
                    "music": batch["music"][:, :T]
                },
                progress=True
            )
        
        return {"output": output}