import torch
import torch.nn as nn
from models.utils import *
from models.blocks import *
from models.flow_blocks import *
from models.flow_matching import RectifiedFlow, FlowType

class FlowNet_Duet(nn.Module):
    """
    Flow network for interactive duet motion generation.
    Predicts the velocity fields for rectified flow.
    """
    def __init__(
        self,
        input_feats,
        latent_dim=512,
        num_frames=240,
        ff_size=1024,
        music_dim=4800,
        num_layers=8,
        num_heads=8,
        dropout=0.1,
        attention_type="flash",
        use_text=True,
        use_music=True,
        cfg_dropout=0.1,
        **kwargs
    ):
        super().__init__()

        self.num_frames = num_frames
        self.latent_dim = latent_dim
        self.input_feats = input_feats
        self.time_embed_dim = latent_dim
        self.use_text = use_text
        self.use_music = use_music
        self.cfg_dropout = cfg_dropout

        self.music_emb_dim = music_dim
        self.text_emb_dim = 768

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, dropout=0)
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        self.motion_embed = nn.Linear(self.input_feats, self.latent_dim)
        self.text_embed = nn.Linear(self.text_emb_dim, self.latent_dim)
        self.music_embed = nn.Linear(self.music_emb_dim, self.latent_dim)
        self.mode_embed = nn.Embedding(2, self.latent_dim)  # 0=duet, 1=react

        self.leader_look_ahead = LookAheadTransformer(
            latent_dim, num_heads, dropout, look_ahead_window=50)

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

        self.blocks = nn.ModuleList()
        if attention_type == "vanilla":
            for i in range(num_layers):
                self.blocks.append(
                    VanillaDuetBlock(
                        latent_dim=latent_dim,
                        num_heads=num_heads,
                        dropout=dropout,
                        ff_size=ff_size
                    )
                )
        elif attention_type == "flash":
            for i in range(num_layers):
                self.blocks.append(
                    FlashDuetBlock(
                        latent_dim=latent_dim,
                        num_heads=num_heads,
                        dropout=dropout,
                        ff_size=ff_size
                    )
                )
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")

        self.out_a = zero_module(FinalLayer(self.latent_dim, self.input_feats))
        self.out_b = zero_module(FinalLayer(self.latent_dim, self.input_feats))
    
    def forward(self, x, timesteps, mask=None, cond=None, music=None, mode="duet"):
        B, T = x.shape[0], x.shape[1]

        x_a, x_b = x[..., :self.input_feats], x[..., self.input_feats:]

        if mask is not None:
            mask = mask[..., 0]

        # CFG dropout: randomly null conditioning during training
        if self.training and self.cfg_dropout > 0:
            if torch.rand(1).item() < self.cfg_dropout:
                cond = torch.zeros_like(cond)
            if torch.rand(1).item() < self.cfg_dropout:
                music = torch.zeros_like(music)

        if not self.use_text:
            cond = torch.zeros_like(cond)
        if not self.use_music:
            music = torch.zeros_like(music)

        mode_idx = torch.zeros(B, dtype=torch.long, device=x.device) if mode == "duet" \
            else torch.ones(B, dtype=torch.long, device=x.device)
        mode_emb = self.mode_embed(mode_idx)  # (B, latent_dim)

        emb = self.embed_timestep(timesteps) + self.text_embed(cond) + mode_emb

        a_emb = self.motion_embed(x_a)
        if mode == "react":
            a_emb = self.leader_look_ahead(a_emb)
        b_emb = self.motion_embed(x_b)

        h_a_prev = self.sequence_pos_encoder(a_emb)
        h_b_prev = self.sequence_pos_encoder(b_emb)

        music_emb = self.music_embed(music)
        music_emb = self.sequence_pos_encoder(music_emb)
        music_emb = self.musicTransEncoder(music_emb)

        if mask is None:
            mask = torch.ones(B, T).to(x_a.device)
        key_padding_mask = ~(mask > 0.5)

        assert music_emb.shape[1] == T, (music_emb.shape, x_a.shape)

        for block in self.blocks:
            h_a_prev, h_b_prev, music_emb = block(h_a_prev, h_b_prev, music_emb, emb, key_padding_mask)

        output_a = self.out_a(h_a_prev)
        output_b = self.out_b(h_b_prev)

        return torch.cat([output_a, output_b], dim=-1)

class FlowMatching_Duet(nn.Module):
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
        self.use_text = cfg.USE_TEXT
        self.use_music = cfg.USE_MUSIC
        self.music_dim = cfg.MUSIC_DIM
        
        # Create the velocity field prediction network
        self.net = FlowNet_Duet(
            self.nfeats,
            self.latent_dim,
            ff_size=self.ff_size,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            dropout=self.dropout,
            activation=self.activation,
            music_dim=self.music_dim,
            use_text=self.use_text,
            use_music=self.use_music,
            cfg_dropout=getattr(cfg, 'CFG_DROPOUT', 0.1),
        )
        
        # Create the rectified flow model
        self.flow = RectifiedFlow(
            num_timesteps=cfg.DIFFUSION_STEPS,
            flow_type=FlowType.RECTIFIED,
            rescale_timesteps=False,
            motion_rep=self.motion_rep
        )
    
    def compute_loss(self, batch):
        x_start = batch["motions"]
        B = x_start.shape[0]
        cond = batch.get("cond", None)
        music = batch.get("music", None)
        task_mode = batch.get("task_mode", "duet")

        mask = self.generate_src_mask(x_start.shape[1], batch["motion_lens"]).to(x_start.device)

        t = torch.randint(0, self.flow.num_timesteps, (B,), device=x_start.device)
        timestep_mask = (t <= self.cfg.T_BAR).float()

        losses = self.flow.compute_loss(
            model=self.net,
            x_start=x_start,
            t=t,
            mask=mask,
            timestep_mask=timestep_mask,
            t_bar=self.cfg.T_BAR,
            mode=task_mode,
            model_kwargs={
                "mask": mask,
                "cond": cond,
                "music": music,
                "mode": task_mode,
            }
        )

        return losses
    
    def generate_src_mask(self, T, length):
        B = length.shape[0]
        indices = torch.arange(T, device=length.device).unsqueeze(0).expand(B, -1)
        mask_1d = (indices < length.unsqueeze(1)).float()
        return mask_1d.unsqueeze(-1).expand(B, T, 2)
    
    def forward(self, batch):
        cond = batch["cond"]
        B = cond.shape[0]
        T = batch["motion_lens"][0]
        task_mode = batch.get("task_mode", "duet")
        music = batch["music"][:, :T]

        lead_motion_normalized = None
        if task_mode == "react" and "lead_motion" in batch:
            lead_motion = batch["lead_motion"].to(torch.float32)
            if lead_motion.shape[1] != T:
                lead_motion = lead_motion[:, :T]
            D = lead_motion.shape[2]
            # Normalizer expects (B, T, 2, D); use a zero-filled second dancer
            # and take only dancer-0 output so per-dancer stats apply correctly
            dummy = torch.zeros_like(lead_motion)
            lead_pair = torch.stack([lead_motion, dummy], dim=2)  # (B, T, 2, D)
            lead_motion_normalized = self.flow.normalizer.forward(lead_pair)[:, :, 0, :]  # (B, T, D)

        output = self.flow.sample(
            model=self.net,
            shape=(B, T, self.nfeats * 2),
            lead_motion=lead_motion_normalized,
            model_kwargs={
                "mask": None,
                "cond": cond,
                "music": music,
                "mode": task_mode,
            },
            progress=True
        )

        return {"output": output}