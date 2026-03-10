import torch
import numpy as np
from tqdm.auto import tqdm
from enum import Enum
from utils.utils import MotionNormalizerTorch
from models.losses import InterLoss, GeometricLoss

class FlowType(Enum):
    """
    Types of flow implementations
    """
    RECTIFIED = "rectified"
    LINEAR = "linear"

class RectifiedFlow:
    """
    Implementation of Rectified Flow with discrete timesteps.
    """
    
    def __init__(
        self, 
        num_timesteps=1000, 
        flow_type=FlowType.RECTIFIED,
        rescale_timesteps=False,
        motion_rep="global",
        contrastive_weight=0.05
    ):
        self.num_timesteps = num_timesteps
        self.flow_type = flow_type
        self.rescale_timesteps = rescale_timesteps
        self.motion_rep = motion_rep
        self.normalizer = MotionNormalizerTorch()
        self.timesteps = torch.linspace(0, 1, num_timesteps)
        self.contrastive_weight = contrastive_weight

    def compute_contrastive_loss(self, pred_velocity, true_velocity):
        B = pred_velocity.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=pred_velocity.device)

        x = pred_velocity.reshape(B, -1)
        y = true_velocity.reshape(B, -1)

        pos_error = ((x - y) ** 2).mean(dim=-1)  # [B]

        neg_indices = (torch.arange(B, device=x.device) + 1) % B
        neg_error = ((x - y[neg_indices]) ** 2).mean(dim=-1)  # [B]

        return (pos_error - self.contrastive_weight * neg_error).mean()
        
    def interpolate(self, x0, x1, t):
        """
        Interpolate between x0 and x1 at time t.
        For rectified flow, this is a straight line path.
        """
        t = t.view(-1, 1, 1)  # Reshape for broadcasting
        return (1 - t) * x0 + t * x1
    
    def compute_vector_field(self, x0, x1, t):
        """
        Compute the vector field (velocity) for rectified flow.
        For rectified flow, the velocity field is a constant (x1 - x0)
        """
        return x1 - x0
    
    def sample_noise(self, shape, device):
        """
        Sample random noise with the same shape as the data.
        """
        return torch.randn(shape, device=device)
    
    def forward_process(self, x_start, t_normalized):
        """
        Forward process: interpolate from data to noise at time t.
        """
        # Generate noise
        noise = self.sample_noise(x_start.shape, x_start.device)
        
        # Interpolate
        x_t = self.interpolate(x_start, noise, t_normalized)
        
        return x_t, noise
    
    # def compute_loss(self, model, x_start, t, mask=None, timestep_mask=None, t_bar=None, mode="duet", model_kwargs=None):
    #     if model_kwargs is None:
    #         model_kwargs = {}

    #     B, T = x_start.shape[:2]
    #     t_normalized = t.float() / self.num_timesteps
    #     x_start_shaped = x_start.reshape(B, T, 2, -1)
    #     D = x_start_shaped.shape[-1]

    #     if mask is not None:
    #         mask = mask.reshape(B, T, -1, 1)

    #     if self.motion_rep == "global":
    #         x_start_normalized = self.normalizer.forward(x_start_shaped).reshape(B, T, -1)
    #     else:
    #         x_start_normalized = x_start

    #     x_start_a = x_start_normalized[..., :D]
    #     x_start_b = x_start_normalized[..., D:]

    #     if mode == "react":
    #         # Leader is fixed: no noise, zero target velocity
    #         noise_b = self.sample_noise((B, T, D), x_start.device)
    #         t_view = t_normalized.view(-1, 1, 1)
    #         x_t_b = (1 - t_view) * x_start_b + t_view * noise_b
    #         x_t = torch.cat([x_start_a, x_t_b], dim=-1)
    #         true_velocity_a = torch.zeros_like(x_start_a)
    #         true_velocity_b = noise_b - x_start_b
    #         true_velocity = torch.cat([true_velocity_a, true_velocity_b], dim=-1)
    #     else:
    #         x_t, noise = self.forward_process(x_start_normalized, t_normalized)
    #         true_velocity = self.compute_vector_field(x_start_normalized, noise, t_normalized)

    #     pred_velocity = model(x_t, self._scale_timesteps(t), **model_kwargs)

    #     losses = {}
    #     simple_loss = ((true_velocity - pred_velocity) ** 2).mean()
    #     losses["simple"] = simple_loss

    #     if mask is not None and self.motion_rep == "global":
    #         prediction = pred_velocity.reshape(B, T, 2, -1)
    #         target = true_velocity.reshape(B, T, 2, -1)

    #         interloss_manager = InterLoss("l2", 22)
    #         interloss_manager.forward(prediction, target, mask, timestep_mask)

    #         loss_a_manager = GeometricLoss("l2", 22, "A")
    #         loss_a_manager.forward(prediction[..., 0, :], target[..., 0, :], mask[..., 0, :], timestep_mask)

    #         loss_b_manager = GeometricLoss("l2", 22, "B")
    #         loss_b_manager.forward(prediction[..., 1, :], target[..., 1, :], mask[..., 0, :], timestep_mask)

    #         losses.update(loss_a_manager.losses)
    #         losses.update(loss_b_manager.losses)
    #         losses.update(interloss_manager.losses)

    #         if mode == "duet":
    #             losses["total"] = loss_a_manager.losses["A"] + loss_b_manager.losses["B"] + interloss_manager.losses["total"]
    #         else:
    #             # React: zero leader loss by construction, only follower + inter
    #             losses["total"] = loss_b_manager.losses["B"] + interloss_manager.losses["total"]
    #     else:
    #         losses["total"] = simple_loss

    #     return losses

    def compute_loss(self, model, x_start, t, mask=None, timestep_mask=None, t_bar=None, mode="duet", model_kwargs=None):
        if model_kwargs is None:
            model_kwargs = {}

        B, T = x_start.shape[:2]
        t_normalized = t.float() / self.num_timesteps
        x_start_shaped = x_start.reshape(B, T, 2, -1)
        D = x_start_shaped.shape[-1]

        if mask is not None:
            mask = mask.reshape(B, T, -1, 1)

        if self.motion_rep == "global":
            x_start_normalized = self.normalizer.forward(x_start_shaped).reshape(B, T, -1)
        else:
            x_start_normalized = x_start

        x_start_a = x_start_normalized[..., :D]
        x_start_b = x_start_normalized[..., D:]

        if mode == "react":
            noise_b = self.sample_noise((B, T, D), x_start.device)
            t_view = t_normalized.view(-1, 1, 1)
            x_t_b = (1 - t_view) * x_start_b + t_view * noise_b
            x_t = torch.cat([x_start_a, x_t_b], dim=-1)
            true_velocity_a = torch.zeros_like(x_start_a)
            true_velocity_b = noise_b - x_start_b
            true_velocity = torch.cat([true_velocity_a, true_velocity_b], dim=-1)
        else:
            x_t, noise = self.forward_process(x_start_normalized, t_normalized)
            true_velocity = self.compute_vector_field(x_start_normalized, noise, t_normalized)

        pred_velocity = model(x_t, self._scale_timesteps(t), **model_kwargs)

        losses = {}
        simple_loss = ((true_velocity - pred_velocity) ** 2).mean()
        losses["simple"] = simple_loss

        if mask is not None and self.motion_rep == "global":
            prediction = pred_velocity.reshape(B, T, 2, -1)
            target = true_velocity.reshape(B, T, 2, -1)

            interloss_manager = InterLoss("l2", 22)
            interloss_manager.forward(prediction, target, mask, timestep_mask)

            loss_a_manager = GeometricLoss("l2", 22, "A")
            loss_a_manager.forward(prediction[..., 0, :], target[..., 0, :], mask[..., 0, :], timestep_mask)

            loss_b_manager = GeometricLoss("l2", 22, "B")
            loss_b_manager.forward(prediction[..., 1, :], target[..., 1, :], mask[..., 0, :], timestep_mask)

            losses.update(loss_a_manager.losses)
            losses.update(loss_b_manager.losses)
            losses.update(interloss_manager.losses)

            if mode == "duet":
                motion_total = loss_a_manager.losses["A"] + loss_b_manager.losses["B"] + interloss_manager.losses["total"]
            else:
                motion_total = loss_b_manager.losses["B"] + interloss_manager.losses["total"]
        else:
            motion_total = simple_loss

        # # ΔFM contrastive term on predicted velocity fields
        # contrastive_loss = self.compute_contrastive_loss(pred_velocity, true_velocity)
        # losses["contrastive"] = contrastive_loss
        losses["total"] = motion_total

        return losses
    
    def _euler_step(self, x, velocity, dt):
        """Helper function for Euler integration step"""
        return x - velocity * dt
    
    def _heun_step(self, model, x, t, t_next, dt, model_kwargs):
        """
        Heun's method (improved Euler) for more accurate integration
        """
        velocity_t = model(x, self._scale_timesteps(t), **model_kwargs)
        x_euler = self._euler_step(x, velocity_t, dt)
        
        velocity_t_next = model(x_euler, self._scale_timesteps(t_next), **model_kwargs)
        velocity_avg = 0.5 * (velocity_t + velocity_t_next)
        
        return self._euler_step(x, velocity_avg, dt)
    
    def sample(self, model, shape, noise=None, lead_motion=None, model_kwargs=None, device=None, progress=False):
        if model_kwargs is None:
            model_kwargs = {}

        if device is None:
            device = next(model.parameters()).device

        D = shape[-1] // 2

        if noise is None:
            noise = self.sample_noise(shape, device)

        if lead_motion is not None:
            # Pin lead channels to real data; only follower is noisy
            follower_noise = self.sample_noise((shape[0], shape[1], D), device)
            noise = torch.cat([lead_motion, follower_noise], dim=-1)

        x = noise.to(torch.float32)
        original_timesteps = list(range(self.num_timesteps))[::-1]
        timesteps_iter = tqdm(original_timesteps) if progress else original_timesteps

        for i, t in enumerate(timesteps_iter):
            t_tensor = torch.tensor([t] * shape[0], device=device)

            if i == len(original_timesteps) - 1:
                with torch.no_grad():
                    velocity = model(x, self._scale_timesteps(t_tensor), **model_kwargs)
                dt = 1.0 / self.num_timesteps
                x = self._euler_step(x, velocity, dt)
            else:
                t_next = original_timesteps[i + 1]
                t_next_tensor = torch.tensor([t_next] * shape[0], device=device)
                dt = 1.0 / self.num_timesteps
                with torch.no_grad():
                    x = self._heun_step(model, x, t_tensor, t_next_tensor, dt, model_kwargs)

            # After each step, clamp lead channels back to fixed ground truth
            if lead_motion is not None:
                x = torch.cat([lead_motion, x[..., D:]], dim=-1)

        return x
    
    def _scale_timesteps(self, t):
        """
        Scale timesteps if needed.
        """
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t