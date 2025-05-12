import sys
sys.path.append(sys.path[0] + r"/../")
import torch
import lightning.pytorch as pl
import torch.optim as optim
from collections import OrderedDict
from datasets import DataModule
from configs import get_config
from os.path import join as pjoin
from torch.utils.tensorboard import SummaryWriter
from models import *
from pathlib import Path
from utils import paramUtil
import torch

os.environ['PL_TORCH_DISTRIBUTED_BACKEND'] = 'nccl'
from lightning.pytorch.strategies import DDPStrategy
torch.set_float32_matmul_precision('medium')
from utils.plot_script import *

class LitTrainModel(pl.LightningModule):
    def __init__(self, model, train_cfg, model_cfg):
        super().__init__()
        # cfg init
        self.cfg = train_cfg
        self.model_cfg = model_cfg
        self.automatic_optimization = False
        self.save_root = pjoin(self.cfg.GENERAL.CHECKPOINT, self.cfg.GENERAL.EXP_NAME)
        self.model_dir = pjoin(self.save_root, 'model')
        self.meta_dir = pjoin(self.save_root, 'meta')
        self.log_dir = pjoin(self.save_root, 'log')
        self.vis_dir = pjoin(self.save_root, 'vis')
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.meta_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.vis_dir, exist_ok=True)

        self.model = model
        self.normalizerTorch = MotionNormalizerTorch()
        self.writer = SummaryWriter(self.log_dir)

    def _configure_optim(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=float(self.cfg.TRAIN.LR), weight_decay=self.cfg.TRAIN.WEIGHT_DECAY)
        scheduler = CosineWarmupScheduler(optimizer=optimizer, warmup=10, max_iters=self.cfg.TRAIN.EPOCH, verbose=True)
        return [optimizer], [scheduler]

    def configure_optimizers(self):
        return self._configure_optim()

    # def plot_motion_intergen(self, motion1, motion2, length, result_root, caption, mode = 'train', motion = 'recon', idx = 0):
    #     # only plot in the main process
    #     if self.device.index != 0:
    #         return
    #     assert motion in ['gen', 'gt', 'recon'], motion
    #     assert mode in ['train', 'val'], mode
    #     motion1 = motion1.cpu().numpy()[:length]
    #     motion2 = motion2.cpu().numpy()[:length]
    #     mp_data = [motion1, motion2]
    #     mp_joint = []
    #     for i, data in enumerate(mp_data):
    #         if i == 0:
    #             joint = data[:,:22*3].reshape(-1,22,3)
    #         else:
    #             joint = data[:,:22*3].reshape(-1,22,3)
    #         mp_joint.append(joint)

    #     result_path = Path(result_root) / f"{mode}_{self.current_epoch}_{idx}_{motion}.mp4"
    #     plot_3d_motion(str(result_path), paramUtil.t2m_kinematic_chain, mp_joint, title=caption, fps=30)

    def plot_motion_intergen(self, gt_motion1, gt_motion2, gen_motion1, gen_motion2, length, result_root, caption, mode='train', idx=0):
        # only plot in the main process
        if self.device.index != 0:
            return
        
        gt_motion1 = gt_motion1.cpu().numpy()[:length]
        gt_motion2 = gt_motion2.cpu().numpy()[:length]
        gen_motion1 = gen_motion1.cpu().numpy()[:length]
        gen_motion2 = gen_motion2.cpu().numpy()[:length]
        
        mp_data = [gt_motion1, gt_motion2, gen_motion1, gen_motion2]
        mp_joint = []
        
        for data in mp_data:
            joint = data[:,:22*3].reshape(-1,22,3)
            mp_joint.append(joint)

        result_path = Path(result_root) / f"{mode}_{self.current_epoch}_{idx}_combined.mp4"
        plot_3d_motion(str(result_path), paramUtil.t2m_kinematic_chain, mp_joint, title=caption, fps=30)
    
    def text_length_to_motion_torch(self, text, music, length):
        # text: 1,*
        # length: 1,
        input_batch = {}
        input_batch["text"] = text
        input_batch["music"] = music
        input_batch["motion_lens"] = length
        # For ReactModel, we need the lead dancer's motion from the test batch
        if self.model_cfg.NAME == "ReactModel":
            # During inference, we'll use ground truth leader motion from the test batch
            # This assumes the lead motion is passed in the batch from sample_text
            if hasattr(self, '_temp_lead_motion'):
                input_batch["lead_motion"] = self._temp_lead_motion
                input_batch["follower_motion"] = self._temp_follower_motion 
                delattr(self, '_temp_lead_motion')  # Clean up after use
                delattr(self, '_temp_follower_motion')  # Clean up after use
        output_batch = self.model.forward_test(input_batch)
        motions_output = output_batch["output"].reshape(output_batch["output"].shape[0], output_batch["output"].shape[1], 2, -1)
        motions_output = self.normalizerTorch.backward(motions_output.detach())
        return motions_output[:,:,0], motions_output[:,:,1]
    
    # def sample_text(self, batch_data, batch_idx, mode):
    #     motion1, motion2, music, text, motion_lens = batch_data['motion1'], \
    #         batch_data['motion2'], batch_data['music'], batch_data['text'], batch_data['length']
        
    #     if self.model_cfg.NAME == "DuetModel":
    #         # Duet dancing: generate both dancer motions simultaneously
    #         motion_gen_1, motion_gen_2 = self.text_length_to_motion_torch(text[0:1], music[0:1], motion_lens[0:1])
            
    #         # Visualize ground truth
    #         self.plot_motion_intergen(motion1[0], motion2[0], 
    #                         motion_lens[0], self.vis_dir, text[0],
    #                         mode=mode, motion='gt', idx=batch_idx)
            
    #         # Visualize generated motions
    #         self.plot_motion_intergen(motion_gen_1[0], motion_gen_2[0], 
    #                         motion_lens[0], self.vis_dir, text[0],
    #                         mode=mode, motion='gen', idx=batch_idx)
        
    #     elif self.model_cfg.NAME == "ReactModel":
    #         # Reactive dancing: use ground truth leader (motion1) and generate follower (motion2)
            
    #         # Store lead motion temporarily for the text_length_to_motion_torch function
    #         self._temp_lead_motion = motion1[0:1]
            
    #         # Generate only the follower's motion
    #         _, motion_gen_follower = self.text_length_to_motion_torch(text[0:1], music[0:1], motion_lens[0:1])
            
    #         # Visualize ground truth
    #         self.plot_motion_intergen(motion1[0], motion2[0], 
    #                         motion_lens[0], self.vis_dir, text[0],
    #                         mode=mode, motion='gt', idx=batch_idx)
            
    #         # Visualize leader (ground truth) with generated follower
    #         self.plot_motion_intergen(motion1[0], motion_gen_follower[0], 
    #                         motion_lens[0], self.vis_dir, text[0],
    #                         mode=mode, motion='gen', idx=batch_idx)

    def sample_text(self, batch_data, batch_idx, mode):
        motion1, motion2, music, text, motion_lens = batch_data['motion1'], \
            batch_data['motion2'], batch_data['music'], batch_data['text'], batch_data['length']
        
        if self.model_cfg.NAME == "DuetModel":
            # Generate both dancer motions
            motion_gen_1, motion_gen_2 = self.text_length_to_motion_torch(text[0:1], music[0:1], motion_lens[0:1])
            
            # Plot both ground truth and generated in one visualization
            self.plot_motion_intergen(
                motion1[0], motion2[0],          # Ground truth motions
                motion_gen_1[0], motion_gen_2[0], # Generated motions
                motion_lens[0], self.vis_dir, text[0],
                mode=mode, idx=batch_idx
            )
        
        elif self.model_cfg.NAME == "ReactModel":
            # Store lead motion for generation
            self._temp_lead_motion = motion1[0:1]
            self._temp_follower_motion = motion2[0:1]
            
            # Generate follower motion
            motion_gen_lead, motion_gen_follower = self.text_length_to_motion_torch(text[0:1], music[0:1], motion_lens[0:1])
            
            # Plot both ground truth and generated in one visualization
            # For ReactModel, lead is the same for both GT and generated
            self.plot_motion_intergen(
                motion1[0], motion2[0],          # Ground truth lead and follower
                motion_gen_lead[0], motion_gen_follower[0], # GT lead and generated follower
                motion_lens[0], self.vis_dir, text[0],
                mode=mode, idx=batch_idx
            )
        
    def forward(self, batch_data):
        motion1, motion2, music, text, motion_lens = batch_data['motion1'], \
            batch_data['motion2'], batch_data['music'], batch_data['text'], batch_data['length']
        motion1 = motion1.detach().float()  # .to(self.device)
        motion2 = motion2.detach().float()  # .to(self.device)
        motions = torch.cat([motion1, motion2], dim=-1)

        B, T = motion1.shape[:2]

        batch = OrderedDict({})
        batch["text"] = text
        batch["music"] = music.float()
        batch["motions"] = motions.reshape(B, T, -1).type(torch.float32)
        batch["motion_lens"] = motion_lens.long()
        # For ReactModel, add leader's motion as input
        if self.model_cfg.NAME == "ReactModel":
            # For reactive dancing, motion1 is considered the leader
            batch["lead_motion"] = motion1
            batch["follower_motion"] = motion2

        loss, loss_logs = self.model(batch)
        return loss, loss_logs

    def on_train_start(self):
        self.rank = 0
        self.world_size = 1
        self.start_time = time.time()
        self.it = self.cfg.TRAIN.LAST_ITER if self.cfg.TRAIN.LAST_ITER else 0
        self.epoch = self.cfg.TRAIN.LAST_EPOCH if self.cfg.TRAIN.LAST_EPOCH else 0
        self.logs = OrderedDict()


    def training_step(self, batch, batch_idx):
        loss, loss_logs = self.forward(batch)
        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        opt.step()
        if batch_idx < 2 and self.trainer.current_epoch % self.cfg.TRAIN.SAVE_EPOCH==0:
            self.model.eval()
            self.sample_text(batch, batch_idx, 'train')
            self.model.train()

        return {"loss": loss,
            "loss_logs": loss_logs}

    def validation_step(self, batch, batch_idx):
        print('validation step')
        if batch_idx > 2:
            return
        self.sample_text(batch, batch_idx, 'val')

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if outputs.get('skip_batch') or not outputs.get('loss_logs'):
            return
        for k, v in outputs['loss_logs'].items():
            if k not in self.logs:
                self.logs[k] = v.item()
            else:
                self.logs[k] += v.item()

        self.it += 1
        if self.it % self.cfg.TRAIN.LOG_STEPS == 0 and self.device.index == 0:
            mean_loss = OrderedDict({})
            for tag, value in self.logs.items():
                mean_loss[tag] = value / self.cfg.TRAIN.LOG_STEPS
                self.writer.add_scalar(tag, mean_loss[tag], self.it)
            self.logs = OrderedDict()
            print_current_loss(self.start_time, self.it, mean_loss,
                               self.trainer.current_epoch,
                               inner_iter=batch_idx,
                               lr=self.trainer.optimizers[0].param_groups[0]['lr'])

    def on_train_epoch_end(self):
        # pass
        sch = self.lr_schedulers()
        if sch is not None:
            sch.step()

    def save(self, file_name):
        state = {}
        try:
            state['model'] = self.model.module.state_dict()
        except:
            state['model'] = self.model.state_dict()
        torch.save(state, file_name, _use_new_zipfile_serialization=False)
        return

def build_models(cfg):
    if cfg.NAME == "DuetModel":
        model = DuetModel(cfg)
    elif cfg.NAME == "ReactModel":
        model = ReactModel(cfg)
    else:
        raise NotImplementedError
    return model

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process a string input.")
    parser.add_argument("--model_cfg", type=str, default="configs/model_interhuman.yaml", help="")
    parser.add_argument("--train_cfg", type=str, default="configs/train_interhuman.yaml", help="")
    parser.add_argument("--data_cfg", type=str, default="configs/datasets_interhuman_prerit.yaml", help="")
    args = parser.parse_args()
    print(args)
    
    model_cfg = get_config(args.model_cfg)
    train_cfg = get_config(args.train_cfg)
    data_cfg = get_config(args.data_cfg).train_set
    val_data_cfg = get_config(args.data_cfg).val_set
    test_data_cfg = get_config(args.data_cfg).test_set

    datamodule = DataModule(data_cfg, val_data_cfg, test_data_cfg, train_cfg.TRAIN.BATCH_SIZE, train_cfg.TRAIN.NUM_WORKERS)
    model = build_models(model_cfg)

    if train_cfg.TRAIN.RESUME:
        ckpt = torch.load(train_cfg.TRAIN.RESUME, map_location="cpu")
        for k in list(ckpt["state_dict"].keys()):
            if "model" in k:
                ckpt["state_dict"][k.replace("model.", "")] = ckpt["state_dict"].pop(k)
        model.load_state_dict(ckpt["state_dict"], strict=True)
        print("checkpoint state loaded!")

    # Count total parameters
    total_params = sum(p.numel() for p in model.parameters())

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")

    litmodel = LitTrainModel(model, train_cfg, model_cfg)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=litmodel.model_dir,
        filename="epoch_{epoch:04d}",  # Add proper filename pattern
        every_n_epochs=train_cfg.TRAIN.SAVE_EPOCH,
        save_top_k=-1,  # Keep all checkpoints
        save_last=True  # Save a 'last.ckpt' file
    )
    
    class CustomCheckpointCallback(pl.callbacks.Callback):
        def on_train_epoch_end(self, trainer, pl_module):
            # Only save on the specified interval
            if (trainer.current_epoch + 1) % trainer.check_val_every_n_epoch == 0:
                # Use your custom save method
                checkpoint_path = pjoin(pl_module.model_dir, f"model={model_cfg.NAME}-epoch={trainer.current_epoch}-step={pl_module.it}.ckpt")
                pl_module.save(checkpoint_path)
                print(f"Custom checkpoint saved at {checkpoint_path}")
    
    trainer = pl.Trainer(
        default_root_dir=litmodel.model_dir,
        # devices="auto",
        devices=1,
        accelerator='gpu',
        max_epochs=train_cfg.TRAIN.EPOCH,
        # strategy=DDPStrategy(find_unused_parameters=True),
        strategy=None,
        precision=32,
        callbacks=[checkpoint_callback, CustomCheckpointCallback()],
        check_val_every_n_epoch = train_cfg.TRAIN.SAVE_EPOCH,
        num_sanity_val_steps=1 # 1
    )
    ckpt_model = litmodel.model_dir + "/last.ckpt"
    if os.path.exists(ckpt_model):
        print('resume from checkpoint')
        trainer.fit(model=litmodel, datamodule=datamodule, ckpt_path=ckpt_model)
    else:
        trainer.fit(model=litmodel, datamodule=datamodule)