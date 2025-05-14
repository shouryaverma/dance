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
from datasets.evaluator_models import InterCLIP
import argparse

os.environ['PL_TORCH_DISTRIBUTED_BACKEND'] = 'nccl'
from lightning.pytorch.strategies import DDPStrategy
torch.set_float32_matmul_precision('medium')

class LitTrainModel(pl.LightningModule):
    def __init__(self, model, cfg):
        super().__init__()
        # cfg init
        self.cfg = cfg
        self.mode = cfg.TRAIN.MODE

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
        self.logs = OrderedDict()
        self.it = self.cfg.TRAIN.LAST_ITER if self.cfg.TRAIN.LAST_ITER else 0
        self.epoch = self.cfg.TRAIN.LAST_EPOCH if self.cfg.TRAIN.LAST_EPOCH else 0
        self.start_time = time.time()
        self.val_it = self.cfg.TRAIN.LAST_ITER if self.cfg.TRAIN.LAST_ITER else 0
        self.val_logs = OrderedDict()
        self.val_local_it = 0

    def _configure_optim(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=float(self.cfg.TRAIN.LR), weight_decay=self.cfg.TRAIN.WEIGHT_DECAY)
        scheduler = CosineWarmupScheduler(optimizer=optimizer, warmup=10, max_iters=self.cfg.TRAIN.EPOCH, verbose=True)
        return [optimizer], [scheduler]

    def configure_optimizers(self):
        return self._configure_optim()
    
    def forward(self, batch_data):
        motions = torch.cat([batch_data['motion1'], batch_data['motion2']], dim=-1)
        text, motion_lens = batch_data['text'], batch_data['length']
        batch = OrderedDict({
            "text": text,
            "motions": motions.float(),
            "motion_lens": motion_lens.long()
        })
        loss, loss_logs = self.model(batch)# loss logs
        return loss, loss_logs

    def on_train_start(self):
        self.rank = 0
        self.world_size = 1
        
    def training_step(self, batch, batch_idx):
        loss, loss_logs = self.forward(batch)
        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        opt.step()
            
        return {"loss": loss, "loss_logs": loss_logs}

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            loss, loss_logs = self.forward(batch)
            return {"loss": loss, "loss_logs": loss_logs}
        
    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        if outputs.get('skip_batch') or not outputs.get('loss_logs'):
            return
        for k, v in outputs['loss_logs'].items():
            if k not in self.val_logs:
                if isinstance(v, float):
                    self.val_logs['val_'+k] = [v]
                else:
                    self.val_logs['val_'+k] = [v.item()]
            else:
                if isinstance(v, float):
                    self.val_logs['val_'+k].append(v)
                else:
                    self.val_logs['val_'+k].append(v.item())

        self.val_it += 1

    def on_validation_epoch_end(self):
        # pass
        mean_loss = OrderedDict({})
        for tag, value in self.val_logs.items():
            mean_loss[tag] = np.mean(value)
            self.writer.add_scalar(tag, mean_loss[tag], self.it)
        self.val_logs = OrderedDict()
        print_current_loss(self.start_time, self.val_it, mean_loss,
                            self.trainer.current_epoch,
                            inner_iter=0,
                            lr=self.trainer.optimizers[0].param_groups[0]['lr'])
            
    def on_train_batch_end(self, outputs, batch, batch_idx):
        if outputs.get('skip_batch') or not outputs.get('loss_logs'):
            return
        for k, v in outputs['loss_logs'].items():
            if k not in self.logs:
                if isinstance(v, float):
                    self.logs[k] = v
                else:
                    self.logs[k] = v.item()
            else:
                if isinstance(v, float):
                    self.logs[k] += v
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
    if cfg.NAME == "InterCLIP":
        model = InterCLIP(cfg)
    else:
        raise NotImplementedError
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process a string input.")
    parser.add_argument("--model_cfg", type=str, default="configs/eval_duet_debug.yaml", help="")
    parser.add_argument("--train_cfg", type=str, default="configs/train_interclip_full.yaml", help="")
    args = parser.parse_args()
    print(args)
    
    model_cfg = get_config(args.model_cfg)
    train_cfg = get_config(args.train_cfg)
    # model_cfg = get_config("configs/eval_model.yaml")
    # train_cfg = get_config("configs/train_interclip.yaml")
    data_cfg = get_config("configs/datasets_duet_juke_prerit.yaml").train_set
    val_data_cfg = get_config("configs/datasets_duet_juke_prerit.yaml").test_set

    datamodule = DataModule(data_cfg, val_data_cfg, None, train_cfg.TRAIN.BATCH_SIZE, train_cfg.TRAIN.NUM_WORKERS)
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

    litmodel = LitTrainModel(model, train_cfg)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=litmodel.model_dir,
                                                       every_n_epochs=train_cfg.TRAIN.SAVE_EPOCH*3,
                                                       save_top_k = -1)
    trainer = pl.Trainer(
        default_root_dir=litmodel.model_dir,
        devices=1, accelerator='gpu',
        max_epochs=train_cfg.TRAIN.EPOCH,
        strategy=None,
        precision=32,
        callbacks=[checkpoint_callback],
        check_val_every_n_epoch = train_cfg.TRAIN.SAVE_EPOCH,
        num_sanity_val_steps=1 # 1
    )
    ckpt_model = litmodel.model_dir + "/epoch=449-step=12600.ckpt"
    if os.path.exists(ckpt_model):
        print('resume from checkpoint')
        trainer.fit(model=litmodel, datamodule=datamodule, ckpt_path=ckpt_model)
    else:
        trainer.fit(model=litmodel, datamodule=datamodule)