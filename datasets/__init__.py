import lightning.pytorch as pl
import torch
# from .interhuman import InterHumanDataset
from datasets.evaluator import (
    EvaluatorModelWrapper,
    EvaluationDataset,
    get_dataset_motion_loader,
    get_motion_loader)
from .text2duet import Text2Duet
from .interhuman import InterHumanDataset
from .DD100 import DD100
# from .dataloader import build_dataloader

__all__ = [
    'InterHumanDataset', 'EvaluationDataset',
    'get_dataset_motion_loader', 'get_motion_loader']

def build_loader(cfg, data_cfg):
    # setup data
    # if data_cfg.NAME == "interhuman":
    #     train_dataset = InterHumanDataset(data_cfg)
    if data_cfg.NAME == "duet":
        train_dataset = Text2Duet(data_cfg, data_cfg.music_root, data_cfg.motion_root, data_cfg.text_root, data_cfg.MODE)
    elif data_cfg.NAME == "interhuman":
        train_dataset = InterHumanDataset(data_cfg)
    elif data_cfg.NAME == "DD100":
        train_dataset = DD100(data_cfg, data_cfg.music_root, data_cfg.motion_root, data_cfg.MODE)
    else:
        raise NotImplementedError

    loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        num_workers=1,
        pin_memory=False,
        shuffle=True,
        drop_last=True,
        )

    return loader

class DataModule(pl.LightningDataModule):
    def __init__(self, cfg, val_cfg, test_cfg, batch_size, num_workers):
        """
        Initialize LightningDataModule for ProHMR training
        Args:
            cfg (CfgNode): Config file as a yacs CfgNode containing necessary dataset info.
            dataset_cfg (CfgNode): Dataset configuration file
        """
        super().__init__()
        self.cfg = cfg
        self.val_cfg = val_cfg
        self.test_cfg = test_cfg
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage = None):
        """
        Create train and validation datasets
        """
        if (self.cfg is not None and self.cfg.NAME == "duet") or self.test_cfg.NAME == "duet":
            if self.cfg is not None:
                self.train_dataset = Text2Duet(self.cfg, self.cfg.music_root, self.cfg.motion_root, self.cfg.text_root, self.cfg.MODE)
            if self.val_cfg is not None:
                self.val_dataset = Text2Duet(self.val_cfg, self.val_cfg.music_root, self.val_cfg.motion_root, self.val_cfg.text_root, self.val_cfg.MODE)
            if self.test_cfg is not None:
                self.test_dataset = Text2Duet(self.test_cfg, self.test_cfg.music_root, self.test_cfg.motion_root, self.test_cfg.text_root, self.test_cfg.MODE)
        elif (self.cfg is not None and self.cfg.NAME == "interhuman") or self.test_cfg.NAME == "interhuman":
            if self.cfg is not None:
                self.train_dataset = InterHumanDataset(self.cfg)
            if self.val_cfg is not None:
                self.val_dataset = InterHumanDataset(self.val_cfg)
            if self.test_cfg is not None:
                self.test_dataset = InterHumanDataset(self.test_cfg)
        if (self.cfg is not None and self.cfg.NAME == "DD100") or self.test_cfg.NAME == "DD100":
            if self.cfg is not None:
                self.train_dataset = DD100(self.cfg, self.cfg.music_root, self.cfg.motion_root, self.cfg.MODE)
            if self.val_cfg is not None:
                self.val_dataset = DD100(self.val_cfg, self.val_cfg.music_root, self.val_cfg.motion_root, self.val_cfg.MODE)
            if self.test_cfg is not None:
                self.test_dataset = DD100(self.test_cfg, self.test_cfg.music_root, self.test_cfg.motion_root, self.test_cfg.MODE)
                
        else:
            raise NotImplementedError

    def train_dataloader(self):
        """
        Return train dataloader
        """
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
            shuffle=True,
            drop_last=True,
            )
    
    def val_dataloader(self):
        """
        Return validation dataloader
        """
        # Follow the offical evaluation
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
            shuffle=True,
            drop_last=True,
            ) # shuffle = True
    
    def test_dataloader(self):
        """
        Return test dataloader
        """
        # Follow the offical evaluation
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=1,
            num_workers=self.num_workers,
            pin_memory=False,
            shuffle=True,
            drop_last=False,
            ) # shuffle = True