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

class LitGenModel(pl.LightningModule):
    def __init__(self, model, cfg):
        super().__init__()
        # cfg init
        self.cfg = cfg
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

    def plot_motion_intergen(self, motion1, motion2, length, result_root, caption, mode = 'train', motion = 'recon', fname=""):
        # only plot in the main process
        if self.device.index != 0:
            return
        assert motion in ['gen', 'gt', 'recon'], motion
        assert mode in ['train', 'val', 'test', 'sample'], mode
        motion1 = motion1.cpu().numpy()[:length]
        motion2 = motion2.cpu().numpy()[:length]
        mp_data = [motion1, motion2]
        mp_joint = []
        for i, data in enumerate(mp_data):
            if i == 0:
                joint = data[:,:22*3].reshape(-1,22,3)
            else:
                joint = data[:,:22*3].reshape(-1,22,3)
            mp_joint.append(joint)

        # result_path = Path(result_root) / f"{mode}_{self.current_epoch}_{idx}_{motion}.mp4"
        result_path = Path(result_root) / f"{fname}_{motion}.mp4"
        plot_3d_motion(str(result_path), paramUtil.t2m_kinematic_chain, mp_joint, title=caption, fps=30)

        # joint_file_path_1 = Path(result_root) / f"{mode}_{self.current_epoch}_{idx}_{motion}_l.npy"
        # joint_file_path_2 = Path(result_root) / f"{mode}_{self.current_epoch}_{idx}_{motion}_f.npy"
        # T, 22, 3
        if motion == 'gen':
            joint_file_path_1 = Path(result_root) / f"{fname}_{motion}_l.npy"
            joint_file_path_2 = Path(result_root) / f"{fname}_{motion}_f.npy"
            np.save(joint_file_path_1, mp_joint[0])
            np.save(joint_file_path_2, mp_joint[1])

    def text_length_to_motion_torch(self, text, music, length):
        # text: 1,*
        # length: 1, 
        with torch.no_grad():
            input_batch = {}
            input_batch["text"] = text
            input_batch["music"] = music
            input_batch["motion_lens"] = length
            output_batch = self.model.forward_test(input_batch)
            motions_output = output_batch["output"].reshape(output_batch["output"].shape[0], output_batch["output"].shape[1], 2, -1)
            motions_output = self.normalizerTorch.backward(motions_output.detach())
        return motions_output[:,:,0], motions_output[:,:,1]
    
    def sample_text(self, batch_data, batch_idx, mode):
        motion1, motion2, music, text, motion_lens = batch_data['motion1'], \
            batch_data['motion2'], batch_data['music'], batch_data['text'], batch_data['length']
        fname = batch_data['fname'][0]
        # cond + T
        # generate one sample or two
        motion_gen_1, motion_gen_2 = self.text_length_to_motion_torch(text[0:1], music[0:1], motion_lens[0:1])

        # B, T, 2, 262
        # visualize the 
        self.plot_motion_intergen(motion1[0], motion2[0], 
                        motion_lens[0], self.vis_dir, text[0],
                        mode = mode, motion = 'gt', fname=fname)
        
        self.plot_motion_intergen(motion_gen_1[0], motion_gen_2[0], 
                        motion_lens[0], self.vis_dir, text[0],
                        mode = mode, motion = 'gen', fname=fname)
    
    def sample_given_condition(self, text, music, motion_lens, fname, mode = 'sample'):
        self.model.eval()
        motion_gen_1, motion_gen_2 = self.text_length_to_motion_torch(text, music.to(self.device).float(), motion_lens)
        self.sample_dir = pjoin(self.save_root, 'sample3')
        os.makedirs(self.sample_dir, exist_ok=True)
        # B, T, 2, 262
        self.plot_motion_intergen(motion_gen_1[0], motion_gen_2[0], 
                        motion_lens[0], self.sample_dir, text[0],
                        mode = mode, motion = 'gen', fname=fname)
        # TODO: also save the joint positions

    def test_step(self, batch, batch_idx):
        # if batch_idx > 10:
        #     return
        self.sample_text(batch, batch_idx, 'test')

def build_models(cfg):
    if cfg.NAME == "DuetModel":
        model = DuetModel(cfg)
    elif cfg.NAME == "ReactModel":
        model = ReactModel(cfg)
    else:
        raise NotImplementedError
    return model

def find_file_pairs(directory):
    """
    Returns a dictionary whose keys are the 'base names' (minus extension)
    and values are dictionaries possibly containing 'npy' and 'txt'.
    
    Example of a dictionary entry:
      {
        "AT_004_0_01_012_0_08": {
          "npy": "/path/to/AT_004_0_01_012_0_08.npy",
          "txt": "/path/to/AT_004_0_01_012_0_08.txt"
        },
        ...
      }
    """
    pairs = {}

    for filename in os.listdir(directory):
        full_path = os.path.join(directory, filename)

        # Skip directories or hidden files, etc. if needed:
        if os.path.isdir(full_path):
            continue

        # Check extension
        if filename.endswith(".npy"):
            base_name = filename[:-4]  # remove .npy
            pairs.setdefault(base_name, {})["npy"] = full_path
        elif filename.endswith(".txt"):
            base_name = filename[:-4]  # remove .txt
            pairs.setdefault(base_name, {})["txt"] = full_path

    return pairs

def manual_test(litmodel, test_dataloader):
    """Manual implementation of the test loop"""
    litmodel.eval()
    
    for batch_idx, batch in enumerate(test_dataloader):
        # Move batch to device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(litmodel.device)
        
        # Call the test_step method
        litmodel.test_step(batch, batch_idx)
    
    return {"status": "Completed test"}

if __name__ == '__main__':
    model_cfg = get_config("configs/model_duet_debug.yaml")
    train_cfg = get_config("configs/infer_duet_debug.yaml")
    test_data_cfg = get_config("configs/datasets_duet.yaml").test_set
    datamodule = DataModule(None, None, test_data_cfg, train_cfg.TRAIN.BATCH_SIZE, train_cfg.TRAIN.NUM_WORKERS)
    datamodule.setup()
    model = build_models(model_cfg)

    if model_cfg.CHECKPOINT:
        ckpt = torch.load(model_cfg.CHECKPOINT, map_location="cpu")
        # FIX: Handle different checkpoint formats
        if "state_dict" in ckpt:
            # Handle case where checkpoint has 'state_dict' key
            for k in list(ckpt["state_dict"].keys()):
                if "model" in k:
                    ckpt["state_dict"][k.replace("model.", "")] = ckpt["state_dict"].pop(k)
            model.load_state_dict(ckpt["state_dict"], strict=False)
        elif "model" in ckpt:
            # Handle case where checkpoint has 'model' key
            model.load_state_dict(ckpt["model"], strict=False)
        else:
            # Handle case where checkpoint is already a state_dict
            model.load_state_dict(ckpt, strict=False)
        print("checkpoint state loaded!")

    litmodel = LitGenModel(model, train_cfg).to(torch.device("cuda:0"))
    
    # FIX: Instead of using trainer.test(), implement a simple test loop
    # We've already loaded the model weights manually, so we don't need
    # to use the PyTorch Lightning checkpoint loading mechanism
    print("Starting manual test loop...")
    result = manual_test(litmodel, datamodule.test_dataloader())
    print(f"Test completed with result: {result}")
    
    # # NOTE: This line is replaced with the manual test loop above
    # # trainer.test(litmodel, dataloaders=datamodule.test_dataloader(), ckpt_path=model_cfg.CHECKPOINT)
    
    # # TODO: walk through the folder
    # folder_path = "/scratch/gilbreth/gupta596/MotionGen/Text2Duet/experiment_split/same_text_diff_music"
    # # change to your folder

    # file_pairs = find_file_pairs(folder_path)
    # print(file_pairs)
    # # Now iterate through all potential pairs
    # for base_name, files_dict in file_pairs.items():
    #     npy_path = files_dict.get("npy")
    #     txt_path = files_dict.get("txt")
    #     text = open(txt_path, "r").readlines()[0]
    #     music = np.load(npy_path) # should be T, 54
    #     assert len(music.shape) == 2, music.shape
    #     length = [music.shape[0]]
    #     music =  torch.from_numpy(music).unsqueeze(0) # 1, T, 53
    #     text = [text]s
    #     fname = base_name
    #     litmodel.sample_given_condition(text, music, length, fname, mode = 'sample')