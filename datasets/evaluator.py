from os.path import join as pjoin
from torch.utils.data import Dataset, DataLoader
from datasets.text2duet import Text2Duet
from models import *
import copy
from datasets.evaluator_models import InterCLIP
from tqdm import tqdm

class EvaluationDataset(Dataset):

    def __init__(self, model, dataset, device, mm_num_samples, mm_num_repeats):
        self.normalizer = MotionNormalizer()
        self.model = model.to(device)
        self.model.eval()
        dataloader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=True)
        self.max_length = dataset.max_length

        idxs = list(range(len(dataset)))
        random.shuffle(idxs)
        mm_idxs = idxs[:mm_num_samples]

        generated_motions = []
        mm_generated_motions = []
        # Pre-process all target captions
        with torch.no_grad():
            for i, data in tqdm(enumerate(dataloader)):
                batch = {}
                if not isinstance(data, dict):
                    name, text, motion1, motion2, motion_lens = data
                    if i in mm_idxs:
                        batch["text"] = list(text) * mm_num_repeats
                    else:
                        batch["text"] = list(text)
                else:
                    motion1, motion2, music, text, motion_lens = data['motion1'], \
                        data['motion2'], data['music'], data['text'], data['length']
                    if i in mm_idxs:
                        batch["text"] = list(text) * mm_num_repeats
                    else:
                        batch["text"] = list(text)
                    if i in mm_idxs:
                        batch["music"] = music.repeat(mm_num_repeats, 1, 1).to(device) # B, T,D
                    else:
                        batch["music"] = music.to(device)
                    
                batch["motion_lens"] = motion_lens

                batch = self.model.forward_test(batch)
                motions_output = batch["output"].reshape(batch["output"].shape[0], batch["output"].shape[1], 2, -1)
                motions_output = self.normalizer.backward(motions_output.cpu().detach().numpy())

                B,T = motions_output.shape[0], motions_output.shape[1]
                if T < self.max_length:
                    padding_len = self.max_length - T
                    D = motions_output.shape[-1]
                    padding_zeros = np.zeros((B, padding_len, 2, D))
                    motions_output = np.concatenate((motions_output, padding_zeros), axis=1)
                assert motions_output.shape[1] == self.max_length


                sub_dict = {'motion1': motions_output[0, :,0],
                            'motion2': motions_output[0, :,1],
                            'motion_lens': motion_lens[0],
                            'text': text[0]}
                generated_motions.append(sub_dict)
                if i in mm_idxs:
                    mm_sub_dict = {'mm_motions': motions_output,
                                   'motion_lens': motion_lens[0],
                                    'text': text[0]}
                    mm_generated_motions.append(mm_sub_dict)


        self.generated_motions = generated_motions
        self.mm_generated_motions = mm_generated_motions

    def __len__(self):
        return len(self.generated_motions)

    def __getitem__(self, item):
        data = self.generated_motions[item]
        motion1, motion2, motion_lens, text = data['motion1'], data['motion2'], data['motion_lens'], data['text']
        return "generated", text, motion1, motion2, motion_lens

class MMGeneratedDataset(Dataset):
    def __init__(self, motion_dataset):
        self.dataset = motion_dataset.mm_generated_motions

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data = self.dataset[item]
        mm_motions = data['mm_motions']
        motion_lens = data['motion_lens']
        mm_motions1 = mm_motions[:,:,0]
        mm_motions2 = mm_motions[:,:,1]
        text = data['text']
        motion_lens = np.array([motion_lens]*mm_motions1.shape[0])
        return "mm_generated", text, mm_motions1, mm_motions2, motion_lens

def get_dataset_motion_loader(opt, batch_size):
    opt = copy.deepcopy(opt)
    if opt.NAME == 'duet':
        dataset = Text2Duet(opt, opt.music_root, opt.motion_root, opt.text_root, opt.MODE)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0, drop_last=True, shuffle=True)
    else:
        raise KeyError('Dataset not Recognized !!')

    print('Ground Truth Dataset Loading Completed!!!')
    return dataloader, dataset

def get_motion_loader(batch_size, model, ground_truth_dataset, device, mm_num_samples, mm_num_repeats):
    # Currently the configurations of two datasets are almost the same
    dataset = EvaluationDataset(model, ground_truth_dataset, device, mm_num_samples=mm_num_samples, mm_num_repeats=mm_num_repeats)
    mm_dataset = MMGeneratedDataset(dataset)

    motion_loader = DataLoader(dataset, batch_size=batch_size, drop_last=True, num_workers=0, shuffle=True)
    mm_motion_loader = DataLoader(mm_dataset, batch_size=1, num_workers=0)

    print('Generated Dataset Loading Completed!!!')

    return motion_loader, mm_motion_loader

def build_models(cfg):
    model = InterCLIP(cfg)
    checkpoint = torch.load("/scratch/gilbreth/gupta596/MotionGen/DualFlow/dance/checkpoints/InterCLIP_MDD/epoch=599-step=16800.ckpt", map_location="cpu")
    
    # Handle different checkpoint formats
    if "state_dict" in checkpoint:
        # Handle case where checkpoint has 'state_dict' key
        for k in list(checkpoint["state_dict"].keys()):
            if "model" in k:
                checkpoint["state_dict"][k.replace("model.", "")] = checkpoint["state_dict"].pop(k)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
    elif "model" in checkpoint:
        # Handle case where checkpoint has 'model' key
        model.load_state_dict(checkpoint["model"], strict=True)
    else:
        # Handle case where checkpoint is already a state_dict
        model.load_state_dict(checkpoint, strict=True)
    
    return model

class EvaluatorModelWrapper(object):

    def __init__(self, cfg, device):

        self.model = build_models(cfg)
        self.cfg = cfg
        self.device = device

        self.model = self.model.to(device)
        self.model.eval()


    # Please note that the results does not following the order of inputs
    def get_co_embeddings(self, batch_data):
        with torch.no_grad():
            if not isinstance(batch_data, dict):
                name, text, motion1, motion2, motion_lens = batch_data
            else:
                motion1, motion2, _, text, motion_lens = batch_data['motion1'], \
                    batch_data['motion2'], batch_data['music'], batch_data['text'], batch_data['length']
        
            motion1 = motion1.detach().float()  # .to(self.device)
            motion2 = motion2.detach().float()  # .to(self.device)
            motions = torch.cat([motion1, motion2], dim=-1)
            motions = motions.detach().to(self.device).float()

            align_idx = np.argsort(motion_lens.data.tolist())[::-1].copy()
            motions = motions[align_idx]
            motion_lens = motion_lens[align_idx]
            text = list(text)

            B, T = motions.shape[:2]
            cur_len = torch.LongTensor([min(T, m_len) for m_len in motion_lens]).to(self.device)
            padded_len = cur_len.max()

            batch = {}
            batch["text"] = text
            batch["motions"] = motions.reshape(B, T, -1)[:, :padded_len]
            batch["motion_lens"] = motion_lens

            '''Motion Encoding'''
            motion_embedding = self.model.encode_motion(batch)['motion_emb']

            '''Text Encoding'''
            text_embedding = self.model.encode_text(batch)['text_emb'][align_idx]

        return text_embedding, motion_embedding

    # Please note that the results does not following the order of inputs
    def get_motion_embeddings(self, batch_data):
        with torch.no_grad():
            if not isinstance(batch_data, dict):
                name, text, motion1, motion2, motion_lens = batch_data
            else:
                motion1, motion2, music, text, motion_lens = batch_data['motion1'], \
                    batch_data['motion2'], batch_data['music'], batch_data['text'], batch_data['length']
        
            motion1 = motion1.detach().float()  # .to(self.device)
            motion2 = motion2.detach().float()  # .to(self.device)
            motions = torch.cat([motion1, motion2], dim=-1)
            motions = motions.detach().to(self.device).float()

            align_idx = np.argsort(motion_lens.data.tolist())[::-1].copy()
            motions = motions[align_idx]
            motion_lens = motion_lens[align_idx]
            text = list(text)

            B, T = motions.shape[:2]
            cur_len = torch.LongTensor([min(T, m_len) for m_len in motion_lens]).to(self.device)
            padded_len = cur_len.max()

            batch = {}
            batch["text"] = text
            batch["motions"] = motions.reshape(B, T, -1)[:, :padded_len]
            batch["motion_lens"] = motion_lens

            '''Motion Encoding'''
            motion_embedding = self.model.encode_motion(batch)['motion_emb']

        return motion_embedding