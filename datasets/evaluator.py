from os.path import join as pjoin
from torch.utils.data import Dataset, DataLoader
from datasets.text2duet import Text2Duet
from models import *
import copy
from datasets.evaluator_models import InterCLIP
from tqdm import tqdm

class EvaluationDataset(Dataset):
    def __init__(self, model, dataset, device, mm_num_samples, mm_num_repeats, eval_sample_size=None):
        self.normalizer = MotionNormalizer()
        self.model = model.to(device)
        self.model.eval()

        self.is_react_model = False
        print(f"Model is ReactModel: {self.is_react_model}")
        
        # Get a subset of indices for evaluation
        total_samples = len(dataset)
        if eval_sample_size is not None and eval_sample_size < total_samples:
            # Randomly select eval_sample_size indices
            all_idxs = list(range(total_samples))
            random.shuffle(all_idxs)
            selected_idxs = all_idxs[:eval_sample_size]
            # Create a subset loader with only these indices
            subset_dataset = torch.utils.data.Subset(dataset, selected_idxs)
            dataloader = DataLoader(subset_dataset, batch_size=1, num_workers=0, shuffle=False)
            print(f"Using {eval_sample_size} randomly selected samples for evaluation")
        else:
            # Use the full dataset
            dataloader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=True)
            selected_idxs = list(range(total_samples))
            
        self.max_length = dataset.max_length if hasattr(dataset, 'max_length') else 300  # Default fallback
        print(f"Using max_length: {self.max_length}")

        # For multimodal evaluation, select from the subset
        mm_count = min(mm_num_samples, len(selected_idxs))
        mm_indices = random.sample(range(len(selected_idxs)), mm_count)
        mm_idxs = [i for i, idx in enumerate(selected_idxs) if i in mm_indices]
        print(f"Selected {len(mm_idxs)} indices for multimodal sampling")

        generated_motions = []
        mm_generated_motions = []
        
        # Check data format from first item
        first_item = dataset[0]
        is_dict_dataset = isinstance(first_item, dict)
        print(f"Dataset format: {'dictionary' if is_dict_dataset else 'tuple'}")
        
        # Pre-process all target captions
        with torch.no_grad():
            for i, data in tqdm(enumerate(dataloader)):
                try:
                    batch = {}
                    
                    # Handle dictionary format (Text2Duet)
                    if is_dict_dataset or isinstance(data, dict):
                        #print(f"Processing dictionary batch {i} with keys: {data.keys() if isinstance(data, dict) else 'N/A'}")
                        
                        if isinstance(data, dict):
                            motion1, motion2 = data['motion1'], data['motion2']
                            text, motion_lens = data['text'], data['length']
                            music = data.get('music', None)
                            fname = data.get('fname', ["unknown"])
                        else:
                            # Dictionary was wrapped in a list by DataLoader
                            batch_dict = data[0] if isinstance(data, list) else data
                            motion1, motion2 = batch_dict['motion1'], batch_dict['motion2']
                            text, motion_lens = batch_dict['text'], batch_dict['length']
                            music = batch_dict.get('music', None)
                            fname = batch_dict.get('fname', ["unknown"])
                        
                        if i in mm_idxs:
                            batch["text"] = [text[0]] * mm_num_repeats
                            if music is not None:
                                batch["music"] = music.repeat(mm_num_repeats, 1, 1).to(device) if isinstance(music, torch.Tensor) else torch.tensor(music).repeat(mm_num_repeats, 1, 1).to(device)
                        else:
                            batch["text"] = [text[0]]
                            if music is not None:
                                batch["music"] = music.to(device) if isinstance(music, torch.Tensor) else torch.tensor(music).to(device)
                        
                        batch["motion_lens"] = motion_lens
                    
                    # Handle tuple format (InterHuman)
                    else:
                        print(f"Processing tuple batch {i}")
                        name, text, motion1, motion2, motion_lens = data
                        
                        if i in mm_idxs:
                            batch["text"] = [text[0]] * mm_num_repeats
                        else:
                            batch["text"] = [text[0]]
                        
                        batch["motion_lens"] = motion_lens
                    
                    # print(f"Batch prepared with text: {batch['text'][0][:30]}...")
                    
                    # Forward pass through model
                    batch = self.model.forward_test(batch)
                    
                    # Process output
                    if "output" in batch:
                        # print(f"Model output shape: {batch['output'].shape}")
                        motions_output = batch["output"].reshape(batch["output"].shape[0], batch["output"].shape[1], 2, -1)
                        motions_output = self.normalizer.backward(motions_output.cpu().detach().numpy())
                        if self.is_react_model:
                            # Get original ground truth leader motion
                            gt_leader = motion1.cpu().numpy()
                            # Replace the generated leader with ground truth leader
                            T = min(motions_output.shape[1], gt_leader.shape[1])
                            motions_output[0, :T, 0] = gt_leader[0, :T]
                            print("ReactModel: Using ground truth leader motion for evaluation")    

                        B, T = motions_output.shape[0], motions_output.shape[1]
                        if T < self.max_length:
                            padding_len = self.max_length - T
                            D = motions_output.shape[-1]
                            padding_zeros = np.zeros((B, padding_len, 2, D))
                            motions_output = np.concatenate((motions_output, padding_zeros), axis=1)
                        
                        # Make sure output shape is consistent
                        assert motions_output.shape[1] == self.max_length
                        
                        # Store results
                        sub_dict = {
                            'motion1': motions_output[0, :, 0],
                            'motion2': motions_output[0, :, 1],
                            'motion_lens': motion_lens[0],
                            'text': text[0]
                        }
                        
                        if is_dict_dataset and music is not None:
                            sub_dict['music'] = music[0].cpu().numpy() if isinstance(music, torch.Tensor) else music[0]
                            
                        generated_motions.append(sub_dict)
                        
                        # Handle multimodal sampling
                        if i in mm_idxs:
                            mm_sub_dict = {
                                'mm_motions': motions_output,
                                'motion_lens': motion_lens[0],
                                'text': text[0]
                            }
                            
                            if is_dict_dataset and music is not None:
                                mm_sub_dict['music'] = music[0].cpu().numpy() if isinstance(music, torch.Tensor) else music[0]
                                
                            mm_generated_motions.append(mm_sub_dict)
                    else:
                        print(f"WARNING: No 'output' in model results. Keys: {batch.keys()}")
                    
                except Exception as e:
                    print(f"Error processing item {i}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue

        self.generated_motions = generated_motions
        self.mm_generated_motions = mm_generated_motions
        print(f"Created evaluation dataset with {len(generated_motions)} samples and {len(mm_generated_motions)} multimodal samples")

    def __len__(self):
        return len(self.generated_motions)

    def __getitem__(self, item):
        data = self.generated_motions[item]
        motion1, motion2, motion_lens, text = data['motion1'], data['motion2'], data['motion_lens'], data['text']
        
        # For tuple-style return for compatibility with existing code
        return "generated", text, motion1, motion2, motion_lens


class MMGeneratedDataset(Dataset):
    def __init__(self, motion_dataset):
        self.dataset = motion_dataset.mm_generated_motions
        self.is_react_model = motion_dataset.is_react_model

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data = self.dataset[item]
        mm_motions = data['mm_motions']
        if self.is_react_model:
            # All generated variations should use the first leader (ground truth)
            leader_motion = mm_motions[0, :, 0]
            for i in range(1, mm_motions.shape[0]):
                mm_motions[i, :, 0] = leader_motion
        motion_lens = data['motion_lens']
        mm_motions1 = mm_motions[:, :, 0]
        mm_motions2 = mm_motions[:, :, 1]
        text = data['text']
        motion_lens = np.array([motion_lens] * mm_motions1.shape[0])
        return "mm_generated", text, mm_motions1, mm_motions2, motion_lens


def get_dataset_motion_loader(opt, batch_size):
    opt = copy.deepcopy(opt)
    # Handle different dataset types
    if opt.NAME == 'interhuman':
        print('Loading dataset %s ...' % opt.NAME)
        dataset = InterHumanDataset(opt)
    elif opt.NAME == 'duet':
        print('Loading dataset %s ...' % opt.NAME)
        # Ensure proper configuration for Text2Duet
        try:
            dataset = Text2Duet(opt, opt.music_root, opt.motion_root, opt.text_root, opt.MODE)
            # Set identifier to help with processing
            dataset.motion_rep = 'duet'
            print(f"Text2Duet dataset loaded with {len(dataset)} samples")
            # Print a few dataset samples to verify
            print(f"Sample types: {type(dataset[0])}")
            # if isinstance(dataset[0], dict):
                # print(f"Sample keys: {dataset[0].keys()}")
                # print(f"Sample shapes - motion1: {dataset[0]['motion1'].shape}, motion2: {dataset[0]['motion2'].shape}, music: {dataset[0]['music'].shape}")
        except Exception as e:
            print(f"Error loading Text2Duet dataset: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    else:
        raise KeyError('Dataset not Recognized !!')

    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0, drop_last=True, shuffle=True)
    print('Ground Truth Dataset Loading Completed!!!')
    return dataloader, dataset


def get_motion_loader(batch_size, model, ground_truth_dataset, device, mm_num_samples, mm_num_repeats, eval_sample_size=None):
    # Create evaluation datasets with limited sample size
    dataset = EvaluationDataset(model, ground_truth_dataset, device, 
                                mm_num_samples=mm_num_samples, 
                                mm_num_repeats=mm_num_repeats,
                                eval_sample_size=eval_sample_size)
    mm_dataset = MMGeneratedDataset(dataset)

    motion_loader = DataLoader(dataset, batch_size=batch_size, drop_last=True, num_workers=0, shuffle=True)
    mm_motion_loader = DataLoader(mm_dataset, batch_size=1, num_workers=0)

    print('Generated Dataset Loading Completed!!!')
    return motion_loader, mm_motion_loader


def build_models(cfg):
    model = InterCLIP(cfg)

    # Load checkpoint mdd interclip
    checkpoint = torch.load("/home/verma198/Public/results/epoch=599-step=16800.ckpt", map_location="cpu")
    # Load checkpoint interhuman interclip
    # checkpoint = torch.load("/home/verma198/epoch=4049-step=712800.ckpt", map_location="cpu")
    for k in list(checkpoint["state_dict"].keys()):
        if "model" in k:
            checkpoint["state_dict"][k.replace("model.", "")] = checkpoint["state_dict"].pop(k)
    model.load_state_dict(checkpoint["state_dict"], strict=True)

    return model


class EvaluatorModelWrapper(object):
    def __init__(self, cfg, device):
        self.model = build_models(cfg)
        self.cfg = cfg
        self.device = device

        self.model = self.model.to(device)
        self.model.eval()
        print("EvaluatorModelWrapper initialized successfully")

    def get_co_embeddings(self, batch_data):
        with torch.no_grad():
            # Enhanced debug information
            # print(f"get_co_embeddings processing batch_data of type: {type(batch_data)}")
            
            # Handle different data formats
            if isinstance(batch_data, dict):
                # print(f"Dictionary batch with keys: {batch_data.keys()}")
                motion1 = batch_data['motion1']
                motion2 = batch_data['motion2']
                text = batch_data['text']
                motion_lens = batch_data.get('length', batch_data.get('motion_lens'))
                music = batch_data.get('music', None)
            else:
                # print(f"Tuple/list batch with length: {len(batch_data)}")
                name, text, motion1, motion2, motion_lens = batch_data
                music = None
            
            # print(f"Data shapes - motion1: {motion1.shape}, motion2: {motion2.shape}, lens: {motion_lens.shape if isinstance(motion_lens, torch.Tensor) else len(motion_lens)}")
            
            # Process motions
            motion1 = motion1.detach().float()
            motion2 = motion2.detach().float()
            motions = torch.cat([motion1, motion2], dim=-1)
            motions = motions.detach().to(self.device).float()

            # Sort by motion length for efficient processing
            if isinstance(motion_lens, torch.Tensor):
                motion_lens_data = motion_lens.data.tolist()
            else:
                motion_lens_data = motion_lens
                
            align_idx = np.argsort(motion_lens_data)[::-1].copy()
            # print(f"Sorted motion lengths: {sorted(motion_lens_data, reverse=True)}")
            
            motions = motions[align_idx]
            if isinstance(motion_lens, torch.Tensor):
                motion_lens = motion_lens[align_idx]
            else:
                motion_lens = np.array(motion_lens)[align_idx]
                motion_lens = torch.tensor(motion_lens, device=self.device)
            
            # Convert text to list if needed
            if isinstance(text, torch.Tensor):
                text = [t for t in text]
            else:
                text = list(text)

            B, T = motions.shape[:2]
            cur_len = torch.LongTensor([min(T, int(m_len)) for m_len in motion_lens]).to(self.device)
            padded_len = cur_len.max()
            # print(f"Batch size: {B}, max seq length: {padded_len}")

            # Prepare batch for model
            batch = {}
            batch["text"] = text
            batch["motions"] = motions.reshape(B, T, -1)[:, :padded_len]
            batch["motion_lens"] = motion_lens
            
            # Add music if available
            if music is not None and isinstance(music, torch.Tensor):
                batch["music"] = music.to(self.device)[:, :padded_len]

            # Get embeddings
            motion_embedding = self.model.encode_motion(batch)['motion_emb']
            text_embedding = self.model.encode_text(batch)['text_emb'][align_idx]
            
            # print(f"Output embedding shapes - text: {text_embedding.shape}, motion: {motion_embedding.shape}")
            return text_embedding, motion_embedding

    def get_motion_embeddings(self, batch_data):
        with torch.no_grad():
            # Enhanced debug information
            # print(f"get_motion_embeddings processing batch_data of type: {type(batch_data)}")
            
            # Handle different data formats
            if isinstance(batch_data, dict):
                # print(f"Dictionary batch with keys: {batch_data.keys()}")
                motion1 = batch_data['motion1']
                motion2 = batch_data['motion2']
                motion_lens = batch_data.get('length', batch_data.get('motion_lens'))
                music = batch_data.get('music', None)
            else:
                # print(f"Tuple/list batch with length: {len(batch_data)}")
                name, text, motion1, motion2, motion_lens = batch_data
                music = None
            
            # print(f"Data shapes - motion1: {motion1.shape}, motion2: {motion2.shape}, lens: {motion_lens.shape if isinstance(motion_lens, torch.Tensor) else len(motion_lens)}")
            
            # Process motions
            motion1 = motion1.detach().float()
            motion2 = motion2.detach().float()
            motions = torch.cat([motion1, motion2], dim=-1)
            motions = motions.detach().to(self.device).float()

            # Sort by motion length
            if isinstance(motion_lens, torch.Tensor):
                motion_lens_data = motion_lens.data.tolist()
            else:
                motion_lens_data = motion_lens
                
            align_idx = np.argsort(motion_lens_data)[::-1].copy()
            # print(f"Sorted motion lengths: {sorted(motion_lens_data, reverse=True)}")
            
            motions = motions[align_idx]
            if isinstance(motion_lens, torch.Tensor):
                motion_lens = motion_lens[align_idx]
            else:
                motion_lens = np.array(motion_lens)[align_idx]
                motion_lens = torch.tensor(motion_lens, device=self.device)

            B, T = motions.shape[:2]
            cur_len = torch.LongTensor([min(T, int(m_len)) for m_len in motion_lens]).to(self.device)
            padded_len = cur_len.max()
            # print(f"Batch size: {B}, max seq length: {padded_len}")

            # Prepare batch
            batch = {}
            batch["motions"] = motions.reshape(B, T, -1)[:, :padded_len]
            batch["motion_lens"] = motion_lens
            
            # Add music if available
            if music is not None and isinstance(music, torch.Tensor):
                batch["music"] = music.to(self.device)[:, :padded_len]

            # Get embeddings
            motion_embedding = self.model.encode_motion(batch)['motion_emb']
            # print(f"Output motion embedding shape: {motion_embedding.shape}")
            
            return motion_embedding
