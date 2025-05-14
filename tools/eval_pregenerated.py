import sys
sys.path.append(sys.path[0]+r"/../")
import numpy as np
import torch
import os
import argparse
from datetime import datetime
from datasets import get_dataset_motion_loader
from utils.metrics import *
from datasets import EvaluatorModelWrapper
from collections import OrderedDict
from utils.plot_script import *
from utils.utils import *
from configs import get_config
from os.path import join as pjoin
from tqdm import tqdm
from pathlib import Path
import random

def get_npy_files_from_directory(directory):
    """Find all leader and follower NPY files in the directory"""
    leader_files = {}
    follower_files = {}
    
    for filename in os.listdir(directory):
        if filename.endswith("_gen_l.npy"):
            base_name = filename.replace("_gen_l.npy", "")
            leader_files[base_name] = os.path.join(directory, filename)
        elif filename.endswith("_gen_f.npy"):
            base_name = filename.replace("_gen_f.npy", "")
            follower_files[base_name] = os.path.join(directory, filename)
    
    # Find matching pairs
    common_keys = set(leader_files.keys()) & set(follower_files.keys())
    paired_files = {
        key: (leader_files[key], follower_files[key]) 
        for key in common_keys
    }
    
    return paired_files

def canonicalize_sequence(motion_1, motion_2):
    # NOTE: velocity is not perturbed
    joints_1 = motion_1[:,:22*3].reshape(-1,22,3)
    joints_2 = motion_2[:,:22*3].reshape(-1,22,3)
    center_positions = (joints_1[:,0] + joints_2[:,0])/2
    # translate first frame to origin
    pelvis_init = center_positions[0:1]
    joints_1 = joints_1 - pelvis_init
    joints_2 = joints_2 - pelvis_init
    motion_1[:,:22*3] = joints_1.reshape(-1, 22*3)
    motion_2[:,:22*3] = joints_2.reshape(-1, 22*3)
    return motion_1, motion_2

class PreGeneratedMotionLoader:
    """Simplified loader for pre-generated motion files"""
    def __init__(self, paired_files, gt_dataset, device, use_gt_leader=False):
        self.paired_files = paired_files
        self.gt_dataset = gt_dataset
        self.device = device
        self.file_keys = list(paired_files.keys())
        self.max_length = getattr(gt_dataset, 'max_length', 300)
        self.use_gt_leader = use_gt_leader
        
    def __iter__(self):
        # Get a batch from ground truth to understand format
        gt_batch = next(iter(gt_loader))
        
        # Get feature dimensions from ground truth
        if isinstance(gt_batch, dict):
            motion1_shape = gt_batch['motion1'].shape  
            feature_dim = motion1_shape[-1]
        else:
            if len(gt_batch) >= 3:
                motion1_shape = gt_batch[2].shape
                feature_dim = motion1_shape[-1]
            else:
                # Default values if we can't determine
                feature_dim = 262  # Standard feature dimension
        
        # Process all files into one batch
        all_leader_motions = []
        all_follower_motions = []
        all_lengths = []
        all_texts = []
        
        # If using GT leader, grab leaders from ground truth
        gt_leaders = None
        gt_lengths = None
        if self.use_gt_leader:
            if isinstance(gt_batch, dict):
                gt_leaders = gt_batch['motion1'].cpu().numpy()
                if 'length' in gt_batch:
                    gt_lengths = gt_batch['length'].cpu().numpy()
            elif len(gt_batch) >= 3:
                gt_leaders = gt_batch[2].cpu().numpy()
                if len(gt_batch) >= 5:
                    gt_lengths = gt_batch[4].cpu().numpy()
            
        for i, key in enumerate(self.file_keys):
            leader_path, follower_path = self.paired_files[key]
            
            # Load follower motion (always from generated files)
            follower_motion = np.load(follower_path)  # T, 22, 3
            
            # Get leader motion - either from GT or from generated files
            if self.use_gt_leader and gt_leaders is not None and i < len(gt_leaders):
                # Use ground truth leader motion
                leader_motion = gt_leaders[i]
                
                # Determine motion length
                if gt_lengths is not None and i < len(gt_lengths):
                    gt_length = int(gt_lengths[i])
                    motion_len = min(len(leader_motion), len(follower_motion), self.max_length, gt_length)
                else:
                    motion_len = min(len(leader_motion), len(follower_motion), self.max_length)
            else:
                # Use pre-generated leader motion
                leader_motion = np.load(leader_path)  # T, 22, 3
                motion_len = min(len(leader_motion), len(follower_motion), self.max_length)
            
            # Reshape to match expected format (flattened joints)
            leader_motion_flat = leader_motion[:motion_len].reshape(motion_len, -1)  # T, 66
            follower_motion_flat = follower_motion[:motion_len].reshape(motion_len, -1)  # T, 66
            
            # Create a feature vector compatible with what the model expects
            # First ensure both leader and follower have the same feature dimension
            if leader_motion_flat.shape[1] != follower_motion_flat.shape[1]:
                min_dim = min(leader_motion_flat.shape[1], follower_motion_flat.shape[1])
                leader_motion_flat = leader_motion_flat[:, :min_dim]
                follower_motion_flat = follower_motion_flat[:, :min_dim]
                
            # Now adjust to match the expected feature dimension
            if leader_motion_flat.shape[1] < feature_dim:
                # Pad features to match expected dimension
                padding = np.zeros((motion_len, feature_dim - leader_motion_flat.shape[1]))
                leader_motion_flat = np.concatenate([leader_motion_flat, padding], axis=1)
                follower_motion_flat = np.concatenate([follower_motion_flat, padding], axis=1)
            elif leader_motion_flat.shape[1] > feature_dim:
                # Truncate features to match expected dimension
                leader_motion_flat = leader_motion_flat[:, :feature_dim]
                follower_motion_flat = follower_motion_flat[:, :feature_dim]
            
            # Pad to max_length if needed for batch processing
            if motion_len < self.max_length:
                pad_length = self.max_length - motion_len
                leader_pad = np.zeros((pad_length, feature_dim))
                follower_pad = np.zeros((pad_length, feature_dim))
                leader_motion_flat = np.concatenate([leader_motion_flat, leader_pad], axis=0)
                follower_motion_flat = np.concatenate([follower_motion_flat, follower_pad], axis=0)
                
            all_leader_motions.append(leader_motion_flat)
            all_follower_motions.append(follower_motion_flat)
            all_lengths.append(motion_len)
            
            # Use a generic text prompt
            all_texts.append("Two dancers performing")
            
        # Stack into tensors
        leader_tensor = torch.tensor(np.stack(all_leader_motions), device=self.device).float()
        follower_tensor = torch.tensor(np.stack(all_follower_motions), device=self.device).float()
        length_tensor = torch.tensor(all_lengths, device=self.device)
        
        # Create a batch with all samples
        batch = {
            'motion1': leader_tensor,
            'motion2': follower_tensor,
            'music': None,  # Not needed for evaluation
            'text': all_texts,
            'length': length_tensor
        }
        
        yield batch
            
    def __len__(self):
        return 1  # One batch with all samples

class MultimodalPreGeneratedLoader:
    """Simplified loader for multimodality evaluation"""
    def __init__(self, paired_files, gt_dataset, device, use_gt_leader=False):
        self.paired_files = paired_files
        self.gt_dataset = gt_dataset
        self.device = device
        self.file_keys = list(paired_files.keys())[:10]  # Just use first 10 samples
        self.max_length = getattr(gt_dataset, 'max_length', 300)
        self.use_gt_leader = use_gt_leader
        
    def __iter__(self):
        # Get a batch from ground truth for GT leaders if needed
        gt_leaders = None
        gt_lengths = None
        
        if self.use_gt_leader:
            gt_batch = next(iter(gt_loader))
            if isinstance(gt_batch, dict):
                gt_leaders = gt_batch['motion1'].cpu().numpy()
                if 'length' in gt_batch:
                    gt_lengths = gt_batch['length'].cpu().numpy()
            elif len(gt_batch) >= 3:
                gt_leaders = gt_batch[2].cpu().numpy()
                if len(gt_batch) >= 5:
                    gt_lengths = gt_batch[4].cpu().numpy()
        
        # Create a simple multimodal batch for each sample
        for i, key in enumerate(self.file_keys):
            leader_path, follower_path = self.paired_files[key]
            
            # Load follower motion (always from generated files)
            follower_motion = np.load(follower_path)  # T, 22, 3
            
            # Get leader motion - either from GT or from generated files
            if self.use_gt_leader and gt_leaders is not None and i < len(gt_leaders):
                # Use ground truth leader motion
                leader_motion = gt_leaders[i]
                
                # Determine motion length
                if gt_lengths is not None and i < len(gt_lengths):
                    gt_length = int(gt_lengths[i])
                    motion_len = min(len(leader_motion), len(follower_motion), self.max_length, gt_length)
                else:
                    motion_len = min(len(leader_motion), len(follower_motion), self.max_length)
            else:
                # Use pre-generated leader motion
                leader_motion = np.load(leader_path)  # T, 22, 3
                motion_len = min(len(leader_motion), len(follower_motion), self.max_length)
            
            # Reshape to match expected format
            leader_flat = leader_motion[:motion_len].reshape(motion_len, -1)
            follower_flat = follower_motion[:motion_len].reshape(motion_len, -1)
            
            # First ensure both leader and follower have the same feature dimension
            if leader_flat.shape[1] != follower_flat.shape[1]:
                min_dim = min(leader_flat.shape[1], follower_flat.shape[1])
                leader_flat = leader_flat[:, :min_dim]
                follower_flat = follower_flat[:, :min_dim]
            
            # Create duplicate copies (5) to simulate multiple generations for the same prompt
            mm_repeats = 5
            leader_repeats = np.tile(leader_flat, (mm_repeats, 1, 1))
            follower_repeats = np.tile(follower_flat, (mm_repeats, 1, 1))
            
            # Convert to tensors
            leader_tensor = torch.tensor(leader_repeats, device=self.device).float()
            follower_tensor = torch.tensor(follower_repeats, device=self.device).float()
            length_tensor = torch.tensor([motion_len] * mm_repeats, device=self.device)
            
            # Create batch in the format expected by evaluator - using a LIST instead of tuple
            # This allows the evaluate_multimodality function to modify elements
            text = ["Two dancers performing"]
            batch = [
                "mm_generated",  # name
                text,            # text
                leader_tensor,   # motion1
                follower_tensor, # motion2
                length_tensor    # motion_lens
            ]
            
            yield batch
    
    def __len__(self):
        return len(self.file_keys)

def evaluate_matching_score(motion_loaders, file):
    match_score_dict = OrderedDict({})
    R_precision_dict = OrderedDict({})
    activation_dict = OrderedDict({})
    print('========== Evaluating MM Distance ==========')
    for motion_loader_name, motion_loader in motion_loaders.items():
        all_motion_embeddings = []
        score_list = []
        all_size = 0
        mm_dist_sum = 0
        top_k_count = 0
        
        with torch.no_grad():
            for idx, batch in tqdm(enumerate(motion_loader)):
                text_embeddings, motion_embeddings = eval_wrapper.get_co_embeddings(batch)
                dist_mat = euclidean_distance_matrix(text_embeddings.cpu().numpy(),
                                                    motion_embeddings.cpu().numpy())
                mm_dist_sum += dist_mat.trace()

                argsmax = np.argsort(dist_mat, axis=1)
                top_k_mat = calculate_top_k(argsmax, top_k=3)
                top_k_count += top_k_mat.sum(axis=0)

                all_size += text_embeddings.shape[0]
                all_motion_embeddings.append(motion_embeddings.cpu().numpy())

            all_motion_embeddings = np.concatenate(all_motion_embeddings, axis=0)
            mm_dist = mm_dist_sum / all_size
            R_precision = top_k_count / all_size
            match_score_dict[motion_loader_name] = mm_dist
            R_precision_dict[motion_loader_name] = R_precision
            activation_dict[motion_loader_name] = all_motion_embeddings

        print(f'---> [{motion_loader_name}] MM Distance: {mm_dist:.4f}')
        print(f'---> [{motion_loader_name}] MM Distance: {mm_dist:.4f}', file=file, flush=True)

        line = f'---> [{motion_loader_name}] R_precision: '
        for i in range(len(R_precision)):
            line += '(top %d): %.4f ' % (i+1, R_precision[i])
        print(line)
        print(line, file=file, flush=True)

    return match_score_dict, R_precision_dict, activation_dict

def evaluate_fid(groundtruth_loader, activation_dict, file):
    eval_dict = OrderedDict({})
    gt_motion_embeddings = []
    print('========== Evaluating FID ==========')
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(groundtruth_loader)):
            motion_embeddings = eval_wrapper.get_motion_embeddings(batch)
            gt_motion_embeddings.append(motion_embeddings.cpu().numpy())
    gt_motion_embeddings = np.concatenate(gt_motion_embeddings, axis=0)
    gt_mu, gt_cov = calculate_activation_statistics(gt_motion_embeddings)

    for model_name, motion_embeddings in activation_dict.items():
        mu, cov = calculate_activation_statistics(motion_embeddings)
        fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
        print(f'---> [{model_name}] FID: {fid:.4f}')
        print(f'---> [{model_name}] FID: {fid:.4f}', file=file, flush=True)
        eval_dict[model_name] = fid
    return eval_dict

def evaluate_diversity(activation_dict, file):
    eval_dict = OrderedDict({})
    print('========== Evaluating Diversity ==========')
    for model_name, motion_embeddings in activation_dict.items():
        diversity = calculate_diversity(motion_embeddings, diversity_times)
        eval_dict[model_name] = diversity
        print(f'---> [{model_name}] Diversity: {diversity:.4f}')
        print(f'---> [{model_name}] Diversity: {diversity:.4f}', file=file, flush=True)
    return eval_dict

def evaluate_multimodality(mm_motion_loaders, file):
    eval_dict = OrderedDict({})
    print('========== Evaluating MultiModality ==========')
    for model_name, mm_motion_loader in mm_motion_loaders.items():
        mm_motion_embeddings = []
        with torch.no_grad():
            for idx, batch in enumerate(mm_motion_loader):
                # Ensure batch has the expected format
                if isinstance(batch, list) and len(batch) >= 3:
                    # Make a copy of the batch to avoid modifying the original
                    batch_copy = batch.copy()
                    batch_copy[2] = batch_copy[2][0] if isinstance(batch_copy[2], list) else batch_copy[2]
                    if len(batch_copy) > 3:
                        batch_copy[3] = batch_copy[3][0] if isinstance(batch_copy[3], list) else batch_copy[3]
                    if len(batch_copy) > 4:
                        batch_copy[4] = batch_copy[4][0] if isinstance(batch_copy[4], list) else batch_copy[4]
                    motion_embedings = eval_wrapper.get_motion_embeddings(batch_copy)
                    mm_motion_embeddings.append(motion_embedings.unsqueeze(0))
        if len(mm_motion_embeddings) == 0:
            multimodality = 0
        else:
            mm_motion_embeddings = torch.cat(mm_motion_embeddings, dim=0).cpu().numpy()
            multimodality = calculate_multimodality(mm_motion_embeddings, mm_num_times)
        print(f'---> [{model_name}] Multimodality: {multimodality:.4f}')
        print(f'---> [{model_name}] Multimodality: {multimodality:.4f}', file=file, flush=True)
        eval_dict[model_name] = multimodality
    return eval_dict

def get_metric_statistics(values):
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    conf_interval = 1.96 * std / np.sqrt(replication_times)
    return mean, conf_interval

def evaluation(log_file, generated_dir, use_gt_leader=False):
    # Get pre-generated motion files
    paired_files = get_npy_files_from_directory(generated_dir)
    
    if not paired_files:
        print(f"No matching leader/follower NPY files found in {generated_dir}")
        return
    
    print(f"Found {len(paired_files)} paired motion files")
    
    # Determine model name based on whether we're using GT leader
    model_name = "PreGenerated_GTLeader" if use_gt_leader else "PreGenerated"
    
    with open(log_file, 'w') as f:
        all_metrics = OrderedDict({
            'MM Distance': OrderedDict({}),
            'R_precision': OrderedDict({}),
            'FID': OrderedDict({}),
            'Diversity': OrderedDict({})  # Removed MultiModality
        })
                               
        for replication in range(replication_times):
            motion_loaders = {}
            
            # Add ground truth loader
            motion_loaders['ground truth'] = gt_loader
            
            # Add pre-generated motion loader with the use_gt_leader option
            pregen_loader = PreGeneratedMotionLoader(
                paired_files, 
                gt_dataset, 
                device, 
                use_gt_leader=use_gt_leader
            )
            
            motion_loaders[model_name] = pregen_loader

            print(f'==================== Replication {replication} ====================')
            print(f'==================== Replication {replication} ====================', file=f, flush=True)
            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            
            mat_score_dict, R_precision_dict, acti_dict = evaluate_matching_score(motion_loaders, f)

            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            fid_score_dict = evaluate_fid(gt_loader, acti_dict, f)

            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            div_score_dict = evaluate_diversity(acti_dict, f)

            print(f'!!! DONE !!!')
            print(f'!!! DONE !!!', file=f, flush=True)
            
            # Store results
            for key, item in mat_score_dict.items():
                if key not in all_metrics['MM Distance']:
                    all_metrics['MM Distance'][key] = [item]
                else:
                    all_metrics['MM Distance'][key] += [item]

            for key, item in R_precision_dict.items():
                if key not in all_metrics['R_precision']:
                    all_metrics['R_precision'][key] = [item]
                else:
                    all_metrics['R_precision'][key] += [item]

            for key, item in fid_score_dict.items():
                if key not in all_metrics['FID']:
                    all_metrics['FID'][key] = [item]
                else:
                    all_metrics['FID'][key] += [item]

            for key, item in div_score_dict.items():
                if key not in all_metrics['Diversity']:
                    all_metrics['Diversity'][key] = [item]
                else:
                    all_metrics['Diversity'][key] += [item]

        # Print summary
        for metric_name, metric_dict in all_metrics.items():
            print('========== %s Summary ==========' % metric_name)
            print('========== %s Summary ==========' % metric_name, file=f, flush=True)

            for model_name, values in metric_dict.items():
                mean, conf_interval = get_metric_statistics(np.array(values))
                if isinstance(mean, np.float64) or isinstance(mean, np.float32):
                    print(f'---> [{model_name}] Mean: {mean:.4f} CInterval: {conf_interval:.4f}')
                    print(f'---> [{model_name}] Mean: {mean:.4f} CInterval: {conf_interval:.4f}', file=f, flush=True)
                elif isinstance(mean, np.ndarray):
                    line = f'---> [{model_name}]'
                    for i in range(len(mean)):
                        line += '(top %d) Mean: %.4f CInt: %.4f;' % (i+1, mean[i], conf_interval[i])
                    print(line)
                    print(line, file=f, flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate pre-generated motion files')
    parser.add_argument('--use_gt_leader', action='store_true', default=False,
                        help='Replace generated leader motions with ground truth leader motions')
    parser.add_argument('--generated_dir', type=str, 
                        default="/depot/bera89/data/li5280/project/iccv25/InterGen/checkpoints/InterGen_Baseline_infer/vis",
                        help='Directory containing pre-generated NPY files')
    args = parser.parse_args()
    
    mm_num_samples = 20
    mm_num_repeats = 10
    mm_num_times = 5

    diversity_times = 3  # 100
    replication_times = 5  # 20

    batch_size = 64
    data_cfg = get_config("/scratch/gilbreth/gupta596/MotionGen/DualFlow/dance/configs/datasets_duet_juke_prerit.yaml").test_set
    
    # Use the directory from command line args
    generated_dir = args.generated_dir
    
    device = torch.device('cuda:%d' % 0 if torch.cuda.is_available() else 'cpu')
    gt_loader, gt_dataset = get_dataset_motion_loader(data_cfg, batch_size)
    evalmodel_cfg = get_config("/scratch/gilbreth/gupta596/MotionGen/DualFlow/dance/configs/eval_duet_debug.yaml")
    eval_wrapper = EvaluatorModelWrapper(evalmodel_cfg, device)

    # Create unique log file name based on the mode
    leader_mode = "gt_leader" if args.use_gt_leader else "gen_leader"
    log_file = f'./evaluation_{leader_mode}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    with torch.no_grad():
        evaluation(log_file, generated_dir, use_gt_leader=args.use_gt_leader)