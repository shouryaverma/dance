import sys
sys.path.append(sys.path[0]+r"/../")
import numpy as np
import torch

from datetime import datetime
from datasets import get_dataset_motion_loader, get_motion_loader
from models import *
from utils.metrics import *
from datasets import EvaluatorModelWrapper
from collections import OrderedDict
from utils.plot_script import *
from utils.utils import *
from configs import get_config
from os.path import join as pjoin
from tqdm import tqdm

os.environ['WORLD_SIZE'] = '1'
os.environ['RANK'] = '0'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12345'
torch.multiprocessing.set_sharing_strategy('file_system')

def build_models(cfg):
    # if cfg.NAME == "InterGen":
    #     model = InterGen(cfg)
    if cfg.NAME == "DuetModel":
        model = DuetModel(cfg)
    elif cfg.NAME == "ReactModel":
        model = ReactModel(cfg)
    else:
        raise NotImplementedError
    return model


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
        print(f"Processing loader: {motion_loader_name}")
        with torch.no_grad():
            try:
                # Check if loader is empty
                if len(motion_loader) == 0:
                    print(f"WARNING: Motion loader '{motion_loader_name}' is empty!")
                    continue
                
                for idx, batch in tqdm(enumerate(motion_loader)):
                    try:
                        # # Debug batch contents
                        # print(f"Processing batch {idx}, type: {type(batch)}")
                        # if isinstance(batch, dict):
                        #     print(f"Batch keys: {batch.keys()}")
                        # elif isinstance(batch, (list, tuple)):
                        #     print(f"Batch length: {len(batch)}")
                            
                        text_embeddings, motion_embeddings = eval_wrapper.get_co_embeddings(batch)
                        # print(f"Got embeddings - text: {text_embeddings.shape}, motion: {motion_embeddings.shape}")
                        
                        # Compute distance matrix
                        dist_mat = euclidean_distance_matrix(text_embeddings.cpu().numpy(),
                                                            motion_embeddings.cpu().numpy())
                        mm_dist_sum += dist_mat.trace()

                        argsmax = np.argsort(dist_mat, axis=1)
                        top_k_mat = calculate_top_k(argsmax, top_k=3)
                        top_k_count += top_k_mat.sum(axis=0)

                        all_size += text_embeddings.shape[0]
                        all_motion_embeddings.append(motion_embeddings.cpu().numpy())
                    except Exception as e:
                        # print(f"Error processing batch {idx}: {str(e)}")
                        import traceback
                        traceback.print_exc()
                        continue
                
                # Handle empty embeddings list
                if len(all_motion_embeddings) == 0:
                    print(f"WARNING: No valid embeddings found for '{motion_loader_name}'")
                    # Add placeholder values
                    match_score_dict[motion_loader_name] = float('nan')
                    R_precision_dict[motion_loader_name] = np.array([float('nan')] * 3)
                    activation_dict[motion_loader_name] = np.zeros((1, 512))  # Placeholder
                    continue
                
                all_motion_embeddings = np.concatenate(all_motion_embeddings, axis=0)
                if all_size > 0:
                    mm_dist = mm_dist_sum / all_size
                    R_precision = top_k_count / all_size
                else:
                    mm_dist = float('nan')
                    R_precision = np.array([float('nan')] * 3)
                
                match_score_dict[motion_loader_name] = mm_dist
                R_precision_dict[motion_loader_name] = R_precision
                activation_dict[motion_loader_name] = all_motion_embeddings
            
            except Exception as e:
                print(f"Error processing loader '{motion_loader_name}': {str(e)}")
                import traceback
                traceback.print_exc()
                
                # Add placeholder values
                match_score_dict[motion_loader_name] = float('nan')
                R_precision_dict[motion_loader_name] = np.array([float('nan')] * 3)
                activation_dict[motion_loader_name] = np.zeros((1, 512))  # Placeholder

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
                # (1, mm_replications, dim_pos)
                batch[2] = batch[2][0]
                batch[3] = batch[3][0]
                batch[4] = batch[4][0]
                motion_embedings = eval_wrapper.get_motion_embeddings(batch)
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


def evaluation(log_file):
    with open(log_file, 'w') as f:
        all_metrics = OrderedDict({'MM Distance': OrderedDict({}),
                                   'R_precision': OrderedDict({}),
                                   'FID': OrderedDict({}),
                                   'Diversity': OrderedDict({}),
                                   'MultiModality': OrderedDict({})})
                                   
        for replication in range(replication_times):
            motion_loaders = {}
            mm_motion_loaders = {}
            motion_loaders['ground truth'] = gt_loader
            for motion_loader_name, motion_loader_getter in eval_motion_loaders.items():
                motion_loader, mm_motion_loader = motion_loader_getter()
                motion_loaders[motion_loader_name] = motion_loader
                mm_motion_loaders[motion_loader_name] = mm_motion_loader

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

            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            mm_score_dict = evaluate_multimodality(mm_motion_loaders, f)

            print(f'!!! DONE !!!')
            print(f'!!! DONE !!!', file=f, flush=True)

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

            for key, item in mm_score_dict.items():
                if key not in all_metrics['MultiModality']:
                    all_metrics['MultiModality'][key] = [item]
                else:
                    all_metrics['MultiModality'][key] += [item]


        # print(all_metrics['Diversity'])
        for metric_name, metric_dict in all_metrics.items():
            print('========== %s Summary ==========' % metric_name)
            print('========== %s Summary ==========' % metric_name, file=f, flush=True)

            for model_name, values in metric_dict.items():
                # print(metric_name, model_name)
                mean, conf_interval = get_metric_statistics(np.array(values))
                # print(mean, mean.dtype)
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
    mm_num_samples = 10
    mm_num_repeats = 10
    mm_num_times = 1

    diversity_times = 5 # 100
    replication_times = 5 # 20

    # batch_size is fixed to 64!!
    batch_size = 5
    data_cfg = get_config("configs/datasets_duet.yaml").test_set
    cfg_path_list = ["configs/model_duet_debug.yaml"]
    print(f"Using config paths: {cfg_path_list}")

    eval_motion_loaders = {}
    for cfg_path in cfg_path_list:
        model_cfg = get_config(cfg_path)
        device = torch.device('cuda:%d' % 0 if torch.cuda.is_available() else 'cpu')
        torch.cuda.set_device(0)
        
        # Build model
        model = build_models(model_cfg)
        
        # Load checkpoint with improved error handling
        if model_cfg.CHECKPOINT:
            checkpoint = torch.load(model_cfg.CHECKPOINT, map_location='cpu')
            if "state_dict" in checkpoint:
                # Handle case where checkpoint has 'state_dict' key
                for k in list(checkpoint["state_dict"].keys()):
                    if "model" in k:
                        checkpoint["state_dict"][k.replace("model.", "")] = checkpoint["state_dict"].pop(k)
                model.load_state_dict(checkpoint["state_dict"], strict=False)
            elif "model" in checkpoint:
                # Handle case where checkpoint has 'model' key
                model.load_state_dict(checkpoint["model"], strict=False)
            else:
                # Handle case where checkpoint is already a state_dict
                model.load_state_dict(checkpoint, strict=False)
            print(f"Checkpoint {model_cfg.CHECKPOINT} loaded successfully!")

        eval_motion_loaders[model_cfg.NAME] = lambda: get_motion_loader(
                                                batch_size,
                                                model,
                                                gt_dataset,
                                                device,
                                                mm_num_samples,
                                                mm_num_repeats,
                                                eval_sample_size=10
                                                )

    device = torch.device('cuda:%d' % 0 if torch.cuda.is_available() else 'cpu')
    gt_loader, gt_dataset = get_dataset_motion_loader(data_cfg, batch_size)
    evalmodel_cfg = get_config("configs/eval_duet_debug.yaml")
    eval_wrapper = EvaluatorModelWrapper(evalmodel_cfg, device)

    log_file = f'./evaluation_{1}.log'
    with torch.no_grad():
        evaluation(log_file)
