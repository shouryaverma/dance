import numpy as np
import torch
import torch.utils.data
from torch.utils.data import Dataset
import os

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.utils import rigid_transform
from utils.quaternion import *
from tqdm import tqdm

def matrix_to_rotation_6d_np(matrix: np.ndarray) -> np.ndarray:
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al.
    by dropping the last row. Note that 6D representation is not unique.
    
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6)
    """
    return matrix[..., :2, :].reshape(*matrix.shape[:-2], 6).copy()

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

# BUG: fix it
def process_motion_np(motion, rotations, feet_thre = 0.001, n_joints =  22):
    face_joint_indx = [2, 1, 17, 16]
    fid_r, fid_l = [8, 11], [7, 10]
    # (seq_len, joints_num, 3)
    #     '''Down Sample'''
    #     positions = positions[::ds_num]

    '''Uniform Skeleton'''
    # positions = uniform_skeleton(positions, tgt_offsets)

    positions = motion[:, :n_joints*3].reshape(-1, n_joints, 3)

    '''Put on Floor'''
    floor_height = positions.min(axis=0).min(axis=0)[1]
    positions[:, :, 1] -= floor_height


    '''XZ at origin'''
    root_pos_init = positions[0]
    root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
    positions = positions - root_pose_init_xz

    '''All initially face Z+'''
    r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
    across = root_pos_init[r_hip] - root_pos_init[l_hip]
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

    # forward (3,), rotate around y-axis
    forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
    # forward (3,)
    forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]

    target = np.array([[0, 0, 1]])
    root_quat_init = qbetween_np(forward_init, target)
    root_quat_init_for_all = np.ones(positions.shape[:-1] + (4,)) * root_quat_init


    positions = qrot_np(root_quat_init_for_all, positions)

    """ Get Foot Contacts """

    def foot_detect(positions, thres):
        velfactor, heightfactor = np.array([thres, thres]), np.array([0.12, 0.05])

        feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
        feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
        feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
        feet_l_h = positions[:-1,fid_l,1]
        feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)).astype(np.float32)

        feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
        feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
        feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
        feet_r_h = positions[:-1,fid_r,1]
        feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)).astype(np.float32)
        return feet_l, feet_r
    #
    feet_l, feet_r = foot_detect(positions, feet_thre)


    '''Get Joint Rotation Representation'''
    rot_data = rotations

    '''Get Joint Rotation Invariant Position Represention'''
    joint_positions = positions.reshape(len(positions), -1)
    joint_vels = positions[1:] - positions[:-1]
    joint_vels = joint_vels.reshape(len(joint_vels), -1)

    data = joint_positions[:-1]
    data = np.concatenate([data, joint_vels], axis=-1)
    data = np.concatenate([data, rot_data[:-1]], axis=-1)
    data = np.concatenate([data, feet_l, feet_r], axis=-1)

    return data, root_quat_init, root_pose_init_xz[None]

class DD100(Dataset):
    def __init__(self, cfg, music_root, motion_root, split='train', fps = 30, dtype='pos3d', music_dance_rate=1):
        self.dances = {'rotmatl':[], 'rotmatf':[], 'pos3dl':[], 'pos3df':[], 'music':[], 'music_wav_path': []}
        dtypes = ['rotmat', 'pos3d']
        self.cfg = cfg
        self.dtype = dtype
        self.names = []
        self.music_root = music_root
        self.motion_root = motion_root
        self.music_files = {}
        self.agent_files = {'leader':{}, 'follower':{}}
        #self.text_files = {}
        #self.music_seqs = {}
        skip = int(30/fps)
        self.max_len = self.max_length = 300

        # We need format: joint positions, joint velocity, joint rotations, foot contacts
        if isinstance(split, str):
            self.load_split(split)
        elif isinstance(split, list):
            for spl in split:
                self.load_split(spl)
        else:
            raise ValueError("split should be either a string or a list of strings")
        agent_files = self.agent_files
        music_files = self.music_files
        #text_files = self.text_files
        # music files, text files and motion files are ready
        for take in agent_files['follower']:
            # print(take)
            if take not in agent_files['leader'] or take not in music_files:
                continue
            # music:
            music_path = music_files[take]
            np_music = np.load(music_path).astype(np.float32) # music features
            
            #load the txt file from text_path
            #text_path = text_files[take]
            #text = open(text_path, "r").readlines()[0] # first line
            # original music sequence
            #music_wav_path = self.music_seqs[take]
            #self.dances['music_wav_path'].append(music_wav_path)
            for dtype_folder in dtypes:
                this_pair = {}
                for agent in agent_files:
                    dance_path = agent_files[agent][take].replace('pos3d', dtype_folder)
                    np_dance = np.load(dance_path)
                    np_dance = np_dance[::skip]
                    
                    if dtype_folder == 'pos3d':
                        # T, 165 = 3*55
                        np_dance = np_dance[:, :66] # 22*3, all global
                    if dtype_folder == 'rotmat':
                        # T, 498
                        np_dance = np_dance[:, 3:] # T, 55*9
                        np_dance = np_dance[:, :9*22] # 22 global + local rotations 
                        np_dance = matrix_to_rotation_6d_np(np_dance.reshape(-1, 22, 3, 3))[:, 1:] # T, 22, 6
                        np_dance = np_dance.reshape(-1, 21*6) 

                    this_pair[agent] = np_dance
                

                ldance, fdance = this_pair['leader'], this_pair['follower']

                lenf, dim = ldance.shape
                lenl, dim = fdance.shape
                seq_len = min(lenf, lenl)
                # print(lenf, lenl)
                self.dances[dtype_folder+'l'].append(ldance[:seq_len])
                self.dances[dtype_folder+'f'].append(fdance[:seq_len])
                if dtype_folder != 'rotmat':
                    self.dances['music'].append(np_music[:seq_len])
                    #self.dances['text'].append(text)
                    self.names.append(take)
        # NOTE: now we have the primary data
        print('dataset first-round loading done')
        
        print(f"# rotmatl: {len(self.dances['rotmatl'])}")
        print(f"# rotmatf: {len(self.dances['rotmatf'])}")
        print(f"# pos3dl: {len(self.dances['pos3dl'])}")
        print(f"# pos3df: {len(self.dances['pos3df'])}")
        print(f"# music: {len(self.dances['music'])}")
        
        self.dances_processed = {'motionl':[], 'motionf':[], 'music':[], 'length':[], 'music_wav_path': []}
        for index in tqdm(range(len(self.dances['music']))):
            #music_wav_path = self.dances['music_wav_path'][index]
            music = self.dances['music'][index]
            #text = self.dances['text'][index]
            rotation_6d_l = self.dances['rotmatl'][index]
            rotation_6d_f = self.dances['rotmatf'][index]
            pos3d_l = self.dances['pos3dl'][index]
            pos3d_f = self.dances['pos3df'][index]
            joints_l_canon, joints_f_canon= canonicalize_sequence(pos3d_l, pos3d_f)
            motion1, root_quat_init1, root_pos_init1 = process_motion_np(joints_l_canon, rotation_6d_l)
            motion2, root_quat_init2, root_pos_init2 = process_motion_np(joints_f_canon, rotation_6d_f)
            r_relative = qmul_np(root_quat_init2, qinv_np(root_quat_init1))
            angle = np.arctan2(r_relative[:, 2:3], r_relative[:, 0:1])

            xz = qrot_np(root_quat_init1, root_pos_init2 - root_pos_init1)[:, [0, 2]]
            relative = np.concatenate([angle, xz], axis=-1)[0]
            motion2 = rigid_transform(relative, motion2)
            assert len(motion1) == len(motion2)
            self.dances_processed['motionl'].append(motion1)
            self.dances_processed['motionf'].append(motion2)
            self.dances_processed['music'].append(music)
            #self.dances_processed['text'].append(text)
            self.dances_processed['length'].append(len(motion1))
            #self.dances_processed['music_wav_path'].append(music_wav_path)
            
    
    def load_split(self, split):
        if self.cfg.music_rep == "simple":           
            for mname in os.listdir(os.path.join(self.music_root,  'feature', split)):
                path = os.path.join(self.music_root, 'feature', split, mname)
                self.music_files[mname[:-4]] = path
                
        elif self.cfg.music_rep == "jukebox":
            for mname in os.listdir(os.path.join(self.music_root,  'Juke_features', split)):
                path = os.path.join(self.music_root, 'Juke_features', split, mname)
                self.music_files[mname[:-4]] = path

        # for mname in os.listdir(os.path.join(self.music_root,  'mp3', 'all')):
        #     path = os.path.join(self.music_root, 'mp3', 'all', mname)
        #     self.music_seqs[mname[:-4]] = path
            


        # for genre in os.listdir(os.path.join(self.text_root, 'processed', split)):
        #     for root, dirs, tnames in os.walk(os.path.join(self.text_root,  'processed', split, genre)):
        #         for tname in tnames:
        #             path = os.path.join(root, tname)
        #             self.text_files[tname[:-4]] = path
        
        for fname in os.listdir(os.path.join(self.motion_root, 'pos3d', split)): 
            path = os.path.join(self.motion_root, 'pos3d', split, fname)
            if not os.path.isfile(path):
                continue  # Skip directories or non-files
            if path.endswith('_00.npy'):
                self.agent_files['follower'][fname[:-7]] = path
            elif path.endswith('_01.npy'):
                self.agent_files['leader'][fname[:-7]] = path

        # for genre in os.listdir(os.path.join(self.motion_root,  'pos3d', split)):
        #     for root, dirs, fnames in os.walk(os.path.join(self.motion_root,  'pos3d', split, genre)):
        #         for fname in fnames:
        #             path = os.path.join(root, fname)
        #             if path.endswith('_00.npy'):
        #                 self.agent_files['follower'][fname[:-7]] = path
        #             elif path.endswith('_01.npy'):
        #                 self.agent_files['leader'][fname[:-7]] = path

    def __len__(self):
        return len(self.dances['pos3dl'])

    def __getitem__(self, index):
        # motion1, motion2, music feature, text, motion length
        motion1 = self.dances_processed['motionl'][index]
        motion2 = self.dances_processed['motionf'][index]
        music = self.dances_processed['music'][index]
        #text = self.dances_processed['text'][index]
        length = self.dances_processed['length'][index]
        
        # Padding motions to max_len
        motion1 = np.pad(motion1, ((0, self.max_len - motion1.shape[0]), (0, 0)), mode='constant')
        motion2 = np.pad(motion2, ((0, self.max_len - motion2.shape[0]), (0, 0)), mode='constant')
        music = np.pad(music, ((0, self.max_len - music.shape[0]), (0, 0)), mode='constant')
        # NOTE: dict is more flexible
        # NOTE: load relevant sequences ()
        # NOTE: fname means id.
        # NOTE: text file id
        item_dict = {
            'motion1': motion1,
            'motion2': motion2,
            'music': music,
            'text': "Placeholder",
            'length': length,
            'fname':self.names[index]
            #'music_wav_path': self.dances_processed['music_wav_path'][index]
        }
        # return item_dict['motion1'], item_dict['motion2'], item_dict['music'], item_dict['text'], item_dict['length']
        return item_dict

# DEBUG CODE
from utils import paramUtil
from utils.plot_script import *
def plot_t2m(mp_data, result_path, caption):
    mp_joint = []
    for i, data in enumerate(mp_data):
        if i == 0:
            joint = data[:,:22*3].reshape(-1,22,3)
        else:
            joint = data[:,:22*3].reshape(-1,22,3)

        mp_joint.append(joint)

    plot_3d_motion(result_path, paramUtil.t2m_kinematic_chain, mp_joint, title=caption, fps=30)

# unit test
if __name__ == '__main__':

    music_root = '/scratch/gilbreth/gupta596/MotionGen/DualFlow/dance/data_folders/DuoLando/DD100/music'
    motion_root = '/scratch/gilbreth/gupta596/MotionGen/DualFlow/dance/data_folders/DuoLando/DD100/motion'

    cfg = {
        'music_rep': 'simple'
    }
    from types import SimpleNamespace
    cfg = SimpleNamespace(**cfg)
    
    t2d = DD100(cfg, music_root, motion_root, split='train', dtype='pos3d', music_dance_rate=1)
    print(len(t2d))
    print(t2d[0])
    item_dict = t2d[0] # music: T-1, 54
    length = item_dict['length']
    motion_1 = item_dict['motion1']
    motion_2 = item_dict['motion2']
    music = item_dict['music']
    text = item_dict['text']
    print(music)
    print(type(music))
    exit()
    print(length)
    print(motion_1.shape)   #(300, 262)
    print(music.shape)      #(300, 54)
    result_path = "results/debug_ori.mp4"
    plot_t2m([motion_1, motion_2],
                      result_path,
                      text)
    # TODO: visualization code
