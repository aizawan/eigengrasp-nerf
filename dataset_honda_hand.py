import os
import numpy as np
import imageio 
import yaml
import torch.nn.functional as F
import cv2
import glob
import pickle

from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as R
from sklearn.decomposition import PCA


link_names = [
    'RHand_WRZ',
    'RHand_T1Z', 'RHand_T1Y', 'RHand_T2Y', 'RHand_T3Y',
    'RHand_I1Z', 'RHand_I1Y', 'RHand_I2Y', 'RHand_I3Y',
    'RHand_M1Z', 'RHand_M1Y', 'RHand_M2Y', 'RHand_M3Y',
    'RHand_R1Z', 'RHand_R1Y', 'RHand_R2Y', 'RHand_R3Y']

SPARSE_LINK_NAMES = [
    'RHand_WRZ',
    'RHand_T3Y',
    'RHand_I3Y',
    'RHand_M3Y',
    'RHand_R3Y']

parents = np.array([
    -1,
    0, 1, 2, 3,
    0, 5, 6, 7, 
    0, 9, 10, 11,
    0, 13, 14, 15])

trans_t = lambda t : np.array([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).astype(np.float32)

rot_phi = lambda phi : np.array([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).astype(np.float32)

rot_theta = lambda th : np.array([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).astype(np.float32)


def convert_translation(position, orientation):
    x, y, z = position
    rot = R.from_quat(orientation)
    cv2gl = np.array([[1,0,0,0], [0,-1,0,0], [0,0,-1,0], [0,0,0,1.]])
    rot_m = rot.as_matrix()
    pose_ = np.array([
        [rot_m[0,0], rot_m[0,1], rot_m[0,2], x],
        [rot_m[1,0], rot_m[1,1], rot_m[1,2], y],
        [rot_m[2,0], rot_m[2,1], rot_m[2,2], z],
        [0,0,0,1]
    ])
    return pose_ @ cv2gl


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]) @ c2w
    return c2w


def load_yaml(path):
    with open(path) as f:
        return yaml.full_load(f)


class HondaHandDataset(Dataset):

    def __init__(self, basedir, split, use_eigengrasp=True,
                 n_components=2, half_res=False,
                 testskip=1, bg_depth=65000, num_frames=4, 
                 use_mask=False, use_depth=False, link_mode='full'):

        # img path settings
        self.basedir = basedir
        self.split = split

        # eigengrasp settings
        self.use_eigengrasp = use_eigengrasp
        self.n_components = n_components

        # rendering settings
        self.half_res = half_res
        self.bg_depth = bg_depth
        self.num_frames = num_frames
        self.use_mask = use_mask
        self.use_depth = use_depth

        # link
        self.link_mode = link_mode
        if link_mode == 'full':
            self.link_names = link_names
            self.num_joints = 17
        elif link_mode == 'sparse':
            self.link_names = SPARSE_LINK_NAMES
            self.num_joints = 5

        if self.split == 'same_dist':
            self.skip = 1
        else:
            self.skip = testskip

        self.make_pathlist()
        self.num_imgs = len(self.rgb_paths) // self.skip

        self.poses = self.create_pose_cache()
        links = self.create_link_cache()
        if self.use_eigengrasp:
            links = self.create_eigengrasp_cache(links)
        self.links = links

        self.make_render_params()
        _ = self.set_info()

    def make_pathlist(self):
        _basepath = f'{self.basedir}/{self.split}'
        self.rgb_paths = sorted(glob.glob(f'{_basepath}/*_rgb.png'))
        self.link_paths = sorted(glob.glob(f'{_basepath}/*_link.yml'))
        self.cam_paths = sorted(glob.glob(f'{_basepath}/*_trans.yml'))

        if self.use_depth:
            self.depth_paths = sorted(glob.glob(f'{_basepath}/*_depth.png'))

    def make_link_param(self, link_param):
        links = []
        for link_name in self.link_names:
            p = link_param[link_name]
            p = convert_translation(p['position'], p['orientation'])
            links.append(p)
        return np.array(links)
    
    def create_link_cache(self):

        self.link_cache_path = os.path.join(
            self.basedir, f'link_{self.split}_{self.link_mode}.npy')
        
        if os.path.exists(self.link_cache_path):
            links = np.load(self.link_cache_path)

        else:
            links = []
            for link_path in self.link_paths:
                link_param = load_yaml(link_path)
                link = self.make_link_param(link_param)
                links.append(link)
            # links: [num_samples, num_joints, 4, 4]
            links = np.array(links).astype(np.float32)
            np.save(self.link_cache_path, links)

        return links

    def create_eigengrasp_cache(self, links):

        self.eigengrasp_link_cache_path = os.path.join(
            self.basedir,
            f'link_eigengrasp-{self.split}_d-{self.n_components}_{self.link_mode}.npy')
        
        if os.path.exists(self.eigengrasp_link_cache_path):
            links = np.load(self.eigengrasp_link_cache_path)

        else:
            # links: [num_samples, n_components]
            links = self.create_eigengrasp_rep(links)

            np.save(self.eigengrasp_link_cache_path, links)

        return links


    def create_pose_cache(self):

        self.pose_cache_path = os.path.join(
            self.basedir, f'pose_{self.split}.npy')

        if os.path.exists(self.pose_cache_path):
            poses = np.load(self.pose_cache_path)

        else:
            poses = []
            for cam_path in self.cam_paths:
                cam_param = load_yaml(cam_path)
                pose = convert_translation(
                    cam_param['camera_position'], cam_param['camera_quaternion'])
                poses.append(pose)
            poses = np.array(poses).astype(np.float32)

            np.save(self.pose_cache_path, poses)

        return poses

    def create_eigengrasp_rep(self, links):
        """
        links: [num_data, num_joints, 4, 4]
        """

        is_train = True if self.split == 'same_dist' else False

        n_samples = links.shape[0]
        links = links.reshape(n_samples, -1)

        pca_path = os.path.join(
            self.basedir, f'pca_d-{self.n_components}.pkl')

        if os.path.exists(pca_path):
            with open(pca_path, 'rb') as f:
                self.pca_model = pickle.load(f)
            print('load pca model.')

        else:
            if is_train:
                print('fit pca model.')
                self.pca_model = PCA(n_components=self.n_components)
                self.pca_model.fit(links)

                with open(pca_path, 'wb') as f:
                    pickle.dump(self.pca_model, f)
                print('save pca model.')
        
            else:
                print('please firstly fit pca model from same dist data (train data).')
                import sys; sys.exit()
        
        links_pca = self.pca_model.transform(links)
        return links_pca

    def get_render_params(self):
        return (self.render_gif_imgs,
                self.render_gif_links,
                self.render_links_fixed,
                self.render_poses)

    def set_info(self):
        img = np.array(imageio.imread(self.rgb_paths[0]))
        self.H, self.W = img.shape[:2]

        self.intrinsic_param = load_yaml(
            os.path.join(self.basedir, 'camera_intrinsic_parameters.yml'))

        self.focal = self.intrinsic_param['intrinsic_matrix'][0]

        if self.half_res:
            self.H = self.H//2
            self.W = self.W//2
            self.focal = self.focal/2.

    def get_info(self):
        return self.H, self.W, self.focal, self.num_joints

    def __len__(self):
        return self.num_imgs

    def __getitem__(self, i):

        pose = self.poses[i]
        link = self.links[i]

        img = np.array(imageio.imread(self.rgb_paths[i])).astype(np.float32)
        img = img / 255.

        if self.use_depth:
            depth = np.array(imageio.imread(
                self.depth_paths[i])).astype(np.float32)
            mask = np.where(depth < self.bg_depth, 1, 0).astype(np.float32)

        if self.half_res:
            W, H = self.W, self.H
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)

            if self.use_depth:
                depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_AREA)
                mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

        if self.use_mask:
            img = img * mask[...,np.newaxis] + (1. - mask[...,np.newaxis])

        if self.use_depth:
            return img, pose, link, depth, mask
        else:
            dummy = img
            return img, pose, link, dummy, dummy

    def make_render_params(self, r=1.0):
        render_poses = np.stack(
            [pose_spherical(angle, -30.0, r) \
                for angle in np.linspace(-180,180,self.num_frames+1)[:-1]], 0)

        render_gif_imgs = []
        render_gif_links = []
        link_paths = sorted(glob.glob(
            f'{self.basedir}/render_links/*_link.yml'))
        rgb_paths = sorted(glob.glob(
            f'{self.basedir}/render_links/*_rgb.png'))
        
        step = len(rgb_paths) // self.num_frames
        for i in range(0, len(rgb_paths), step):
            link_param = load_yaml(link_paths[i])
            render_gif_links.append(self.make_link_param(link_param))
            render_gif_imgs.append(imageio.imread(rgb_paths[i]))
        
        render_gif_imgs = \
            (np.array(render_gif_imgs) / 255.).astype(np.float32)
        render_gif_links = \
            np.array(render_gif_links).astype(np.float32)

        if self.use_eigengrasp:
            render_gif_links = self.create_eigengrasp_rep(render_gif_links)
            render_links_fixed = render_gif_links[0, :]
            render_links_fixed = render_links_fixed[np.newaxis, :]
        else:
            render_links_fixed = render_gif_links[0, :, :, :]
            render_links_fixed = render_links_fixed[np.newaxis, :, :, :]

        render_links_fixed = np.repeat(
            render_links_fixed, render_gif_imgs.shape[0], axis=0)

        self.render_gif_imgs = render_gif_imgs
        self.render_gif_links = render_gif_links
        self.render_links_fixed = render_links_fixed
        self.render_poses = render_poses


class MultipleActionsHondaHandDataset(HondaHandDataset):

    def __init__(self, basedir, split, source_actions, target_actions,
                 use_eigengrasp=True, n_components=2, half_res=False,
                 testskip=1, bg_depth=65000, num_frames=4, use_mask=False,
                 use_depth=False, link_mode='full'):

        self.basedir = basedir
        self.split = split

        # for multiple actions
        self.source_actions = source_actions
        self.target_actions = target_actions

        # eigengrasp
        self.n_components = n_components
        self.use_eigengrasp = use_eigengrasp

        # rendering settings
        self.half_res = half_res
        self.bg_depth = bg_depth
        self.num_frames = num_frames
        self.use_mask = use_mask
        self.use_depth = use_depth

        # link
        self.link_mode = link_mode
        if link_mode == 'full':
            self.link_names = link_names
            self.num_joints = 17
        elif link_mode == 'sparse':
            self.link_names = SPARSE_LINK_NAMES
            self.num_joints = 5

        if self.split == 'same_dist':
            self.skip = 1
        else:
            self.skip = testskip

        self.make_pathlist()
        self.num_imgs = len(self.rgb_paths) // self.skip

        self.poses = self.create_pose_cache()
        links = self.create_link_cache()
        if self.use_eigengrasp:
            links = self.create_eigengrasp_cache(links)
        self.links = links

        if split in ['same_dist', 'novel_view', 'novel_pose', 'novel_pose_novel_view']:
            _action = self.target_actions[0]
        elif split in self.target_actions:
            _action = split
        print(split, self.target_actions)
        self.make_render_params(_action)
        _ = self.set_info()

    def make_render_params(self, action, r=1.0):
        render_poses = np.stack(
            [pose_spherical(angle, -30.0, r) \
                for angle in np.linspace(-180,180,self.num_frames+1)[:-1]], 0)

        render_gif_imgs = []
        render_gif_links = []
        link_paths = sorted(glob.glob(
            f'{self.basedir}/render_links/{action}/*_link.yml'))
        rgb_paths = sorted(glob.glob(
            f'{self.basedir}/render_links/{action}/*_rgb.png'))

        step = len(rgb_paths) // self.num_frames
        for i in range(0, len(rgb_paths), step):
            link_param = load_yaml(link_paths[i])
            render_gif_links.append(self.make_link_param(link_param))
            render_gif_imgs.append(imageio.imread(rgb_paths[i]))
        
        render_gif_imgs = \
            (np.array(render_gif_imgs) / 255.).astype(np.float32)
        render_gif_links = \
            np.array(render_gif_links).astype(np.float32)

        if self.use_eigengrasp:
            render_gif_links = self.create_eigengrasp_rep(render_gif_links)
            render_links_fixed = render_gif_links[0, :]
            render_links_fixed = render_links_fixed[np.newaxis, :]
        else:
            render_links_fixed = render_gif_links[0, :, :, :]
            render_links_fixed = render_links_fixed[np.newaxis, :, :, :]

        render_links_fixed = np.repeat(
            render_links_fixed, render_gif_imgs.shape[0], axis=0)

        self.render_gif_imgs = render_gif_imgs
        self.render_gif_links = render_gif_links
        self.render_links_fixed = render_links_fixed
        self.render_poses = render_poses