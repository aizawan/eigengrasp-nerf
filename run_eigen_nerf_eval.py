import os
import glob
import argparse
import numpy as np
from PIL import Image
import torch

from skimage.metrics import structural_similarity

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str,
    default='configs/exp_eigen_nerf_1000/ball_tennis_power_sphere.txt')
parser.add_argument('--mode', type=str, default='single_action')
args = parser.parse_args()


def load_text_config(fp):
    raw_config = np.loadtxt(fp, dtype='str', delimiter='\n')
    config = {}
    for p in raw_config:
        p = p.split(' = ')
        config[p[0]] = p[1]

        if p[0] in ['source', 'target']:
            _p = p[1]
            _p = _p[1:len(p[1])-1] # remove '[' and ']' from p[1]
            p_list = [a for a in _p.split(', ')]
            config[p[0]] = p_list
    return config


def calc_score(target_dir):

    target_rgbs = sorted(glob.glob(f'{target_dir}/*_rgb.png'))
    target_gts = sorted(glob.glob(f'{target_dir}/*_gt.png'))

    psnr_list = []
    mse_list = []
    ssim_list = []
    for rgb_path, gt_path in zip(target_rgbs, target_gts):
        # (h,w,3), [0,255], uint8
        rgb = np.asarray(Image.open(rgb_path)) / 255.
        gt = np.asarray(Image.open(gt_path)) / 255.

        mse = np.mean((rgb - gt) ** 2)
        psnr = -10. * np.log10(mse)
        ssim = structural_similarity(
            rgb, gt,
            data_range=gt.max()-gt.min(),
            channel_axis=-1)
        
        mse_list.append(mse)
        psnr_list.append(psnr)
        ssim_list.append(ssim)

    avg_mse = np.mean(mse_list)
    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)

    target_rgbs = np.array(target_rgbs)
    target_gts = np.array(target_gts)
    sorted_target_rgbs = target_rgbs[np.argsort(psnr_list)]
    sorted_target_gts = target_gts[np.argsort(psnr_list)]

    return avg_mse, avg_psnr, avg_ssim, sorted_target_rgbs, sorted_target_gts


def run_eval_single_action():
    config = load_text_config(args.config)
    basedir = config['basedir']
    expname = config['expname']

    output_dir = os.path.join(basedir, expname, 'evaluation')
    os.makedirs(output_dir, exist_ok=True)

    testsets = sorted(glob.glob(f'{basedir}/{expname}/testset_*'))
    splits = ['novel_pose', 'novel_view', 'novel_pose_novel_view']

    for split in splits:
        testset_names = []
        mse_results = []
        psnr_results = []
        ssim_results = []
        psnr_sorted_samples = []
        for testset in testsets:
            results = calc_score(target_dir=f'{testset}/{split}')

            testset_names.append(testset.split('/')[-1])
            mse_results.append(results[0])
            psnr_results.append(results[1])
            ssim_results.append(results[2])
            psnr_sorted_samples.append([results[3], results[4]])

        np.save(f'{output_dir}/testset_names_{split}.npy', testset_names)
        np.save(f'{output_dir}/mse_{split}.npy', mse_results)
        np.save(f'{output_dir}/psnr_{split}.npy', psnr_results)
        np.save(f'{output_dir}/ssim_{split}.npy', ssim_results)
        np.save(f'{output_dir}/psnr_sorted_{split}.npy', psnr_sorted_samples)

        for mse, psnr, ssim, testset_name in zip(
            mse_results, psnr_results, ssim_results, testset_names):
            message = f'[{split}] {testset_name} - ' \
                + f'MSE: {mse} - PSNR: {psnr} - SSIM: {ssim}'
            print(message)


def run_eval_multiple_actions():
    config = load_text_config(args.config)
    basedir = config['basedir']
    expname = config['expname']

    output_dir = os.path.join(basedir, expname, 'evaluation')
    os.makedirs(output_dir, exist_ok=True)

    testsets = sorted(glob.glob(f'{basedir}/{expname}/testset_*'))
    splits = ['novel_pose', 'novel_view', 'novel_pose_novel_view']

    source_actions = config['source']
    target_actions = config['target']
    splits = splits + target_actions

    for split in splits:
        testset_names = []
        mse_results = []
        psnr_results = []
        ssim_results = []
        psnr_sorted_samples = []
        for testset in testsets:
            if not os.path.exists(f'{testset}/{split}'):
                print(f'{testset}/{split} does not exist')
                continue
            results = calc_score(target_dir=f'{testset}/{split}')

            testset_names.append(testset.split('/')[-1])
            mse_results.append(results[0])
            psnr_results.append(results[1])
            ssim_results.append(results[2])
            psnr_sorted_samples.append([results[3], results[4]])

        np.save(f'{output_dir}/testset_names_{split}.npy', testset_names)
        np.save(f'{output_dir}/mse_{split}.npy', mse_results)
        np.save(f'{output_dir}/psnr_{split}.npy', psnr_results)
        np.save(f'{output_dir}/ssim_{split}.npy', ssim_results)
        np.save(f'{output_dir}/psnr_sorted_{split}.npy', psnr_sorted_samples)

        for mse, psnr, ssim, testset_name in zip(
            mse_results, psnr_results, ssim_results, testset_names):
            message = f'[{split}] {testset_name} - ' \
                + f'MSE: {mse} - PSNR: {psnr} - SSIM: {ssim}'
            print(message)


if __name__ == '__main__':

    if args.mode == 'single_action':
        run_eval_single_action()
    elif args.mode == 'multiple_actions':
        run_eval_multiple_actions()
