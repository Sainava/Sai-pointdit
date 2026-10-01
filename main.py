# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Derived from JiT (https://github.com/LTH14/JiT), MIT License,
# Copyright (c) 2025 Tianhong Li. See THIRD_PARTY_NOTICES.
import argparse
import datetime
import numpy as np
import os
import time
from pathlib import Path
import sys

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from accelerate import Accelerator
from datetime import timedelta
from accelerate import InitProcessGroupKwargs

import wandb
import signal

from util import misc

import copy
from engine import train_one_epoch, evaluate_img2point

from denoiser import Denoiser

from dataloader.img_depth_intrinsics import ImageDepthIntrinsicsDataset
from dataloader.eval_depth import EvalDepthDataset, MultiEvalDepthDataset, WildImagesDataset
from dataloader.mixed_dataset import MixedImageDepthIntrinsicsDataset

from util.paths import repo_path
from util.resize_posemb import (resize_patch_embed_and_final_layer,
                                        resize_vit_pos_embed)


def get_args_parser():
    parser = argparse.ArgumentParser('PointDiT', add_help=False)

    # architecture
    parser.add_argument('--model', default='PointDiT-B/16', type=str, metavar='MODEL',
                        help='Name of the model to train')
    parser.add_argument('--img_size', default=256, type=int, help='Image size')
    parser.add_argument('--attn_dropout', type=float, default=0.0, help='Attention dropout rate')
    parser.add_argument('--proj_dropout', type=float, default=0.0, help='Projection dropout rate')

    # training
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='Epochs to warm up LR')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size per GPU (effective batch size = batch_size * # GPUs)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='Learning rate (absolute)')
    parser.add_argument('--blr', type=float, default=5e-5, metavar='LR',
                        help='Base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='Minimum LR for cyclic schedulers that hit 0')
    parser.add_argument('--lr_schedule', type=str, default='constant',
                        help='Learning rate schedule')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Weight decay (default: 0.0)')
    parser.add_argument('--ema_decay1', type=float, default=0.9999,
                        help='The first ema to track. Use the first ema for sampling by default.')
    parser.add_argument('--ema_decay2', type=float, default=0.9996,
                        help='The second ema to track')
    parser.add_argument('--P_mean', default=-0.8, type=float)
    parser.add_argument('--P_std', default=0.8, type=float)
    parser.add_argument('--noise_scale', default=1.0, type=float)
    parser.add_argument('--generate_noise_scale', default=0.0, type=float,
                        help='Scale of the Gaussian noise the sampler starts from. 0.0 (default) '
                             'starts from zeros, making inference deterministic.')
    parser.add_argument('--t_eps', default=5e-2, type=float)
    parser.add_argument('--sample_t_eps', default=0., type=float)

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='Starting epoch')
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for faster GPU transfers')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # sampling
    parser.add_argument('--num_sampling_steps', default=3, type=int,
                        help='Number of ODE steps (released setting: 3)')
    parser.add_argument('--eval_freq', type=int, default=40,
                        help='Frequency (in epochs) for evaluation')
    parser.add_argument('--online_eval', action='store_true')
    parser.add_argument('--evaluate_gen', action='store_true')
    parser.add_argument('--gen_bsz', type=int, default=1,
                        help='Evaluation/inference batch size. Keep at 1: the eval dataloader uses '
                             'drop_last=True, so larger values silently discard the tail of each dataset.')

    # dataset
    parser.add_argument('--data_path', default=None, type=str,
                        help='Root directory of a single training dataset. Ignored when '
                             '--dataset_config is given (the YAML carries a data_path per dataset).')

    # checkpointing
    parser.add_argument('--output_dir', default='./output_dir',
                        help='Directory to save outputs (empty for no saving)')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--save_last_freq', type=int, default=1,
                        help='Frequency (in epochs) to save checkpoints')
    parser.add_argument('--save_ckpt_freq', default=100, type=int)
    parser.add_argument('--save_last_steps', type=int, default=0,
                        help='Save last checkpoint every N steps (0 to disable, use epoch-based saving)')
    parser.add_argument('--log_freq', default=100, type=int)
    parser.add_argument('--log_img_stride', default=10, type=int)
    parser.add_argument('--device', default='cuda',
                        help='Device to use for training/testing')

    # distributed training
    parser.add_argument('--world_size', default=1, type=int,
                        help='Number of distributed processes')
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--local-rank', default=0, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='URL used to set up distributed training')
    parser.add_argument('--distributed', action='store_true',
                        help='Use distributed training')

    # new 
    parser.add_argument('--task', default='img2point', type=str,
                        choices=['img2point'],
                        help='Kept for backwards compatibility with released checkpoints; '
                             'PointDiT only supports img2point.')
    parser.add_argument('--save_path_postfix', default=None, type=str)

    # resume from pretrained model, only the model, this is not resume training
    parser.add_argument('--pretrained', default=None, type=str)
    parser.add_argument('--no_strict_load', action='store_true')

    # data normalization
    parser.add_argument('--center_shift_point', action='store_true', default=True,
                        help='Center each GT point map on the mean of its valid points (default: on)')
    parser.add_argument('--no_center_shift_point', action='store_false', dest='center_shift_point')
    parser.add_argument('--center_shift_z_only', action='store_true')
    parser.add_argument('--normalize_point_by_mean', action='store_true', default=True,
                        help='Normalize point maps by the MEAN distance of the shifted points '
                             '(default: on; off = median)')
    parser.add_argument('--no_normalize_point_by_mean', action='store_false', dest='normalize_point_by_mean')
    parser.add_argument('--compute_scale_factor_only_valid', action='store_true',
                        help='Compute the normalization scale from valid points only. NOTE: ignored '
                             'when --dataset_config is used; set it per dataset in the YAML instead.')
    parser.add_argument('--compute_scale_factor_use_std', action='store_true')
    parser.add_argument('--compute_scale_factor_use_percentile', action='store_true')
    parser.add_argument('--remove_outliers', action='store_true')
    parser.add_argument('--outlier_threshold', default=3.0, type=float)
    parser.add_argument('--scale_factor_augment', action='store_true')
    parser.add_argument('--scale_factor_augment_min', default=0.8, type=float)
    parser.add_argument('--scale_factor_augment_max', default=1.2, type=float)
    parser.add_argument('--no_scale_factor', action='store_true')

    # ema
    parser.add_argument('--eval_no_ema', action='store_true')

    parser.add_argument('--num_overfit_samples', default=None, type=int)
    parser.add_argument('--num_dataset_duplicates', default=None, type=int)

    parser.add_argument('--force_zero_t', action='store_true', default=True,
                        help='Rectified sampling: force a fraction of timesteps to exactly 0 during '
                             'training (default: on, fraction --force_zero_t_ratio)')
    parser.add_argument('--no_force_zero_t', action='store_false', dest='force_zero_t')
    parser.add_argument('--force_zero_t_ratio', default=None, type=float)
    # other datasets
    parser.add_argument('--train_subsample_scenes', default=None, type=int)
    parser.add_argument('--dataset_name', default=None, type=str)

    # multi-dataset config
    parser.add_argument('--dataset_config', default=None, type=str,
                        help='Path to YAML config file for multi-dataset training')
    parser.add_argument('--dataset_weights', default=None, type=str,
                        help='Comma-separated dataset weights to override config (e.g., "0.7,0.3" for 70%%/30%% split)')

    # depth intrinsics dataloader
    parser.add_argument('--depth_intrinsics_resize_height', default=None, type=int,
                        help='Resize height before crop for depth intrinsics dataloader')

    # relative point loss (directly optimizes rel_point_metric)
    parser.add_argument('--rel_point_loss_weight', default=0.1, type=float,
                        help='Weight of the relative point loss (normalized by distance from the '
                             'origin). 0.1 in all released models.')

    parser.add_argument('--noise_fill_invalid', action='store_true', help='Fill invalid regions with noise during training')

    parser.add_argument('--exclude_invalid_gt', action='store_true', default=True,
                        help='Exclude invalid GT pixels from the loss (default: on)')
    parser.add_argument('--no_exclude_invalid_gt', action='store_false', dest='exclude_invalid_gt')

    # wandb
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--resume_wandb_id', default=None, type=str,
                        help='Resume logging into an existing W&B run id')
    parser.add_argument('--wandb_project', default='pointdit', type=str)
    parser.add_argument('--wandb_entity', default=None, type=str,
                        help='W&B entity (team/user). Defaults to your configured default entity.')

    # network attention type
    parser.add_argument('--attention_type', default='torch', choices=['torch', 'flash3'], type=str,
                        help="Attention kernel: 'torch' (default, scaled_dot_product_attention) "
                             "or 'flash3' (needs flash-attn; used by the Stage-1 L and H runs). "
                             "Same weights either way — this is only a kernel choice.")

    # gradient clipping
    parser.add_argument('--grad_clip', action='store_true', help='Use gradient clipping')

    # model design

    # feature embedding
    parser.add_argument('--feature_embedding_type', default='dinov3_vitb16', type=str,
                        help='Frozen image encoder used for conditioning: '
                             'dinov3_{vits16,vits16plus,vitb16,vitl16,vith16plus,vit7b16}. '
                             'Must match --model: PointDiT-B/16 -> dinov3_vitb16, '
                             'PointDiT-L/16 -> dinov3_vitl16, '
                             'PointDiT-H/16 -> dinov3_vith16plus')
    parser.add_argument('--dinov3_use_intermediate_layers', action='store_true', default=True,
                        help='Condition on --dinov3_num_intermediate_layers equally-spaced DINOv3 '
                             'layers instead of only the last one (default: on -- required to load '
                             'the released checkpoints)')
    parser.add_argument('--no_dinov3_intermediate_layers', action='store_false',
                        dest='dinov3_use_intermediate_layers')
    parser.add_argument('--dinov3_num_intermediate_layers', type=int, default=4,
                        help='Number of equally-spaced intermediate layers to extract from DINOv3')
    parser.add_argument('--feature_embedding_lr_scale', type=float, default=0.0,
                        help='LR scale for feature embedding (0 = frozen, 0.1 = 10x smaller LR)')

    # misc
    parser.add_argument('--eval_no_save_gen', action='store_true', help='Do not save generated samples during evaluation')
    parser.add_argument('--gen_output_root', default=None, type=str,
                        help='Root directory for evaluation/demo artifacts. Defaults to '
                             '<repo>/generation, i.e. inside the PointDiT directory rather '
                             'than relative to the current working directory.')
    parser.add_argument('--resize_posemb', action='store_true', help='Resize positional embedding when loading pretrained weights')
    parser.add_argument('--resize_patch_embed', type=int, default=0, help='Original patch size to resize from (e.g., 16 when loading patch16 into patch32)')
    parser.add_argument('--persistent_workers', action='store_true', help='Use persistent workers in DataLoader')
    parser.add_argument('--dataloader_debug_timing', action='store_true', help='Print dataloader timing breakdown every 100 samples')
    parser.add_argument('--debug_train_steps', type=int, default=0, help='Exit training loop after N steps (0=disabled, for debugging)')

    # data augmentation
    parser.add_argument('--more_img_aug', action='store_true', default=True,
                        help='Photometric image augmentation (default: on)')
    parser.add_argument('--no_more_img_aug', action='store_false', dest='more_img_aug')
    parser.add_argument('--stronger_img_aug', action='store_true', default=True,
                        help='Stronger photometric augmentation (default: on)')
    parser.add_argument('--no_stronger_img_aug', action='store_false', dest='stronger_img_aug')

    parser.add_argument('--max_depth', default=10.0, type=float, help='Maximum depth value for depth normalization')
    parser.add_argument('--clamp_max_depth', default=None, type=float, help='Clamp maximum depth value to this value')
    parser.add_argument('--handle_sky', action='store_true', help='Handle sky regions specially in depth normalization for img2point task')

    # eval depth dataset (MoGe format)
    parser.add_argument('--eval_depth_dataset', action='store_true',
                        help='Use EvalDepthDataset for evaluation (MoGe format)')
    parser.add_argument('--eval_depth_data_root', default='datasets/eval', type=str,
                        help='Root directory holding the benchmark evaluation datasets')
    parser.add_argument('--eval_depth_dataset_name', default='NYUv2', type=str,
                        help="Name(s) of eval depth dataset: a single name (e.g. NYUv2) or a "
                             "comma-separated list. The paper's 7: "
                             "DIODE,KITTI,NYUv2,ETH3D,HAMMER,iBims-1,Booster")
    parser.add_argument('--eval_depth_max_samples', default=None, type=int,
                        help='Maximum number of samples per dataset to evaluate (for faster testing)')
    parser.add_argument('--eval_depth_resize_height', default=None, type=int,
                        help='Resize shortest side to this height before crop. Useful for evaluating high-res images at lower resolution')
    parser.add_argument('--eval_depth_original_resolution', action='store_true',
                        help='Evaluate at original resolution (resize input to nearest div-16, predict, resize back)')
    parser.add_argument('--eval_depth_max_resolution', default=1024, type=int,
                        help='Max resolution for longer side in original_resolution mode (default: 1024)')
    parser.add_argument('--eval_depth_max_height', default=None, type=int,
                        help='Max height for original resolution eval. If set with max_width, uses separate H/W constraints instead of max_resolution')
    parser.add_argument('--eval_depth_max_width', default=None, type=int,
                        help='Max width for original resolution eval. If set with max_height, uses separate H/W constraints instead of max_resolution')
    parser.add_argument('--eval_save_per_sample_metrics', action='store_true',
                        help='Save per-sample metrics to JSON file for method comparison')
    parser.add_argument('--eval_intermediate_boundary', action='store_true',
                        help='Compute SI boundary F1 for each intermediate diffusion step (requires --eval_save_per_sample_metrics)')
    parser.add_argument('--eval_diode_split_indoor_outdoor', action='store_true',
                        help='Split DIODE metrics into indoor and outdoor scenes (also keeps full DIODE metrics)')
    parser.add_argument('--eval_depth_range_metrics', action='store_true',
                        help='Compute metrics stratified by depth range (near/mid/far based on GT depth percentiles)')
    parser.add_argument('--eval_boundary_datasets', default=None, type=str,
                        help='Comma-separated list of dataset names to compute SI_boundary_F1 on (e.g., "NYUv2,KITTI")')
    parser.add_argument('--eval_num_tokens', default=0, type=int,
                        help='Number of tokens for eval resizing (MoGe-style). 0 = use existing logic. '
                             'When > 0, resize input to match token count while preserving aspect ratio.')
    parser.add_argument('--resize_input_nearest_res', action='store_true',
                        help='Resize input to nearest resolution divisible by 16 (round to nearest, not floor). '
                             'Optionally combined with --eval_num_tokens or --eval_min_tokens/--eval_max_tokens.')
    parser.add_argument('--eval_min_tokens', default=0, type=int,
                        help='Minimum tokens for resize_input_nearest_res mode. 0 = no minimum.')
    parser.add_argument('--eval_max_tokens', default=0, type=int,
                        help='Maximum tokens for resize_input_nearest_res mode. 0 = no maximum.')
    parser.add_argument('--sky_loss_weight', default=0.0, type=float, help='Weight for sky region loss if handle_sky is enabled')
    parser.add_argument('--use_sky_dome', action='store_true', help='Use sky dome information if available in img2point task')
    parser.add_argument('--sky_far_plane_value', default=3.0, type=float, help='Far plane value for sky regions when using sky dome')

    # remove sky at eval/inference (for models trained with --handle_sky)
    parser.add_argument('--remove_sky', action='store_true', default=True,
                        help='Remove predicted sky points at eval/inference (default: on). Sky is detected on the '
                             'RAW model output (normalized space, before the radial-log inverse and scale/shift '
                             'alignment) by thresholding the per-point value against --remove_sky_threshold. Only '
                             'meaningful for models trained with sky handling (handle_sky, typically set per-dataset '
                             'in the dataset config), which places sky at norm ~sky_far_plane_value (default 3.0); '
                             'pass --no_remove_sky for models trained without it, such as the Stage-1 SceneNet '
                             'checkpoints. Affects saved point clouds and depth panels only, never the metrics.')
    parser.add_argument('--no_remove_sky', action='store_false', dest='remove_sky')
    parser.add_argument('--remove_sky_threshold', default=2.9, type=float,
                        help='Threshold (in normalized model-output space) for sky removal. Points whose raw-output '
                             'value (see --remove_sky_metric) exceeds this are treated as sky and removed. Should be '
                             'slightly below --sky_far_plane_value (default 2.9 for far plane 3.0).')
    parser.add_argument('--remove_sky_metric', default='norm', type=str, choices=['norm', 'depth'],
                        help="Quantity thresholded on the raw model output for sky removal: 'norm' = Euclidean "
                             "distance from origin (default, robust for --use_sky_dome), 'depth' = z coordinate.")

    # dataset split
    parser.add_argument('--split', default=None, type=str,
                        choices=['train', 'val', 'test', 'all'],
                        help='Dataset split to use. "all" uses both train and val for training.')

    # save separate intermediate PLYs
    parser.add_argument('--remove_depth_edge', action='store_true', default=True,
                        help='Remove noisy depth-edge pixels before saving PLY point clouds (default: on). '
                             'Affects saved point clouds only, never the metrics.')
    parser.add_argument('--no_remove_depth_edge', action='store_false', dest='remove_depth_edge')
    parser.add_argument('--depth_edge_rtol', type=float, default=0.04,
                        help='Relative tolerance for depth edge detection (default: 0.04, matching MoGe)')
    parser.add_argument('--save_separate_intermediate_plys', action='store_true',
                        help='Save intermediate diffusion steps as separate PLY files instead of combined')
    parser.add_argument('--save_separate_intermediate_depths', action='store_true',
                        help='Save intermediate diffusion steps as separate depth PNG files')
    parser.add_argument('--save_resolution_in_filename', action='store_true',
                        help='Append _HxW to saved input image filenames to highlight inference resolution')
    parser.add_argument('--save_raw_depth_npy', action='store_true',
                        help='Save raw depth/pointcloud arrays as .npy files for cross-run comparison')
    parser.add_argument('--viz_unaligned_depth', action='store_true',
                        help='Also visualize unaligned predicted depth on benchmark datasets (using shifted_depth normalization)')
    parser.add_argument('--depth_colormap', default='plasma', type=str,
                        help='Matplotlib colormap for depth visualization (e.g. plasma, magma, inferno, turbo)')

    # Wild images inference (no ground truth)
    parser.add_argument('--eval_wild_images', action='store_true',
                        help='Run inference on arbitrary images without ground truth (e.g. DA-2K)')
    parser.add_argument('--eval_wild_images_dir', default=None, type=str,
                        help='Directory containing images for wild inference')
    parser.add_argument('--eval_wild_images_name', default=None, type=str,
                        help='Name of the output subdirectory for wild inference. Defaults to the '
                             'folder name of --eval_wild_images_dir, so images in assets/demo land '
                             'in <generation>/<checkpoint>/demo/')
    parser.add_argument('--eval_wild_target_tokens', default=0, type=int,
                        help='Token budget for full-frame wild inference. The whole image is '
                             'resized (aspect ratio preserved) to the patch-aligned resolution '
                             'whose token count is closest to this budget, the prediction runs '
                             'there, then it is resized back to the native resolution for saving. '
                             '0 (the default) derives the budget from --img_size, so the model '
                             'sees the same number of tokens it was trained on. Pass '
                             '--eval_depth_resize_height instead to use the legacy '
                             'resize-shortest-side-then-center-crop path.')

    return parser


def main(args):
    # 1. Initialize Accelerator. Accelerate drives every run, single process or not, so the
    # device comes from it. cpu=True is how --device cpu still forces CPU on a GPU host;
    # otherwise the accelerator picks CUDA when it is available.
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=1800)) # 30 minutes
    accelerator = Accelerator(cpu=(args.device == 'cpu'), kwargs_handlers=[kwargs])
    device = accelerator.device
    global_rank = accelerator.process_index
    print = accelerator.print

    # remove_sky: thresholding the z coordinate ('depth') under-detects sky in sky-dome
    # mode, where sky is encoded as direction*sky_far_plane_value and z can be small at
    # the horizon. Only the Euclidean norm is reliable there.
    if getattr(args, 'remove_sky', False) and args.remove_sky_metric == 'depth' and args.use_sky_dome:
        print("[Warning] --remove_sky_metric depth is unreliable with --use_sky_dome "
              "(dome sky has small z near the horizon); using 'norm' instead.")
        args.remove_sky_metric = 'norm'

    # Set seeds for reproducibility
    seed = args.seed + global_rank
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # Set up TensorBoard logging (only on main process)
    is_main_process = accelerator.is_main_process

    if is_main_process and args.output_dir is not None and not args.evaluate_gen:
        os.makedirs(args.output_dir, exist_ok=True)
        if args.wandb:
            # Authenticate beforehand with `wandb login` or the WANDB_API_KEY env var.
            wandb.login()

            def signal_handler(sig, frame):
                print('You pressed CTRL+C!')
                # Adjust exit code as needed
                wandb.finish(exit_code=255, quiet=True)
                # Additional cleanup here
                sys.exit(0)

            signal.signal(signal.SIGINT, signal_handler)

            wandb_extra_kwargs = {}

            if args.resume_wandb_id:
                wandb_extra_kwargs.update({
                    'id': args.resume_wandb_id,
                    'resume': 'must'
                })

            wandb.init(entity=args.wandb_entity,
                       project=args.wandb_project,
                       name=os.path.basename(args.output_dir),
                       dir=args.output_dir,
                       **wandb_extra_kwargs,
                       )
            log_writer = None
        else:
            log_writer = SummaryWriter(log_dir=args.output_dir)
    else:
        log_writer = None

    # log slurm id to wandb
    if is_main_process and args.wandb:
        slurm_job_id = os.environ.get('SLURM_JOB_ID')
        if slurm_job_id is not None:
            print('slurm id:', slurm_job_id)

    if not args.evaluate_gen:
        print('Job directory:', os.path.dirname(os.path.realpath(__file__)))
        print("Arguments:\n{}".format(args).replace(', ', ',\n'))

    # train loader
    if not args.evaluate_gen:
        # skip train loader for evaluation only since data loading can be slow
        # TODO: maybe consider random hflip augmentation
        crop_size_for_dataset = args.img_size

        if args.dataset_config:
            # Use MixedImageDepthIntrinsicsDataset (multi-dataset training)
            print(f"Using MixedImageDepthIntrinsicsDataset with config: {args.dataset_config}")

            # Parse dataset weights if provided
            weight_overrides = None
            if args.dataset_weights:
                try:
                    weight_overrides = [float(w.strip()) for w in args.dataset_weights.split(',')]
                    print(f"Using CLI weight overrides: {weight_overrides}")
                except ValueError as e:
                    raise ValueError(
                        f"Invalid --dataset_weights format: '{args.dataset_weights}'. "
                        f"Expected comma-separated floats (e.g., '0.7,0.3')"
                    ) from e

            dataset_train = MixedImageDepthIntrinsicsDataset(
                config_path=args.dataset_config,
                split=args.split if args.split else 'train',
                crop_size=crop_size_for_dataset,
                center_shift=args.center_shift_point,
                center_shift_z_only=args.center_shift_z_only,
                normalize_by_mean=args.normalize_point_by_mean,
                num_overfit_samples=args.num_overfit_samples,
                num_dataset_duplicates=args.num_dataset_duplicates,
                subsample_scenes=args.train_subsample_scenes,
                more_img_aug=args.more_img_aug,
                stronger_img_aug=args.stronger_img_aug,
                debug_timing=args.dataloader_debug_timing,
                weight_overrides=weight_overrides,
            )
        else:
            # Use ImageDepthIntrinsicsDataset (single dataset, raw depth + intrinsics)
            print(f"Using ImageDepthIntrinsicsDataset for {args.dataset_name or 'scenenet'}")

            dataset_name = args.dataset_name or 'scenenet'

            dataset_train = ImageDepthIntrinsicsDataset(
                root_dir=args.data_path,
                dataset_name=dataset_name,
                split=args.split if args.split else 'train',
                num_test_scenes=None,
                test_scenes_file=None,  # ImageDepthIntrinsicsDataset handles test split internally
                samples_per_scene=None,
                crop_size=crop_size_for_dataset,
                resize_height=args.depth_intrinsics_resize_height,
                center_shift=args.center_shift_point,
                center_shift_z_only=args.center_shift_z_only,
                normalize_by_mean=args.normalize_point_by_mean,
                num_overfit_samples=args.num_overfit_samples,
                num_dataset_duplicates=args.num_dataset_duplicates,
                max_depth=args.max_depth,
                subsample_scenes=args.train_subsample_scenes,
                more_img_aug=args.more_img_aug,
                stronger_img_aug=args.stronger_img_aug,
                compute_scale_factor_only_valid=args.compute_scale_factor_only_valid,
                compute_scale_factor_use_std=args.compute_scale_factor_use_std,
                compute_scale_factor_use_percentile=args.compute_scale_factor_use_percentile,
                clamp_max_depth=args.clamp_max_depth,
                handle_sky=args.handle_sky,
                use_sky_dome=args.use_sky_dome,
                sky_far_plane_value=args.sky_far_plane_value,
                debug_timing=args.dataloader_debug_timing,
                remove_outliers=args.remove_outliers,
                outlier_threshold=args.outlier_threshold,
                scale_factor_augment=args.scale_factor_augment,
                scale_factor_augment_range=(args.scale_factor_augment_min, args.scale_factor_augment_max),
                no_scale_factor=args.no_scale_factor,
            )

        print('Train samples:', len(dataset_train))

        # Check if we need WeightedRandomSampler (for mixed datasets)
        use_weighted_sampler = isinstance(dataset_train, MixedImageDepthIntrinsicsDataset)

        # Sampler is created by the accelerator
        # For mixed datasets, use WeightedRandomSampler
        if use_weighted_sampler:
            print("[MixedDataset] Using WeightedRandomSampler for dataset mixing")
            # Create weighted sampler
            sampler_train = dataset_train.create_weighted_sampler()
            shuffle = False  # Don't shuffle when using custom sampler
        else:
            sampler_train = None
            shuffle = True

        data_loader_train = torch.utils.data.DataLoader(
            dataset_train,
            sampler=sampler_train,
            shuffle=shuffle if sampler_train is None else False,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
            persistent_workers=args.persistent_workers,
            prefetch_factor=4 if args.num_workers > 0 else None,
        )

    # test dataloader
    if args.online_eval or args.evaluate_gen:
        # Enforce batch_size=1 whenever inputs keep their native size, since images then
        # differ in size from one another and the default collate cannot stack them.
        use_original_res = getattr(args, 'eval_depth_original_resolution', False)
        wild_full_frame = args.eval_wild_images and args.eval_depth_resize_height is None
        if use_original_res or wild_full_frame:
            if args.batch_size != 1:
                cause = '--eval_depth_original_resolution' if use_original_res else 'full-frame wild inference'
                print(f"[Warning] {cause} requires batch_size=1, overriding batch_size={args.batch_size} to 1")
                args.batch_size = 1

        if args.eval_depth_dataset:
            # Use EvalDepthDataset for evaluation (MoGe format)
            # Parse dataset names (can be comma-separated)
            dataset_names = [name.strip() for name in args.eval_depth_dataset_name.split(',')]

            # Use None for crop_size and resize_height in original resolution mode
            use_original_res = getattr(args, 'eval_depth_original_resolution', False)
            eval_crop_size = None if use_original_res else args.img_size
            eval_resize_height = None if use_original_res else args.eval_depth_resize_height

            if len(dataset_names) == 1:
                # Single dataset
                dataset_test = EvalDepthDataset(
                    data_root=args.eval_depth_data_root,
                    dataset_name=dataset_names[0],
                    crop_size=eval_crop_size,
                    max_samples=args.eval_depth_max_samples,
                    resize_height=eval_resize_height,
                )
            else:
                # Multiple datasets
                dataset_test = MultiEvalDepthDataset(
                    data_root=args.eval_depth_data_root,
                    dataset_names=dataset_names,
                    crop_size=eval_crop_size,
                    max_samples=args.eval_depth_max_samples,
                    resize_height=eval_resize_height,
                )
        elif args.eval_wild_images:
            assert args.eval_wild_images_dir is not None, "--eval_wild_images_dir must be set when using --eval_wild_images"
            # Full frame by default: the dataset hands over the native image untouched and
            # the engine resizes it to the training token count, predicts, then resizes the
            # prediction back to native. --eval_depth_resize_height opts into the legacy
            # resize-shortest-side-then-center-crop path, which throws away everything
            # outside the central crop.
            if use_original_res or args.eval_depth_resize_height is None:
                eval_crop_size = None
                eval_resize_height = None
            else:
                eval_crop_size = args.img_size
                eval_resize_height = args.eval_depth_resize_height
            # Name the output subdirectory after the image folder, so the results are easy to
            # trace back to their input. normpath first, or a trailing slash yields ''.
            if args.eval_wild_images_name is None:
                args.eval_wild_images_name = os.path.basename(
                    os.path.normpath(args.eval_wild_images_dir)) or 'wild'
            dataset_test = WildImagesDataset(
                images_dir=args.eval_wild_images_dir,
                dataset_name=args.eval_wild_images_name,
                crop_size=eval_crop_size,
                resize_height=eval_resize_height,
                max_samples=args.eval_depth_max_samples,
            )
        else:
            raise ValueError(
                'Evaluation requires either --eval_depth_dataset (benchmark datasets in MoGe '
                'format) or --eval_wild_images (inference on arbitrary images).'
            )

        print('Test dataset:', dataset_test)

        # Sampler is created by the accelerator
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test,
            shuffle=False,
            batch_size=args.gen_bsz,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
            persistent_workers=args.persistent_workers,
        )
    else:
        data_loader_test = None

    torch._dynamo.config.cache_size_limit = 128
    torch._dynamo.config.optimize_ddp = False

    # Create denoiser (the PointDiT model)
    model = Denoiser(args)

    if accelerator.is_main_process and args.wandb:
        wandb.log({'model': print(model)})

    model.to(device)

    # Calculate effective batch size based on accelerator's count
    eff_batch_size = args.batch_size * accelerator.num_processes

    if args.lr is None:  # only base_lr (blr) is specified
        args.lr = args.blr * eff_batch_size / 256

    if not args.evaluate_gen:
        print("Base lr: {:.2e}".format(args.lr * 256 / eff_batch_size))
        print("Actual lr: {:.2e}".format(args.lr))
        print("Effective batch size: %d" % eff_batch_size)

    # Prepare for distributed training
    param_groups = misc.add_weight_decay(model, args.weight_decay,
                                         feature_embedding_lr_scale=args.feature_embedding_lr_scale)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    if not args.evaluate_gen:
        print(optimizer)

    # This handles DDP wrapping and Sampler creation automatically
    if args.evaluate_gen:
        model, optimizer, data_loader_test = accelerator.prepare(
            model, optimizer, data_loader_test
        )
    else:
        model, optimizer, data_loader_train, data_loader_test = accelerator.prepare(
            model, optimizer, data_loader_train, data_loader_test
        )
    model_without_ddp = accelerator.unwrap_model(model)

    # Resume from checkpoint if provided
    # Wait for all processes to be ready
    accelerator.wait_for_everyone()

    # resume pretrained feature embedding
    dinov3_weights_loaded = False
    if args.feature_embedding_type.startswith('dinov3'):
        vit_type = args.feature_embedding_type.split('_')[-1]
        assert vit_type in ['vits16', 'vits16plus', 'vitb16', 'vitl16', 'vith16plus', 'vit7b16'], f'ViT type {vit_type} not supported for DINOv3 feature embedding'

        # load pretrained DINOv3 weights
        sha_dict = {
            'vits16': '08c60483',
            'vits16plus': '4057cbaa',
            'vitb16': '73cec8be',
            'vitl16': '8aa4cbdd',
            'vith16plus': '7c1da9a5',
            'vit7b16': 'a955f4ea',
        }
        sha = sha_dict[vit_type]
        dinov3_weights_dir = os.environ.get(
            'DINOV3_WEIGHTS_DIR', repo_path('pretrained/dinov3'))
        ckpt_path = os.path.join(
            dinov3_weights_dir, f'dinov3_{vit_type}_pretrain_lvd1689m-{sha}.pth')

        if os.path.exists(ckpt_path):
            ckpts = torch.load(ckpt_path, map_location='cpu')
            model_without_ddp.net.y_embedder.load_state_dict(ckpts, strict=True)
            dinov3_weights_loaded = True
            print(f'Loaded pretrained DINOv3 {vit_type} from {ckpt_path}')
        elif args.pretrained is None:
            print(f'[DINOv3] {ckpt_path} not found -> encoder left randomly initialised. '
                  f'Download the gated LVD-1689M weights to train from scratch.')

    else:
        raise NotImplementedError(f'Feature embedding type {args.feature_embedding_type} not supported')

    if not args.evaluate_gen:
        print(model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters: {:.2f}M".format(n_params / 1e6))
    n_params_non_train = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print("Number of non-trainable parameters: {:.2f}M".format(n_params_non_train / 1e6))

    # Monotonic training step counter (persisted across resumes so logging and
    # sub-epoch checkpoint intervals stay continuous regardless of dataset length).
    global_step = 0

    checkpoint = None
    if args.resume:
        # Try the latest checkpoint, then the rolling backup. A corrupt/truncated
        # file (e.g. job killed mid-write) is skipped instead of crashing, and if
        # nothing loads we fall through to the --pretrained branch below.
        for cand in [os.path.join(args.output_dir, "checkpoint-last.pth"),
                     os.path.join(args.output_dir, "checkpoint-last-prev.pth")]:
            if not os.path.exists(cand):
                continue
            try:
                checkpoint = torch.load(cand, map_location='cpu', weights_only=False)
                print("Resumed checkpoint from", cand)
                break
            except Exception as e:
                print(f"WARNING: failed to load {cand} ({e}); trying fallback")
        if checkpoint is None:
            print("No valid resume checkpoint; falling back to --pretrained / scratch")
    if checkpoint is not None:
        model_without_ddp.load_state_dict(checkpoint['model'])

        ema_state_dict1 = checkpoint['model_ema1']
        ema_state_dict2 = checkpoint['model_ema2']
        model_without_ddp.ema_params1 = [ema_state_dict1[name].to(device) for name, _ in model_without_ddp.named_parameters()]
        model_without_ddp.ema_params2 = [ema_state_dict2[name].to(device) for name, _ in model_without_ddp.named_parameters()]
        print("Resumed checkpoint from", args.output_dir)

        if 'optimizer' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            # Sub-epoch checkpoints store resume_epoch = in-progress epoch (re-run
            # it from the top); end-of-epoch checkpoints resume at epoch + 1.
            if checkpoint.get('resume_epoch') is not None:
                args.start_epoch = checkpoint['resume_epoch']
            else:
                args.start_epoch = checkpoint['epoch'] + 1
            if checkpoint.get('global_step') is not None:
                global_step = checkpoint['global_step']
            print("Loaded optimizer & scaler state! Resuming at epoch {}, global_step {}".format(
                args.start_epoch, global_step))
        del checkpoint
    elif args.pretrained:
        assert os.path.exists(args.pretrained)
        checkpoint = torch.load(args.pretrained, map_location='cpu', weights_only=False)

        # The released checkpoints ship without the frozen DINOv3 encoder: those weights are
        # gated and cannot be redistributed
        dinov3_stripped = checkpoint.get('dinov3_stripped')
        if dinov3_stripped is not None and not dinov3_weights_loaded:
            raise FileNotFoundError(
                f'"{args.pretrained}" was released without the frozen DINOv3 '
                f'{dinov3_stripped.get("variant", vit_type)} encoder, and the encoder weights '
                f'were not found at "{ckpt_path}".\n'
                f'Request access at https://github.com/facebookresearch/dinov3, then place '
                f'{dinov3_stripped.get("upstream_filename", "the weights")} in '
                f'pretrained/dinov3/, or set DINOV3_WEIGHTS_DIR to the directory holding it.')

        # A checkpoint carries up to two EMA copies. The released ones keep only 'model_ema1'
        ema_keys = [k for k in ('model_ema1', 'model_ema2') if k in checkpoint]
        has_ema = 'model_ema1' in checkpoint

        if args.resize_posemb:
            # Update the checkpoint dict (key matches your error: 'net.pos_embed')
            patch_size = int(args.model.split('/')[-1])
            new_num_patches = (args.img_size // patch_size) ** 2
            for key in ['model'] + ema_keys:
                checkpoint[key] = resize_vit_pos_embed(checkpoint[key], new_num_patches=new_num_patches, key='net.pos_embed')

            # dino feature embedding pos emb
            if 'net.pos_embed_y' in checkpoint['model']:
                # for img2point task with separate pos embed for image and point tokens
                for key in ['model'] + ema_keys:
                    checkpoint[key] = resize_vit_pos_embed(checkpoint[key], new_num_patches=new_num_patches, key='net.pos_embed_y')

        if args.resize_patch_embed > 0:
            new_patch_size = int(args.model.split('/')[-1])
            for key in ['model'] + ema_keys:
                checkpoint[key] = resize_patch_embed_and_final_layer(
                    checkpoint[key], args.resize_patch_embed, new_patch_size, out_channels=3)

        if dinov3_stripped is not None and not args.no_strict_load:
            # Tolerate exactly the encoder keys the release removed -- already loaded above --
            # and keep every other mismatch fatal, which a blanket --no_strict_load would not.
            prefix = dinov3_stripped.get('prefix', 'net.y_embedder.')
            incompatible = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
            missing = [k for k in incompatible.missing_keys if not k.startswith(prefix)]
            if missing or incompatible.unexpected_keys:
                raise RuntimeError(
                    'Error(s) in loading state_dict for {}:\n\tMissing keys: {}\n\t'
                    'Unexpected keys: {}'.format(type(model_without_ddp).__name__,
                                                 missing, incompatible.unexpected_keys))
        else:
            model_without_ddp.load_state_dict(checkpoint['model'], strict=not args.no_strict_load)

        if args.evaluate_gen and 'epoch' in checkpoint:
            args.start_epoch = checkpoint['epoch']

        if not has_ema:
            # Nothing to restore: the weights in 'model' are already the ones we want to
            # evaluate. Leaving ema_params1/2 as None makes the engine skip the EMA swap.
            model_without_ddp.ema_params1 = None
            model_without_ddp.ema_params2 = None
            print('No EMA weights in checkpoint ({}); evaluating the "model" weights directly.'
                  .format(checkpoint.get('extracted_from', 'slim checkpoint')))
        elif args.evaluate_gen:
            # Inference only: write the EMA weights straight into the model instead of keeping
            # GPU copies for the engine to swap in (and back) -- those copies, of the frozen
            # DINOv3 encoder too, take ~4x the model size and do not fit on a 16 GB GPU.
            # Same result as the swap below: parameters missing from 'model_ema1' (the
            # stripped encoder) keep the weights loaded above.
            if not args.eval_no_ema:
                param_names = {name for name, _ in model_without_ddp.named_parameters()}
                ema_state_dict1 = {k: v for k, v in checkpoint['model_ema1'].items() if k in param_names}
                model_without_ddp.load_state_dict(ema_state_dict1, strict=False)
                print('Loaded EMA weights in place for evaluation')
            model_without_ddp.ema_params1 = None
            model_without_ddp.ema_params2 = None
        else:
            ema_params1 = copy.deepcopy(list(model_without_ddp.parameters()))
            ema_params2 = copy.deepcopy(list(model_without_ddp.parameters()))

            print('resume pretrained ema')
            ema_state_dict1 = checkpoint['model_ema1']
            # Only ema_params1 is read at evaluation time, so the released checkpoints drop the
            # second copy; the partial resume below then falls back to the loaded weights for it.
            ema_state_dict2 = checkpoint.get('model_ema2', {})

            # partial resume
            model_without_ddp.ema_params1 = []
            model_without_ddp.ema_params2 = []
            for i, (name, _ )in enumerate(model_without_ddp.named_parameters()):
                if name in ema_state_dict1:
                    model_without_ddp.ema_params1.append(ema_state_dict1[name].to(device))
                else:
                    model_without_ddp.ema_params1.append(ema_params1[i])

                if name in ema_state_dict2:
                    model_without_ddp.ema_params2.append(ema_state_dict2[name].to(device))
                else:
                    model_without_ddp.ema_params2.append(ema_params2[i])

        print('Load pretrained model from', args.pretrained)

    else:
        model_without_ddp.ema_params1 = copy.deepcopy(list(model_without_ddp.parameters()))
        model_without_ddp.ema_params2 = copy.deepcopy(list(model_without_ddp.parameters()))
        print("Training from scratch")

    # Barrier to ensure all processes have finished loading before starting training
    accelerator.wait_for_everyone()

    # Evaluate generation
    if args.evaluate_gen:
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            with torch.no_grad():
                # no log to tb if eval only
                evaluate_img2point(model_without_ddp, args, 0, data_loader_test, log_writer=None)
        return
    
    # Training loop
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        global_step = train_one_epoch(model, model_without_ddp, data_loader_train, optimizer, device, epoch, log_writer=log_writer, args=args, accelerator=accelerator, global_step=global_step)

        # Save checkpoint periodically
        if epoch % args.save_last_freq == 0 or epoch + 1 == args.epochs:

            accelerator.wait_for_everyone()

            misc.save_model(
                args=args,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                epoch=epoch,
                epoch_name="last",
                global_step=global_step
            )

        if epoch % args.save_ckpt_freq == 0 and epoch > 0:

            accelerator.wait_for_everyone()

            misc.save_model(
                args=args,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                epoch=epoch,
                global_step=global_step
            )

        # Perform online evaluation at specified intervals
        if args.online_eval and (epoch % args.eval_freq == 0 or epoch + 1 == args.epochs):
            torch.cuda.empty_cache()
            with torch.no_grad():
                evaluate_img2point(model_without_ddp, args, epoch, data_loader_test, log_writer=log_writer, accelerator=accelerator)
            torch.cuda.empty_cache()

        if accelerator.is_main_process and log_writer is not None:
            log_writer.flush()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time:', total_time_str)


if __name__ == '__main__':
    args = get_args_parser().parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
