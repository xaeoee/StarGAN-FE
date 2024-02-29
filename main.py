import os
import argparse
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn


def str2bool(v):
    return v.lower() in ('true')

def main(config):
    # For fast training.
    cudnn.benchmark = True

    # Create directories if not exist.
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)

    # Data loader.
    data_loader = get_loader(config.image_dir, config.crop_size, config.image_size, config.batch_size, config.mode, config.num_workers)

    # Solver for training and testing StarGAN.
    solver = Solver(data_loader, config)

    if config.mode == 'train':
        solver.train()   

    elif config.mode == 'test':
        solver.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--c_dim', type=int, default=5, help='dimension of domain labels (1st dataset)')
    parser.add_argument('--crop_size', type=int, default=448, help='crop size for the RaFD dataset')
    parser.add_argument('--image_size', type=int, default=256, help='image resolution')
    parser.add_argument('--g_conv_dim', type=int, default=64, help='number of conv filters in the first layer of G')
    parser.add_argument('--d_conv_dim', type=int, default=64, help='number of conv filters in the first layer of D')
    parser.add_argument('--g_repeat_num', type=int, default=6, help='number of residual blocks in G')
    parser.add_argument('--d_repeat_num', type=int, default=6, help='number of strided conv layers in D')
    parser.add_argument('--lambda_cls', type=float, default=1, help='weight for domain classification loss')
    parser.add_argument('--lambda_rec', type=float, default=10, help='weight for reconstruction loss')
    parser.add_argument('--lambda_gp', type=float, default=10, help='weight for gradient penalty')
    
    # Training configuration.
    parser.add_argument('--batch_size', type=int, default=16, help='mini-batch size')
    parser.add_argument('--num_iters', type=int, default=20000, help='number of total iterations for training D') #400000
    parser.add_argument('--num_iters_decay', type=int, default=5000, help='number of iterations for decaying lr')
    parser.add_argument('--g_lr', type=float, default=0.0002, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.0004, help='learning rate for D')
    parser.add_argument('--n_critic', type=int, default=5, help='number of D updates per each G update')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--resume_iters', type=int, default=0, help='resume training from this step')

    # Test configuration.
    parser.add_argument('--test_iters', type=int, default=300000, help='test model from this step')
    parser.add_argument('--test_mode', type=str, default='origin_person', help='test mode')

    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=False)

    # Directories.
    parser.add_argument('--image_dir', type=str, default='data/train')
    parser.add_argument('--log_dir', type=str, default='data/logs')
    parser.add_argument('--model_save_dir', type=str, default='data/models')
    parser.add_argument('--sample_dir', type=str, default='data/samples')
    parser.add_argument('--sample_label_dir', type=str, default='data/train')
    parser.add_argument('--result_dir', type=str, default='data/results')
    parser.add_argument('--image_path', type=str, default=None)
    
    # FID Score
    parser.add_argument('--fid_sample_size', type=int, default=160, help='fid_sample_size')
    parser.add_argument('--fid_sample_image_size', type=int, default=256, help='fid_sample_image_size')

    # Step size.
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=1000)
    parser.add_argument('--model_save_step', type=int, default=10000)
    parser.add_argument('--lr_update_step', type=int, default=1000)

    config = parser.parse_args()
    print(config)
    main(config)
    
'''
python main.py --mode train --crop_size 256 --image_size 256 --c_dim 6 --image_dir data/train --sample_dir stargan_new_6_leaky/samples --log_dir stargan_new_6_leaky/logs --model_save_dir stargan_new_6_leaky/models --result_dir stargan_new_6_leaky/results

'''