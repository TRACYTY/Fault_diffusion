import os
import torch
import argparse
import numpy as np

from engine.logger import Logger
from engine.solver import Trainer
from Data.build_dataloader import build_dataloader, build_dataloader_cond
from Models.interpretable_diffusion.model_utils import unnormalize_to_zero_to_one
from Utils.io_utils import load_yaml_config, seed_everything, merge_opts_to_config, instantiate_from_config

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Training Script')
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--config_file', type=str, default=None, 
                        help='path of config file')
    parser.add_argument('--output', type=str, default='OUTPUT', 
                        help='directory to save the results')
    parser.add_argument('--tensorboard', action='store_true', 
                        help='use tensorboard for logging')
    parser.add_argument('--cudnn_deterministic', action='store_true', default=False,
                        help='set cudnn.deterministic True')
    parser.add_argument('--seed', type=int, default=12345, 
                        help='seed for initializing training.')
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU id to use. If given, only the specific gpu will be used, and ddp will be disabled')
    parser.add_argument('--finetune', action='store_true', default=False, help='Finetune with fault data.')
    parser.add_argument('--pretrained_path', type=str, default=None, help='Path to pretrained weights.')
    parser.add_argument('--finetune_path', type=str, default=None, help='Path to finetuned weights for inference.')
    parser.add_argument('--fault_data', type=str, default=None, help='Path to fault data for finetuning.')
    parser.add_argument('--train', action='store_true', default=False, help='Train or Test.')
    parser.add_argument('--sample', type=int, default=0, 
                        choices=[0, 1], help='Condition or Uncondition.')
    parser.add_argument('--mode', type=str, default='infill',
                        help='Infilling or Forecasting.')
    parser.add_argument('--milestone', type=int, default=3)
    parser.add_argument('--missing_ratio', type=float, default=0., help='Ratio of Missing Values.')
    parser.add_argument('--pred_len', type=int, default=0, help='Length of Predictions.')
    parser.add_argument('opts', help='Modify config options using the command-line',
                        default=None, nargs=argparse.REMAINDER)  
    parser.add_argument('--size_every', type=int, default=2001, help='Batch size for each sampling iteration')
    args = parser.parse_args()
    args.save_dir = os.path.join(args.output, f'{args.name}')
    return args

def main():
    args = parse_args()
    if args.seed is not None:
        seed_everything(args.seed)
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
    
    config = load_yaml_config(args.config_file)
    config = merge_opts_to_config(config, args.opts)
    logger = Logger(args)
    logger.save_config(config)

    model = instantiate_from_config(config['model']).cuda()
    if args.finetune and args.fault_data:
        config['dataloader']['train_dataset']['params']['data_root'] = args.fault_data
    if args.sample == 1 and args.mode in ['infill', 'predict']:
        test_dataloader_info = build_dataloader_cond(config, args)
    dataloader_info = build_dataloader(config, args)

    # 仅在非预训练模式下检查
    if not args.train:
        if (args.pretrained_path is not None and 'finetuned' in args.pretrained_path) or 'fault_model' in args.name:
            model.freeze_parameters(freeze_encoder=True, freeze_decoder=False)
            model.enable_finetuning()

    trainer = Trainer(config=config, args=args, model=model, dataloader=dataloader_info, logger=logger)

    if args.train:
        trainer.train()
    elif args.finetune:
        if args.pretrained_path is None:
            args.pretrained_path = os.path.join(args.save_dir, f'checkpoint_{args.milestone}.pth')
        if not os.path.exists(args.pretrained_path):
            raise FileNotFoundError(f"Pretrained weights not found at {args.pretrained_path}")
        # 显式启用微调模式
        model.freeze_parameters(freeze_encoder=True, freeze_decoder=False)
        model.enable_finetuning()
        trainer.load(args.pretrained_path, verbose=True)
        print(f"Loaded pretrained weights from {args.pretrained_path}")
        finetune_steps = config.get('finetune_steps', 1000)
        # trainer.finetune_with_local_and_kl(finetune_steps=finetune_steps)
        trainer.finetune_with_local_and_kl_freq(finetune_steps=finetune_steps)
        finetune_save_path = os.path.join(trainer.results_folder, f'finetuned_checkpoint-{trainer.milestone}.pt')
        trainer.save(trainer.milestone, filename=finetune_save_path, verbose=True)
        print(f"Finetuned weights saved to {finetune_save_path}")
    elif args.sample == 1 and args.mode in ['infill', 'predict']:
        trainer.load(args.milestone)
        dataloader, dataset = test_dataloader_info['dataloader'], test_dataloader_info['dataset']
        coef = config['dataloader']['test_dataset']['coefficient']
        stepsize = config['dataloader']['test_dataset']['step_size']
        sampling_steps = config['dataloader']['test_dataset']['sampling_steps']
        samples, *_ = trainer.restore(dataloader, [dataset.window, dataset.var_num], coef, stepsize, sampling_steps)
        if dataset.auto_norm:
            samples = unnormalize_to_zero_to_one(samples)
            np.save(os.path.join(args.save_dir, f'ddpm_{args.mode}_{args.name}.npy'), samples)
    else:
        if args.finetune_path:
            if 'finetuned' in args.finetune_path or 'fault_model' in args.name:
                model.freeze_parameters(freeze_encoder=True, freeze_decoder=False)
                model.enable_finetuning()
            trainer.load(args.finetune_path, verbose=True)
        else:
            if 'fault_model' in args.name:
                model.freeze_parameters(freeze_encoder=True, freeze_decoder=False)
                model.enable_finetuning()
                default_path = os.path.join(trainer.results_folder, f'finetuned_checkpoint-{args.milestone}.pt')
                if os.path.exists(default_path):
                    trainer.load(default_path, verbose=True)
                else:
                    trainer.load(args.milestone)
            else:
                trainer.load(args.milestone)
        dataset = dataloader_info['dataset']
        if (args.finetune_path is not None and 'finetuned' in args.finetune_path) or 'finetuned' in args.name:
            num_samples = 200
        else:
            num_samples = len(dataset) 
        samples = trainer.sample(num=num_samples, size_every=args.size_every, shape=[dataset.window, dataset.var_num])
        if dataset.auto_norm:
            samples = unnormalize_to_zero_to_one(samples)
            save_path = os.path.join(args.save_dir, f'ddpm_fake_{args.name}.npy')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.save(save_path, samples)

if __name__ == '__main__':
    main()