import os
import sys
import time
import torch
import numpy as np
import torch.nn.functional as F

from pathlib import Path
from tqdm.auto import tqdm
from ema_pytorch import EMA
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from Utils.io_utils import instantiate_from_config, get_model_parameters_info


sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

def cycle(dl):
    while True:
        for data in dl:
            yield data


class Trainer(object):
    def __init__(self, config, args, model, dataloader, logger=None):
        super().__init__()
        self.config = config
        self.model = model
        self.device = self.model.betas.device
        self.train_num_steps = config['solver']['max_epochs']
        self.gradient_accumulate_every = config['solver']['gradient_accumulate_every']
        self.save_cycle = config['solver']['save_cycle']
        self.dl = cycle(dataloader['dataloader'])
        self.step = 0
        self.milestone = 0
        self.args = args
        self.logger = logger

        self.results_folder = Path(config['solver']['results_folder'] + f'_{model.seq_length}')
        os.makedirs(self.results_folder, exist_ok=True)

        start_lr = config['solver'].get('base_lr', 1.0e-4)
        ema_decay = config['solver']['ema']['decay']
        ema_update_every = config['solver']['ema']['update_interval']
        sc_cfg = config['solver']['scheduler']
        if args.finetune:
            self.opt = None
            self.sch = None
        else:
            # 先初始化优化器
            self.opt = Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=start_lr,
                betas=[0.9, 0.96]
            )
            # 更新调度器配置中的 optimizer
            sc_cfg['params']['optimizer'] = self.opt
            self.sch = instantiate_from_config(sc_cfg)

        # 初始化 EMA
        self.ema = EMA(self.model, beta=ema_decay, update_every=ema_update_every).to(self.device)
        
        if self.logger is not None:
            self.logger.log_info(str(get_model_parameters_info(self.model)))
        self.log_frequency = 100

    def save(self, milestone, filename=None, verbose=False):
        save_path = filename if filename else str(self.results_folder / f'checkpoint-{milestone}.pt')
        if self.logger is not None and verbose:
            self.logger.log_info('Save current model to {}'.format(save_path))
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema.state_dict(),
            'opt': self.opt.state_dict(),
        }
        torch.save(data, save_path)

    def load(self, milestone, verbose=False):
        """
        Load a checkpoint from the specified path or milestone.

        Args:
            milestone (str or int): Path to the checkpoint file (if str) or milestone number (if int).
            verbose (bool): If True, print additional information during loading.
        """
        # Determine checkpoint path
        if isinstance(milestone, str):
            checkpoint_path = milestone
        else:
            checkpoint_path = str(self.results_folder / f'checkpoint-{milestone}.pt')

        # Log checkpoint path if logger is available
        if self.logger is not None and verbose:
            self.logger.log_info('Resume from {}'.format(checkpoint_path))

        # Check if checkpoint file exists
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")

        # Load checkpoint data
        # Note: Using weights_only=False for compatibility with current checkpoint format
        # In the future, consider saving state_dict separately to enable weights_only=True
        data = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state dict (ignore missing keys, e.g., local_adapter in finetuning)
        self.model.load_state_dict(data['model'], strict=False)

        # Load EMA state dict (ignore missing keys)
        # self.ema.load_state_dict(data['ema'], strict=False)

        # Update training step
        self.step = data['step']

        # Update milestone
        self.milestone = int(checkpoint_path.split('-')[-1].split('.')[0]) if isinstance(milestone, str) else milestone

        if 'ema' in data:
            self.ema.load_state_dict(data['ema'], strict=False)

        # 只在训练或微调时初始化并加载优化器
        if self.args.finetune or self.args.train:
            start_lr = self.config.get('finetune_lr', self.config['solver'].get('base_lr', 1.0e-5)) if self.args.finetune else self.config['solver'].get('base_lr', 1.0e-5)
            if self.args.finetune:
                self.opt = Adam(self.model.local_adapters.parameters(), lr=start_lr, betas=[0.9, 0.96])
            else:
                self.opt = Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=start_lr, betas=[0.9, 0.96])
            if 'opt' in data:
                try:
                    self.opt.load_state_dict(data['opt'])
                except ValueError as e:
                    print(f"Warning: Optimizer state dict mismatch, skipping optimizer load: {e}")
            sc_cfg = self.config['solver']['scheduler']
            sc_cfg['params']['optimizer'] = self.opt
            self.sch = instantiate_from_config(sc_cfg)

        # Log loading completion
        if verbose:
            if self.logger is not None:
                self.logger.log_info(f"Loaded checkpoint at step {self.step}")
            else:
                print(f"Loaded checkpoint at step {self.step}")
    def train(self):
        device = self.device
        step = 0
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info('{}: start training...'.format(self.args.name), check_primary=False)

        with tqdm(initial=step, total=self.train_num_steps) as pbar:
            while step < self.train_num_steps:
                total_loss = 0.
                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl).to(device)
                    loss = self.model(data, target=data)
                    loss = loss / self.gradient_accumulate_every
                    loss.backward()
                    total_loss += loss.item()

                pbar.set_description(f'loss: {total_loss:.6f}')
                clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()
                self.sch.step(total_loss)
                self.opt.zero_grad()
                self.step += 1
                step += 1
                self.ema.update()

                with torch.no_grad():
                    if self.step != 0 and self.step % self.save_cycle == 0:
                        self.milestone += 1
                        self.save(self.milestone)
                    
                    if self.logger is not None and self.step % self.log_frequency == 0:
                        self.logger.add_scalar(tag='train/loss', scalar_value=total_loss, global_step=self.step)

                pbar.update(1)

        print('training complete')
        if self.logger is not None:
            self.logger.log_info('Training done, time: {:.2f}'.format(time.time() - tic))

    def finetune(self, finetune_steps=2000):
        device = self.device
        self.model.freeze_parameters(freeze_encoder=True, freeze_decoder=False)
        finetune_lr = self.args.opts.get('finetune_lr', 1e-6) if self.args.opts else 1e-6
        self.opt = Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=finetune_lr, betas=[0.9, 0.96])
        sc_cfg = self.config['solver']['scheduler']
        sc_cfg['params']['optimizer'] = self.opt
        self.sch = instantiate_from_config(sc_cfg)

        if self.logger is not None:
            tic = time.time()
            self.logger.log_info('{}: start finetuning...'.format(self.args.name), check_primary=False)

        start_step = self.step
        with tqdm(initial=start_step, total=start_step + finetune_steps) as pbar:
            while self.step < start_step + finetune_steps:
                total_loss = 0.
                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl).to(device)
                    loss = self.model.finetune_forward(data, target=data)
                    loss = loss / self.gradient_accumulate_every
                    loss.backward()
                    total_loss += loss.item()

                pbar.set_description(f'finetune loss: {total_loss:.6f}')
                clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()
                self.sch.step(total_loss)
                self.opt.zero_grad()
                self.step += 1
                self.ema.update()

                with torch.no_grad():
                    if self.logger is not None and self.step % self.log_frequency == 0:
                        self.logger.add_scalar(tag='finetune/loss', scalar_value=total_loss, global_step=self.step)

                pbar.update(1)

        print('finetuning complete')
        if self.logger is not None:
            self.logger.log_info('Finetuning done, time: {:.2f}'.format(time.time() - tic))
    # def finetune_with_local_and_kl_freq(self, finetune_steps=2000, kl_weight_initial=0.01, kl_weight_final=0.1, 
    #                            fault_weight=0.5, freq_weight=0.3, diversity_weight=0.1):
    #     device = self.device
    #     self.model.enable_finetuning()
    #     self.ema = EMA(self.model, beta=self.config['solver']['ema']['decay'], 
    #                 update_every=self.config['solver']['ema']['update_interval']).to(device)
        
        
    #     opt = Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=1e-5, betas=[0.9, 0.96])

    #     if self.logger is not None:
    #         tic = time.time()
    #         self.logger.log_info('{}: start finetuning with local attention and KL...'.format(self.args.name))

    #     # 固定目标步数
    #     target_step = self.step + finetune_steps
    #     print(f"Initial step: {self.step}, finetune_steps: {finetune_steps}, target_step: {target_step}")

    #     # 定义辅助函数
    #     # 时域特征损失：突变点差异
    #     def compute_fault_feature_loss(generated, target):
    #         gen_diff = torch.diff(generated, dim=-1)
    #         target_diff = torch.diff(target, dim=-1)
    #         fault_loss = F.mse_loss(gen_diff, target_diff)
    #         return fault_loss

    #     # 频域特征损失：频谱差异
    #     def compute_freq_loss(generated, target):
    #         gen_fft = torch.fft.fft(generated, dim=-1).abs()
    #         target_fft = torch.fft.fft(target, dim=-1).abs()
    #         freq_loss = F.mse_loss(gen_fft, target_fft)
    #         return freq_loss

    #     # 多样性损失：计算生成样本之间的均方差
    #     def compute_diversity_loss(outputs, num_samples=2):
    #         samples = []
    #         for _ in range(num_samples):
    #             sample = self.model.finetune_forward(data, t=t)
    #             samples.append(sample)
    #         samples = torch.stack(samples, dim=0)  # [num_samples, batch_size, ...]
    #         diversity_loss = torch.mean((samples[0] - samples[1]) ** 2)
    #         return diversity_loss

    #     # 动态KL权重
    #     def get_dynamic_kl_weight(step, total_steps, initial_weight, final_weight):
    #         progress = step / total_steps
    #         return initial_weight + (final_weight - initial_weight) * progress

    #     # 动态EMA更新频率
    #     def get_dynamic_update_interval(step, total_steps, initial_interval=10, final_interval=1):
    #         progress = step / total_steps
    #         return int(initial_interval - (initial_interval - final_interval) * progress)

    #     # 微调循环
    #     with tqdm(initial=self.step, total=target_step) as pbar:
    #         while self.step < target_step:
    #             total_loss = 0.
    #             for _ in range(self.gradient_accumulate_every):
    #                 # 加载数据（无条件信息）
    #                 data = next(self.dl).to(device)
    #                 t = torch.randint(0, self.model.num_timesteps, (data.shape[0],), device=device).long()

    #                 # 预训练模型输出
    #                 with torch.no_grad():
    #                     pretrained_out = self.ema.ema_model.output(data, t)

    #                 # 微调模型输出（无条件信息）
    #                 finetune_out = self.model.finetune_forward(data, t=t)

    #                 # 基础去噪损失
    #                 base_loss = self.model._train_loss(x_start=data, t=t, target=data)

    #                 # KL散度损失
    #                 kl_loss = F.kl_div(
    #                     F.log_softmax(finetune_out.view(-1, finetune_out.shape[-1]), dim=-1),
    #                     F.softmax(pretrained_out.view(-1, pretrained_out.shape[-1]), dim=-1),
    #                     reduction='batchmean'
    #                 )

    #                 # 故障特征损失
    #                 fault_loss = compute_fault_feature_loss(finetune_out, data)
    #                 freq_loss = compute_freq_loss(finetune_out, data)

    #                 # 多样性损失
    #                 diversity_loss = compute_diversity_loss(finetune_out)

    #                 # 动态KL权重
    #                 kl_weight = get_dynamic_kl_weight(self.step, target_step, kl_weight_initial, kl_weight_final)

    #                 # 总损失
    #                 loss = (base_loss + kl_weight * kl_loss + fault_weight * fault_loss + 
    #                         freq_weight * freq_loss + diversity_weight * diversity_loss) / self.gradient_accumulate_every
    #                 loss.backward()
    #                 total_loss += loss.item()

    #             # 更新进度条描述
    #             pbar.set_description(f'finetune loss: {total_loss:.6f}, kl_loss: {kl_loss.item():.6f}, '
    #                             f'fault_loss: {fault_loss.item():.6f}, freq_loss: {freq_loss.item():.6f}, '
    #                             f'diversity_loss: {diversity_loss.item():.6f}')

    #             # 梯度裁剪
    #             torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

    #             # 优化器更新
    #             opt.step()
    #             opt.zero_grad()
    #             self.step += 1

    #             # 动态调整EMA更新频率
    #             self.ema.update_every = get_dynamic_update_interval(self.step, target_step)
    #             self.ema.update()

    #             # 日志记录
    #             if self.logger is not None and self.step % self.log_frequency == 0:
    #                 self.logger.add_scalar(tag='finetune/total_loss', scalar_value=total_loss, global_step=self.step)
    #                 self.logger.add_scalar(tag='finetune/kl_loss', scalar_value=kl_loss.item(), global_step=self.step)
    #                 self.logger.add_scalar(tag='finetune/fault_loss', scalar_value=fault_loss.item(), global_step=self.step)
    #                 self.logger.add_scalar(tag='finetune/freq_loss', scalar_value=freq_loss.item(), global_step=self.step)
    #                 self.logger.add_scalar(tag='finetune/diversity_loss', scalar_value=diversity_loss.item(), global_step=self.step)

    #             pbar.update(1)

    #     print('finetuning with local attention and KL complete')
    #     if self.logger is not None:
    #         self.logger.log_info('Finetuning done, time: {:.2f}'.format(time.time() - tic))
    def finetune_with_local_and_kl_freq(self, finetune_steps=2000, kl_weight_initial=0.01, kl_weight_final=0.1, 
                                    fault_weight=0.5, freq_weight=0.3, diversity_weight=0.1):
        device = self.device
        self.model.enable_finetuning()  # 启用微调模式，初始化两个局部注意力适配器
        self.ema = EMA(self.model, beta=self.config['solver']['ema']['decay'], 
                    update_every=self.config['solver']['ema']['update_interval']).to(device)
        
        # 只优化局部注意力适配器参数
        opt = Adam(self.model.local_adapters.parameters(), lr=1e-5, betas=[0.9, 0.96])

        if self.logger is not None:
            tic = time.time()
            self.logger.log_info('{}: start finetuning with local attention and KL...'.format(self.args.name))

        # 固定目标步数
        target_step = self.step + finetune_steps
        print(f"Initial step: {self.step}, finetune_steps: {finetune_steps}, target_step: {target_step}")

        # 定义辅助函数
        # 时域特征损失：突变点差异
        def compute_fault_feature_loss(generated, target):
            gen_diff = torch.diff(generated, dim=-1)
            target_diff = torch.diff(target, dim=-1)
            fault_loss = F.mse_loss(gen_diff, target_diff)
            return fault_loss

        # 频域特征损失：频谱差异
        def compute_freq_loss(generated, target):
            gen_fft = torch.fft.fft(generated, dim=-1).abs()
            target_fft = torch.fft.fft(target, dim=-1).abs()
            freq_loss = F.mse_loss(gen_fft, target_fft)
            return freq_loss

        # 多样性损失：计算生成样本之间的均方差
        def compute_diversity_loss(outputs, num_samples=2):
            samples = []
            for _ in range(num_samples):
                sample = self.model.finetune_forward(data, t=t)  # 使用当前数据和时间步
                samples.append(sample)
            samples = torch.stack(samples, dim=0)  # [num_samples, batch_size, ...]
            diversity_loss = torch.mean((samples[0] - samples[1]) ** 2)
            return diversity_loss

        # 动态KL权重
        def get_dynamic_kl_weight(step, total_steps, initial_weight, final_weight):
            progress = step / total_steps
            return initial_weight + (final_weight - initial_weight) * progress

        # 动态EMA更新频率
        def get_dynamic_update_interval(step, total_steps, initial_interval=10, final_interval=1):
            progress = step / total_steps
            return int(initial_interval - (initial_interval - final_interval) * progress)

        # 微调循环
        with tqdm(initial=self.step, total=target_step) as pbar:
            while self.step < target_step:
                total_loss = 0.
                for _ in range(self.gradient_accumulate_every):
                    # 加载数据（无条件信息）
                    data = next(self.dl).to(device)
                    # print(f"Data shape: {data.shape}")  # 添加调试语句
                    t = torch.randint(0, self.model.num_timesteps, (data.shape[0],), device=device).long()

                    # 预训练模型输出
                    with torch.no_grad():
                        pretrained_out = self.ema.ema_model.output(data, t)

                    # 微调模型输出（调用新的 finetune_forward）
                    finetune_out = self.model.finetune_forward(data, t=t)

                    # 基础去噪损失
                    base_loss = self.model._train_loss(x_start=data, t=t, target=data)

                    # KL散度损失
                    kl_loss = F.kl_div(
                        F.log_softmax(finetune_out.view(-1, finetune_out.shape[-1]), dim=-1),
                        F.softmax(pretrained_out.view(-1, pretrained_out.shape[-1]), dim=-1),
                        reduction='batchmean'
                    )

                    # 故障特征损失
                    fault_loss = compute_fault_feature_loss(finetune_out, data)
                    freq_loss = compute_freq_loss(finetune_out, data)

                    # 多样性损失
                    diversity_loss = compute_diversity_loss(finetune_out)

                    # 动态KL权重
                    kl_weight = get_dynamic_kl_weight(self.step, target_step, kl_weight_initial, kl_weight_final)

                    # 总损失
                    loss = (base_loss + kl_weight * kl_loss + fault_weight * fault_loss + 
                            freq_weight * freq_loss + diversity_weight * diversity_loss) / self.gradient_accumulate_every
                    loss.backward()
                    total_loss += loss.item()

                # 更新进度条描述
                pbar.set_description(f'finetune loss: {total_loss:.6f}, kl_loss: {kl_loss.item():.6f}, '
                                    f'fault_loss: {fault_loss.item():.6f}, freq_loss: {freq_loss.item():.6f}, '
                                    f'diversity_loss: {diversity_loss.item():.6f}')

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.local_adapters.parameters(), 1.0)

                # 优化器更新
                opt.step()
                opt.zero_grad()
                self.step += 1

                # 动态调整EMA更新频率
                self.ema.update_every = get_dynamic_update_interval(self.step, target_step)
                self.ema.update()

                # 定期保存权重
                with torch.no_grad():
                    if self.step != 0 and self.step % self.save_cycle == 0:
                        self.milestone += 1
                        save_filename = str(self.results_folder / f'checkpoint-finetune-{self.step}.pt')
                        self.save(self.milestone, filename=save_filename, verbose=True)


                # 日志记录
                if self.logger is not None and self.step % self.log_frequency == 0:
                    self.logger.add_scalar(tag='finetune/total_loss', scalar_value=total_loss, global_step=self.step)
                    self.logger.add_scalar(tag='finetune/kl_loss', scalar_value=kl_loss.item(), global_step=self.step)
                    self.logger.add_scalar(tag='finetune/fault_loss', scalar_value=fault_loss.item(), global_step=self.step)
                    self.logger.add_scalar(tag='finetune/freq_loss', scalar_value=freq_loss.item(), global_step=self.step)
                    self.logger.add_scalar(tag='finetune/diversity_loss', scalar_value=diversity_loss.item(), global_step=self.step)

                pbar.update(1)
        # 保存最终权重
        with torch.no_grad():
            save_filename = str(self.results_folder / f'checkpoint-finetune-{self.step}-final.pt')
            self.save(self.milestone + 1, filename=save_filename, verbose=True)
        print('finetuning with local attention and KL complete')
        if self.logger is not None:
            self.logger.log_info('Finetuning done, time: {:.2f}'.format(time.time() - tic))
        def finetune_with_local_and_kl(self, finetune_steps=2000, kl_weight=0.1):
            device = self.device
            self.model.enable_finetuning()
            self.ema = EMA(self.model, beta=self.config['solver']['ema']['decay'], update_every=self.config['solver']['ema']['update_interval']).to(device)
            opt = Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=1e-6, betas=[0.9, 0.96])

            if self.logger is not None:
                tic = time.time()
                self.logger.log_info('{}: start finetuning with local attention and KL...'.format(self.args.name))

            # 固定目标步数
            target_step = self.step + finetune_steps
            print(f"Initial step: {self.step}, finetune_steps: {finetune_steps}, target_step: {target_step}")
            with tqdm(initial=self.step, total=target_step) as pbar:
                while self.step < target_step:
                    total_loss = 0.
                    for _ in range(self.gradient_accumulate_every):
                        data = next(self.dl).to(device)
                        t = torch.randint(0, self.model.num_timesteps, (data.shape[0],), device=device).long()
                        with torch.no_grad():
                            pretrained_out = self.ema.ema_model.output(data, t)
                        finetune_out = self.model.finetune_forward(data, t=t)
                        base_loss = self.model._train_loss(x_start=data, t=t, target=data)
                        kl_loss = F.kl_div(
                            F.log_softmax(finetune_out.view(-1, finetune_out.shape[-1]), dim=-1),
                            F.softmax(pretrained_out.view(-1, pretrained_out.shape[-1]), dim=-1),
                            reduction='batchmean'
                        )
                        loss = base_loss + kl_weight * kl_loss
                        loss = loss / self.gradient_accumulate_every
                        loss.backward()
                        total_loss += loss.item()

                    pbar.set_description(f'finetune loss: {total_loss:.6f}, kl_loss: {kl_loss.item():.6f}')
                    clip_grad_norm_(self.model.parameters(), 1.0)
                    opt.step()
                    opt.zero_grad()
                    self.step += 1
                    self.ema.update()

                    if self.logger is not None and self.step % self.log_frequency == 0:
                        self.logger.add_scalar(tag='finetune/total_loss', scalar_value=total_loss, global_step=self.step)
                        self.logger.add_scalar(tag='finetune/kl_loss', scalar_value=kl_loss.item(), global_step=self.step)

                    pbar.update(1)

            print('finetuning with local attention and KL complete')
            if self.logger is not None:
                self.logger.log_info('Finetuning done, time: {:.2f}'.format(time.time() - tic))

    def sample(self, num, size_every, shape=None):
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info('Begin to sample...')
        samples = np.empty([0, shape[0], shape[1]])
        num_cycle = int(num // size_every) + 1

        for _ in range(num_cycle):
            sample = self.ema.ema_model.generate_mts(batch_size=size_every)
            samples = np.row_stack([samples, sample.detach().cpu().numpy()])
            torch.cuda.empty_cache()

        if self.logger is not None:
            self.logger.log_info('Sampling done, time: {:.2f}'.format(time.time() - tic))
        return samples

    def restore(self, raw_dataloader, shape=None, coef=1e-1, stepsize=1e-1, sampling_steps=50):
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info('Begin to restore...')
        model_kwargs = {}
        model_kwargs['coef'] = coef
        model_kwargs['learning_rate'] = stepsize
        samples = np.empty([0, shape[0], shape[1]])
        reals = np.empty([0, shape[0], shape[1]])
        masks = np.empty([0, shape[0], shape[1]])

        for idx, (x, t_m) in enumerate(raw_dataloader):
            x, t_m = x.to(self.device), t_m.to(self.device)
            if sampling_steps == self.model.num_timesteps:
                sample = self.ema.ema_model.sample_infill(shape=x.shape, target=x*t_m, partial_mask=t_m,
                                                          model_kwargs=model_kwargs)
            else:
                sample = self.ema.ema_model.fast_sample_infill(shape=x.shape, target=x*t_m, partial_mask=t_m, model_kwargs=model_kwargs,
                                                               sampling_timesteps=sampling_steps)
            samples = np.row_stack([samples, sample.detach().cpu().numpy()])
            reals = np.row_stack([reals, x.detach().cpu().numpy()])
            masks = np.row_stack([masks, t_m.detach().cpu().numpy()])
        
        if self.logger is not None:
            self.logger.log_info('Imputation done, time: {:.2f}'.format(time.time() - tic))
        return samples, reals, masks