#!/usr/bin/env python3
"""Trains Karras et al. (2022) diffusion models."""

import argparse , math , json , accelerate, torch
from copy import deepcopy
from functools import partial
from pathlib import Path
from fastcore.script import call_parse

from torch import nn, optim
from torch import multiprocessing as mp
from torch.utils import data
from torchvision import datasets, transforms, utils
from tqdm.auto import trange, tqdm

import k_diffusion as K

@call_parse
def main(
    config:str, # the configuration file
    batch_size:int=256, # the batch size
    demo_every:int=500, # save a demo grid every this many steps
    evaluate_every:int=10000, # save a demo grid every this many steps
    evaluate_n:int=2000, # the number of samples to draw to evaluate
    gns:bool=False, # measure the gradient noise scale (DDP only)
    grad_accum_steps:int=1, # the number of gradient accumulation steps
    grow:str=None, # the checkpoint to grow from
    grow_cfg:str=None, # the configuration file of the model to grow from
    lr:float=None, # the learning rate
    name:str='model', # the name of the run
    num_workers:int=8, # the number of data loader workers
    resume:str=None, # the checkpoint to resume from
    sample_n:int=64, # the number of images to sample for demo grids
    save_every:int=10000, # save every this many steps
    seed:int=None, # the random seed
    start_method:str='spawn' # the multiprocessing start method
):
    #choices=['fork', 'forkserver', 'spawn'],
    mp.set_start_method(start_method)
    torch.backends.cuda.matmul.allow_tf32 = True

    config = K.config.load_cfg(open(config))
    model_cfg = config['model']
    dataset_cfg = config['dataset']
    opt_cfg = config['optimizer']
    sched_cfg = config['lr_sched']
    ema_sched_cfg = config['ema_sched']

    # TODO: allow non-square input sizes
    assert len(model_cfg['input_size']) == 2 and model_cfg['input_size'][0] == model_cfg['input_size'][1]
    size = model_cfg['input_size']

    ddp_kwargs = accelerate.DistributedDataParallelKwargs(find_unused_parameters=model_cfg['skip_stages'] > 0)
    accelerator = accelerate.Accelerator(kwargs_handlers=[ddp_kwargs], gradient_accumulation_steps=grad_accum_steps)
    device = accelerator.device
    print(f'Process {accelerator.process_index} using device: {device}', flush=True)

    if seed is not None:
        seeds = torch.randint(-2 ** 63, 2 ** 63 - 1, [accelerator.num_processes], generator=torch.Generator().manual_seed(seed))
        torch.manual_seed(seeds[accelerator.process_index])

    inner_model = K.config.make_model(config)
    if accelerator.is_main_process: print('Parameters:', K.utils.n_params(inner_model))

    if not lr: lr = opt_cfg['lr']
    if opt_cfg['type'] == 'adamw':
        opt = optim.AdamW(inner_model.parameters(), lr=lr,
                          betas=tuple(opt_cfg['betas']),
                          eps=opt_cfg['eps'],
                          weight_decay=opt_cfg['weight_decay'])
    elif opt_cfg['type'] == 'sgd':
        opt = optim.SGD(inner_model.parameters(), lr=lr,
                        momentum=opt_cfg.get('momentum', 0.),
                        nesterov=opt_cfg.get('nesterov', False),
                        weight_decay=opt_cfg.get('weight_decay', 0.))
    else: raise ValueError('Invalid optimizer type')

    if sched_cfg['type'] == 'inverse':
        sched = K.utils.InverseLR(opt, inv_gamma=sched_cfg['inv_gamma'], power=sched_cfg['power'], warmup=sched_cfg['warmup'])
    elif sched_cfg['type'] == 'exponential':
        sched = K.utils.ExponentialLR(opt, num_steps=sched_cfg['num_steps'], decay=sched_cfg['decay'], warmup=sched_cfg['warmup'])
    else: raise ValueError('Invalid schedule type')

    assert ema_sched_cfg['type'] == 'inverse'
    ema_sched = K.utils.EMAWarmup(power=ema_sched_cfg['power'], max_value=ema_sched_cfg['max_value'])

    tf = transforms.Compose([
        transforms.Resize(size[0], interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop(size[0]),
        K.augmentation.KarrasAugmentationPipeline(model_cfg['augment_prob']),
    ])

    if dataset_cfg['type'] == 'imagefolder':
        train_set = K.utils.FolderOfImages(dataset_cfg['location'], transform=tf)
    elif dataset_cfg['type'] == 'cifar10':
        train_set = datasets.CIFAR10(dataset_cfg['location'], train=True, download=True, transform=tf)
    elif dataset_cfg['type'] == 'fashion':
        train_set = datasets.FashionMNIST(dataset_cfg['location'], train=True, download=True, transform=tf)
    elif dataset_cfg['type'] == 'mnist':
        train_set = datasets.MNIST(dataset_cfg['location'], train=True, download=True, transform=tf)
    elif dataset_cfg['type'] == 'huggingface':
        from datasets import load_dataset
        train_set = load_dataset(dataset_cfg['location'])
        train_set.set_transform(partial(K.utils.hf_datasets_augs_helper, transform=tf, image_key=dataset_cfg['image_key']))
        train_set = train_set['train']
    else: raise ValueError('Invalid dataset type')

    if accelerator.is_main_process:
        try: print('Number of items in dataset:', len(train_set))
        except TypeError: pass

    image_key = dataset_cfg.get('image_key', 0)
    train_dl = data.DataLoader(train_set, batch_size, shuffle=True, drop_last=True, num_workers=num_workers, persistent_workers=True)

    if grow:
        if not grow_cfg: raise ValueError('--grow requires --grow-config')
        ckpt = torch.load(grow, map_location='cpu')
        old_cfg = K.config.load_cfg(open(grow_cfg))
        old_inner_model = K.config.make_model(old_cfg)
        old_inner_model.load_state_dict(ckpt['model_ema'])
        if old_cfg['model']['skip_stages'] != model_cfg['skip_stages']: old_inner_model.set_skip_stages(model_cfg['skip_stages'])
        if old_cfg['model']['patch_size'] !=  model_cfg['patch_size']:  old_inner_model.set_patch_size(model_cfg['patch_size'])
        inner_model.load_state_dict(old_inner_model.state_dict())
        del ckpt, old_inner_model

    inner_model, opt, train_dl = accelerator.prepare(inner_model, opt, train_dl)
    if gns:
        gns_stats_hook = K.gns.DDPGradientStatsHook(inner_model)
        gns_stats = K.gns.GradientNoiseScale()
    else: gns_stats = None
    sigma_min = model_cfg['sigma_min']
    sigma_max = model_cfg['sigma_max']
    sample_density = K.config.make_sample_density(model_cfg)
    model = K.config.make_denoiser_wrapper(config)(inner_model)
    model_ema = deepcopy(model)
    state_path = Path(f'{name}_state.json')

    if state_path.exists() or resume:
        if resume: ckpt_path = resume
        if not resume:
            state = json.load(open(state_path))
            ckpt_path = state['latest_checkpoint']
        if accelerator.is_main_process: print(f'Resuming from {ckpt_path}...')
        ckpt = torch.load(ckpt_path, map_location='cpu')
        accelerator.unwrap_model(model.inner_model).load_state_dict(ckpt['model'])
        accelerator.unwrap_model(model_ema.inner_model).load_state_dict(ckpt['model_ema'])
        opt.load_state_dict(ckpt['opt'])
        sched.load_state_dict(ckpt['sched'])
        ema_sched.load_state_dict(ckpt['ema_sched'])
        epoch = ckpt['epoch'] + 1
        step = ckpt['step'] + 1
        if gns and ckpt.get('gns_stats', None) is not None: gns_stats.load_state_dict(ckpt['gns_stats'])
        del ckpt
    else: epoch,step = 0,0

    evaluate_enabled = evaluate_every > 0 and evaluate_n > 0
    if evaluate_enabled:
        extractor = K.evaluation.InceptionV3FeatureExtractor(device=device)
        train_iter = iter(train_dl)
        if accelerator.is_main_process: print('Computing features for reals...')
        reals_features = K.evaluation.compute_features(accelerator, lambda x: next(train_iter)[image_key][1], extractor, evaluate_n, batch_size)
        if accelerator.is_main_process: metrics_log = K.utils.CSVLogger(f'{name}_metrics.csv', ['step', 'fid', 'kid'])
        del train_iter

    @torch.no_grad()
    @K.utils.eval_mode(model_ema)
    def demo():
        if accelerator.is_main_process: tqdm.write('Sampling...')
        filename = f'{name}_demo_{step:08}.png'
        n_per_proc = math.ceil(sample_n / accelerator.num_processes)
        x = torch.randn([n_per_proc, model_cfg['input_channels'], size[0], size[1]], device=device) * sigma_max
        sigmas = K.sampling.get_sigmas_karras(50, sigma_min, sigma_max, rho=7., device=device)
        x_0 = K.sampling.sample_lms(model_ema, x, sigmas, disable=not accelerator.is_main_process)
        x_0 = accelerator.gather(x_0)[:sample_n]
        if accelerator.is_main_process:
            grid = utils.make_grid(x_0, nrow=math.ceil(sample_n ** 0.5), padding=0)
            K.utils.to_pil_image(grid).save(filename)

    @torch.no_grad()
    @K.utils.eval_mode(model_ema)
    def evaluate():
        if not evaluate_enabled: return
        if accelerator.is_main_process: tqdm.write('Evaluating...')
        sigmas = K.sampling.get_sigmas_karras(50, sigma_min, sigma_max, rho=7., device=device)
        def sample_fn(n):
            x = torch.randn([n, model_cfg['input_channels'], size[0], size[1]], device=device) * sigma_max
            x_0 = K.sampling.sample_lms(model_ema, x, sigmas, disable=True)
            return x_0
        fakes_features = K.evaluation.compute_features(accelerator, sample_fn, extractor, evaluate_n, batch_size)
        if accelerator.is_main_process:
            fid = K.evaluation.fid(fakes_features, reals_features)
            kid = K.evaluation.kid(fakes_features, reals_features)
            print(f'FID: {fid.item():g}, KID: {kid.item():g}')
            if accelerator.is_main_process: metrics_log.write(step, fid.item(), kid.item())

    def save():
        accelerator.wait_for_everyone()
        filename = f'{name}_{step:08}.pth'
        if accelerator.is_main_process: tqdm.write(f'Saving to {filename}...')
        obj = {
            'model': accelerator.unwrap_model(model.inner_model).state_dict(),
            'model_ema': accelerator.unwrap_model(model_ema.inner_model).state_dict(),
            'opt': opt.state_dict(),
            'sched': sched.state_dict(),
            'ema_sched': ema_sched.state_dict(),
            'epoch': epoch,
            'step': step,
            'gns_stats': gns_stats.state_dict() if gns_stats is not None else None,
        }
        accelerator.save(obj, filename)
        if accelerator.is_main_process:
            state_obj = {'latest_checkpoint': filename}
            json.dump(state_obj, open(state_path, 'w'))

    try:
        while True:
            for batch in tqdm(train_dl, disable=not accelerator.is_main_process):
                with accelerator.accumulate(model):
                    reals, _, aug_cond = batch[image_key]
                    noise = torch.randn_like(reals)
                    sigma = sample_density([reals.shape[0]], device=device)
                    losses = model.loss(reals, noise, sigma, aug_cond=aug_cond)
                    losses_all = accelerator.gather(losses)
                    loss = losses_all.mean()
                    accelerator.backward(losses.mean())
                    if gns:
                        sq_norm_small_batch, sq_norm_large_batch = gns_stats_hook.get_stats()
                        gns_stats.update(sq_norm_small_batch, sq_norm_large_batch, reals.shape[0], reals.shape[0] * accelerator.num_processes)
                    opt.step()
                    sched.step()
                    opt.zero_grad()
                    if accelerator.sync_gradients:
                        ema_decay = ema_sched.get_value()
                        K.utils.ema_update(model, model_ema, ema_decay)
                        ema_sched.step()

                if accelerator.is_main_process:
                    if step % 25 == 0:
                        if gns: tqdm.write(f'Epoch: {epoch}, step: {step}, loss: {loss.item():g}, gns: {gns_stats.get_gns():g}')
                        else: tqdm.write(f'Epoch: {epoch}, step: {step}, loss: {loss.item():g}')

                if step % demo_every == 0: demo()
                if evaluate_enabled and step > 0 and step % evaluate_every == 0: evaluate()
                if step > 0 and step % save_every == 0: save()
                step += 1
            epoch += 1
    except KeyboardInterrupt:
        pass

