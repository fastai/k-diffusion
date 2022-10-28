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
    evaluate_every:int=5000, # save a demo grid every this many steps
    evaluate_n:int=2000, # the number of samples to draw to evaluate
    lr:float=None, # the learning rate
    name:str='model', # the name of the run
    num_workers:int=8, # the number of data loader workers
    sample_n:int=64, # the number of images to sample for demo grids
    save_every:int=10000, # save every this many steps
    seed:int=None, # the random seed
    start_method:str='spawn' # the multiprocessing start method
):
    #choices=['fork', 'forkserver', 'spawn'],
    mp.set_start_method(start_method)
    torch.backends.cuda.matmul.allow_tf32 = True
    path = Path('outputs')
    path.mkdir(exist_ok=True)

    config = K.config.load_config(open(config))
    model_cfg = config['model']
    dataset_cfg = config['dataset']
    opt_cfg = config['optimizer']
    sched_cfg = config['lr_sched']
    ema_sched_cfg = config['ema_sched']

    # TODO: allow non-square input sizes
    assert len(model_cfg['input_size']) == 2 and model_cfg['input_size'][0] == model_cfg['input_size'][1]
    size = model_cfg['input_size']

    accelerator = accelerate.Accelerator()
    device = accelerator.device
    print(f'Process {accelerator.process_index} using device: {device}', flush=True)

    if seed is not None: torch.manual_seed(seed)
    inner_model = K.config.make_model(config)
    print('Parameters:', K.utils.n_params(inner_model))

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

    try: print('Number of items in dataset:', len(train_set))
    except TypeError: pass

    image_key = dataset_cfg.get('image_key', 0)
    train_dl = data.DataLoader(train_set, batch_size, shuffle=True, drop_last=True, num_workers=num_workers, persistent_workers=True)

    inner_model, opt, train_dl = accelerator.prepare(inner_model, opt, train_dl)
    sigma_min = model_cfg['sigma_min']
    sigma_max = model_cfg['sigma_max']
    sample_density = K.config.make_sample_density(model_cfg)
    model = K.config.make_denoiser_wrapper(config)(inner_model)
    model_ema = deepcopy(model)
    epoch,step = 0,0

    evaluate_enabled = evaluate_every > 0 and evaluate_n > 0
    if evaluate_enabled:
        extractor = K.evaluation.InceptionV3FeatureExtractor(device=device)
        train_iter = iter(train_dl)
        print('Computing features for reals...')
        reals_features = K.evaluation.compute_features(accelerator, lambda x: next(train_iter)[image_key][1], extractor, evaluate_n, batch_size)
        metrics_log = K.utils.CSVLogger(path/f'{name}_metrics.csv', ['step', 'fid', 'kid'])
        del train_iter

    @torch.no_grad()
    @K.utils.eval_mode(model_ema)
    def demo():
        tqdm.write('Sampling...')
        filename = path/f'{name}_demo_{step:08}.png'
        x = torch.randn([sample_n, model_cfg['input_channels'], size[0], size[1]], device=device) * sigma_max
        sigmas = K.sampling.get_sigmas_karras(50, sigma_min, sigma_max, rho=7., device=device)
        x_0 = K.sampling.sample_lms(model_ema, x, sigmas, disable=not accelerator.is_main_process)
        x_0 = x_0[:sample_n]
        grid = utils.make_grid(-x_0, nrow=math.ceil(sample_n ** 0.5), padding=0)
        K.utils.to_pil_image(grid).save(filename)

    @torch.no_grad()
    @K.utils.eval_mode(model_ema)
    def evaluate():
        if not evaluate_enabled: return
        tqdm.write('Evaluating...')
        sigmas = K.sampling.get_sigmas_karras(50, sigma_min, sigma_max, rho=7., device=device)
        def sample_fn(n):
            x = torch.randn([n, model_cfg['input_channels'], size[0], size[1]], device=device) * sigma_max
            x_0 = K.sampling.sample_lms(model_ema, x, sigmas, disable=True)
            return x_0
        fakes_features = K.evaluation.compute_features(accelerator, sample_fn, extractor, evaluate_n, batch_size)
        fid = K.evaluation.fid(fakes_features, reals_features)
        kid = K.evaluation.kid(fakes_features, reals_features)
        print(f'FID: {fid.item():g}, KID: {kid.item():g}')
        metrics_log.write(step, fid.item(), kid.item())

    def save():
        filename = path/f'{name}_{step:08}.pth'
        tqdm.write(f'Saving to {filename}...')
        obj = {
            'model': accelerator.unwrap_model(model.inner_model).state_dict(),
            'model_ema': accelerator.unwrap_model(model_ema.inner_model).state_dict(),
            'opt': opt.state_dict(), 'sched': sched.state_dict(), 'ema_sched': ema_sched.state_dict(),
            'epoch': epoch, 'step': step, }
        accelerator.save(obj, filename)

    try:
        while True:
            for batch in tqdm(train_dl):
                reals, _, aug_cond = batch[image_key]
                noise = torch.randn_like(reals)
                sigma = sample_density([reals.shape[0]], device=device)
                loss = model.loss(reals, noise, sigma, aug_cond=aug_cond).mean()
                accelerator.backward(loss)
                opt.step()
                sched.step()
                opt.zero_grad()
                ema_decay = ema_sched.get_value()
                K.utils.ema_update(model, model_ema, ema_decay)
                ema_sched.step()

                if step % 25 == 0: tqdm.write(f'Epoch: {epoch}, step: {step}, loss: {loss.item():g}, lr: {lr}')
                if step % demo_every == 0: demo()
                if evaluate_enabled and step > 0 and step % evaluate_every == 0: evaluate()
                if step > 0 and step % save_every == 0: save()
                step += 1
            epoch += 1
    except KeyboardInterrupt: pass

