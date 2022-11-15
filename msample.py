#!/usr/bin/env python3
"""Samples from k-diffusion models."""

import argparse, math, accelerate, torch, k_diffusion as K
from fastcore.script import call_parse
from tqdm import trange, tqdm
from torchvision import utils

@call_parse
def main(
    config: str, # model config
    checkpoint: str, # checkpoint to use
    batch_size:int=64, # batch size
    n:int=64, # number of images to sample
    out:str='out', # output file name without extension
    steps:int=50, # number of denoising steps
    seed:int=0,  # random seed
    churn:float=0.,  # sampler churn
    sampler:str='sample_lms',  # sample_lms, sample_dpm_2, sample_euler, etc
):
    sampler = getattr(K.sampling, sampler)
    if seed: torch.manual_seed(seed)
    config = K.config.load_config(open(config))
    model_config = config['model']
    size = model_config['input_size']
    accelerator = accelerate.Accelerator()
    device = accelerator.device
    inner_model = K.config.make_model(config).eval().requires_grad_(False).to(device)
    inner_model.load_state_dict(torch.load(checkpoint, map_location='cpu')['model_ema'])
    model = K.config.make_denoiser_wrapper(config)(inner_model)

    sigma_max = model_config['sigma_max']
    sigmas = K.sampling.get_sigmas_karras(steps, model_config['sigma_min'], sigma_max, rho=7., device=device)
    def sample_fn(n):
        x = torch.randn([n, model_config['input_channels'], *size], device=device) * sigma_max
        return sampler(model, x, sigmas, **{'s_churn':churn} if churn else {})
    x_0 = K.evaluation.compute_features(accelerator, sample_fn, lambda x: x, n, batch_size)
    grid = utils.make_grid(x_0, nrow=math.ceil(n ** 0.5), padding=0)
    K.utils.to_pil_image(grid).save(f'{out}.png')

