import argparse
import json
import os
import time
from logging import info
from distutils.util import strtobool
from multiprocessing import Process
from typing import List
import torch
from pydreamer.models import Dreamer

import generator
import train
from pydreamer.tools import (configure_logging, mlflow_log_params,
                             mlflow_init, print_once, read_yamls)
import time
from collections import defaultdict
from datetime import datetime
from logging import critical, debug, error, info, warning
from typing import Iterator, Optional

import mlflow
import numpy as np
import scipy.special
import torch
import numpy as np
from torch import Tensor
from torch.cuda.amp import GradScaler, autocast
from torch.profiler import ProfilerActivity
from torch.utils.data import DataLoader

from pydreamer import tools
from pydreamer.data import DataSequential, MlflowEpisodeRepository
from pydreamer.models import *
from pydreamer.models.functions import map_structure, nanmean
from pydreamer.preprocessing import Preprocessor, WorkerInfoPreprocess
from pydreamer.tools import *
from make_gif import make_gif_wm

def make_args():
    parser = argparse.ArgumentParser(description="argument parser", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--artifact_uri", required=False, type=str, default="/home/chenghan/pydreamer/mlruns/0/c1d429acbbff43afb0e2edd18d11ebce/artifacts")
    parser.add_argument("--action_type", default='fixed_online', type=str)
    parser.add_argument("--index", default=0, type=int)
    parser.add_argument('--configs', nargs='+', required=True)

    args, remaining = parser.parse_known_args()
    # print(args)
    # print(remaining)
    conf = {
        "artifact_uri": args.artifact_uri,
        "action_type": args.action_type,
        "index": args.index,
    } ##所有参数的集合
    configs = read_yamls('./config')
    for name in args.configs:
        if ',' in name:
            for n in name.split(','):
                conf.update(configs[n])
        else:
            conf.update(configs[name])

    # Override config from command-line，覆盖掉yaml中的设置

    parser = argparse.ArgumentParser(description="argument parser", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('--configs', nargs='+', required=True)
    for key, value in conf.items():
        type_ = type(value) if value is not None else str
        if type_ == bool:
            type_ = lambda x: bool(strtobool(x))
        parser.add_argument(f'--{key}', type=type_, default=value)
    
    conf = parser.parse_args(remaining)

    return conf



def prepare_batch_npz(data: Dict[str, Tensor], take_b=999):

    def unpreprocess(key: str, val: Tensor) -> np.ndarray:
        if take_b < val.shape[1]:
            val = val[:, :take_b]

        x = val.detach().cpu().numpy()  # (T,B,*)
        if x.dtype in [np.float16, np.float64]:
            x = x.astype(np.float32)

        if len(x.shape) == 2:  # Scalar
            pass

        elif len(x.shape) == 3:  # 1D vector
            pass

        elif len(x.shape) == 4:  # 2D tensor
            pass

        elif len(x.shape) == 5:  # 3D tensor - image
            assert x.dtype == np.float32 and (key.startswith('image') or key.startswith('map')), \
                f'Unexpected 3D tensor: {key}: {x.shape}, {x.dtype}'

            if x.shape[-1] == x.shape[-2]:  # (T,B,C,W,W)
                x = x.transpose(0, 1, 3, 4, 2)  # => (T,B,W,W,C)
            assert x.shape[-2] == x.shape[-3], 'Assuming rectangular images, otherwise need to improve logic'

            if x.shape[-1] in [1, 3]:
                # RGB or grayscale
                x = ((x + 0.5) * 255.0).clip(0, 255).astype('uint8')
            elif np.allclose(x.sum(axis=-1), 1.0) and np.allclose(x.max(axis=-1), 1.0):
                # One-hot
                x = x.argmax(axis=-1)
            else:
                # Categorical logits
                assert key in ['map_rec', 'image_rec', 'image_pred'], \
                    f'Unexpected 3D categorical logits: {key}: {x.shape}'
                x = scipy.special.softmax(x, axis=-1)

        x = x.swapaxes(0, 1)  # type: ignore  # (T,B,*) => (B,T,*)
        return x

    return {k: unpreprocess(k, v) for k, v in data.items()}

# def test_dream(model,obs,states,action_type):
#     with torch.no_grad():  # careful not to invoke modules first time under no_grad (https://github.com/pytorch/pytorch/issues/60164)
#         # The reason we don't just take real features_dream is because it's really big (H*T*B*I),
#         # and here for inspection purposes we only dream from first step, so it's (H*B).
#         # Oh, and we set here H=T-1, so we get (T,B), and the dreamed experience aligns with actual.
#         # 这里实际做的时候，T=1，只从第一步想象
#         in_state_dream=map_structure(states, lambda x: x.detach()[0, :, 0])  # type: ignore  # (T,B,I) => (B)
#         ## 基本上只改了这一步
#         # non_zero_indices = torch.nonzero(in_state_dream[1])
#         if action_type=='fixed_online':
#             features_dream, actions_dream, rewards_dream, terminals_dream = model.dream_cond_action(in_state_dream, obs['action'])
#             image_dream = model.wm.decoder.image.forward(features_dream)
#             dream_tensors = dict(action_pred=actions_dream,  # first action is real from previous step
#                                     reward_pred=rewards_dream.mean,
#                                     terminal_pred=terminals_dream.mean,
#                                     image_pred=image_dream,)

#         elif action_type=='adapt_online':
#             features_dream, actions_dream, rewards_dream, terminals_dream = model.dream(in_state_dream, obs['action'].shape[0] - 1)  # H = T-1
#             image_dream = model.wm.decoder.image.forward(features_dream)
#             dream_tensors = dict(action_pred=torch.cat([obs['action'][:1], actions_dream]),  # first action is real from previous step
#                              reward_pred=rewards_dream.mean,
#                              terminal_pred=terminals_dream.mean,
#                              image_pred=image_dream,
#                              )
#         elif action_type=='disturb_online':
#             features_dream, actions_dream, rewards_dream, terminals_dream = model.dream(in_state_dream, obs['action'].shape[0] - 1,perturb='1')  # H = T-1
#             image_dream = model.wm.decoder.image.forward(features_dream)
#             dream_tensors = dict(action_pred=torch.cat([obs['action'][:1], actions_dream]),  # first action is real from previous step
#                              reward_pred=rewards_dream.mean,
#                              terminal_pred=terminals_dream.mean,
#                              image_pred=image_dream,
#                              )
#         elif action_type=='offline':
#             features_dream, actions_dream, rewards_dream, terminals_dream = model.dream(in_state_dream, obs['action'].shape[0] - 1,perturb='2')  # H = T-1
#             image_dream = model.wm.decoder.image.forward(features_dream)
#             dream_tensors = dict(action_pred=torch.cat([obs['action'][:1], actions_dream]),  # first action is real from previous step
#                              reward_pred=rewards_dream.mean,
#                              terminal_pred=terminals_dream.mean,
#                              image_pred=image_dream,
#                              )
#         assert dream_tensors['action_pred'].shape == obs['action'].shape
#         assert dream_tensors['image_pred'].shape == obs['image'].shape
#         return dream_tensors

def test_dream(model, obs, states, action_type):
    # with torch.no_grad():
    in_state_dream = map_structure(states, lambda x: x.detach()[0, :, 0])

    # Process fixed_online case separately
    if action_type == 'fixed_online':
        features_dream, actions_dream, rewards_dream, terminals_dream = model.dream_cond_action(
            in_state_dream, obs['action'])
    elif action_type =='deter_offline':
        action=torch.zeros_like(obs['action'])
        action[:,:,2]=1
        features_dream, actions_dream, rewards_dream, terminals_dream = model.dream_cond_action(
            in_state_dream, action)
    else:
        # Set perturb according to action_type
        if action_type == 'disturb_online':
            perturb = 'guassian'
        elif action_type == 'offline':
            perturb = 'random_policy'
        ## 扰动的大小不同
        elif action_type == 'attack_online_s':
            perturb = 'attack_s'
        elif action_type == 'attack_online_m':
            perturb = 'attack_m'
        elif action_type == 'attack_online_l':
            perturb = 'attack_l'
        else:
            perturb = 'None'

        # Perform dreaming
        features_dream, actions_dream, rewards_dream, terminals_dream = model.dream(
            in_state_dream, obs['action'].shape[0] - 1, perturb=perturb)

    image_dream = model.wm.decoder.image.forward(features_dream)

    # Prepend the real action from the previous step if necessary
    if action_type in ['adapt_online', 'disturb_online', 'offline', 'attack_online_s','attack_online_m','attack_online_l']:
        actions_dream = torch.cat([obs['action'][:1], actions_dream])

    dream_tensors = dict(
        action_pred=actions_dream,
        reward_pred=rewards_dream.mean,
        terminal_pred=terminals_dream.mean,
        image_pred=image_dream,
    )

    assert dream_tensors['action_pred'].shape == obs['action'].shape
    assert dream_tensors['image_pred'].shape == obs['image'].shape

    return dream_tensors


def process_dream_tensors(model, obs, states, action_type, index, conf):
    dream_tensors = test_dream(model, obs, states, action_type)
    dream_tensors_cpu = {key: dream_tensor.cpu() for key, dream_tensor in dream_tensors.items()}
    dream_tensors_cpu = prepare_batch_npz(dream_tensors_cpu)
    np.savez(f'wm_results/dream_{action_type}_{conf.env_id}_{index}_data.npz', **dream_tensors_cpu)
    make_gif_wm(conf.env_id,f'wm_results/dream_{action_type}_{conf.env_id}_{index}_data.npz',index=index,action_type=action_type,dream=True)        

def main(conf):
    # Add Some new parameter

    artifact_uri=conf.artifact_uri
    index=conf.index
    action_type=conf.action_type

    device = torch.device(conf.device)

    # 创建模型
    model = Dreamer(conf)
    model.to(device)
    print(device)
    # 加载模型参数
    checkpoint = torch.load(f'{artifact_uri}/checkpoints/latest.pt',map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])

    # # 打印模型参数
    # print("Model Parameters:")
    # for name, param in model.named_parameters():
    #     print(f"{name}: {param.shape}")

    # # # 打印模型结构
    # print("\nModel Architecture:")
    # print(model)

    ## data reader
    input_dirs = [
                f'{artifact_uri}/episodes/{i}'
                for i in range(max(conf.generator_workers_train, conf.generator_workers))
            ]
    # print(input_dirs)
    data = DataSequential(MlflowEpisodeRepository(input_dirs),
                            conf.batch_length,
                            conf.batch_size,
                            skip_first=True,
                            reload_interval=120 ,
                            buffer_size=conf.buffer_size ,
                            reset_interval=conf.reset_interval,
                            allow_mid_reset=conf.allow_mid_reset)
    # data=data.to(device)
    preprocess = Preprocessor(image_categorical=conf.image_channels if conf.image_categorical else None,
                                image_key=conf.image_key,
                                map_categorical=conf.map_channels if conf.map_categorical else None,
                                map_key=conf.map_key,
                                action_dim=conf.action_dim,
                                clip_rewards=conf.clip_rewards,
                                amp=conf.amp and device.type == 'cuda')

    data_iter = iter(DataLoader(WorkerInfoPreprocess(preprocess(data)),
                                    batch_size=None,
                                    num_workers=conf.data_workers,
                                    # num_workers=1,
                                    prefetch_factor=20 if conf.data_workers else 2,  # GCS download has to be shorter than this many batches (e.g. 1sec < 20*300ms)
                                    pin_memory=True))
    # data_iter=data_iter.to(device)
    states={}
    batch, wid = next(data_iter)
    obs: Dict[str, Tensor] = map_structure(batch, lambda x: x.to(device))
    obs_cpu = {key: tensor.cpu() for key, tensor in obs.items()}
    obs_cpu=prepare_batch_npz(obs_cpu)
    np.savez(f'wm_results/origin_{conf.env_id}_{index}_data.npz', **obs_cpu)
    make_gif_wm(conf.env_id,f'wm_results/origin_{conf.env_id}_{index}_data.npz',index=index,action_type=action_type,dream=False)


    # Get the starting states(h,z)
    in_state = states.get(wid)
    if in_state is None:
        in_state = model.init_state(conf.batch_size * conf.iwae_samples)
    loss, features, states, out_state, metrics, tensors = \
                model.wm.training_step(obs, in_state, forward_only=True)
   

    # action_types = ['fixed_online', 'adapt_online', 'disturb_online','attack_online','offline','deter_offline','all']
    action_types = ['attack_online_s','attack_online_m','attack_online_l','deter_offline','fixed_online', 'adapt_online', 'disturb_online','offline','all']
    # action_types=['adapt_online','attack_online_s','attack_online_m','attack_online_l','all']
    
    if action_type in action_types[:len(action_types)-1]: # if action_type is 'fixed_online' or 'adapt_online'
        process_dream_tensors(model, obs, states, action_type, index, conf)
    elif action_type == 'all': 
        for act_type in action_types[:len(action_types)-1]: # act_type will be 'fixed_online' and then 'adapt_online'
            
            process_dream_tensors(model, obs, states, act_type, index, conf)




if __name__ == "__main__":
    conf = make_args()
    main(conf)
