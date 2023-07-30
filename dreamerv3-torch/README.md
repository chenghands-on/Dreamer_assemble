# dreamerv3-torch
Pytorch implementation of [Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104v1). DreamerV3 is a scalable algorithm that outperforms previous approaches across various domains with fixed hyperparameters.

## Instructions

Get dependencies:
```
pip install -r requirements.txt
```
Run training on DMC Vision:
```
python3 dreamer.py --configs dmc_vision --task dmc_walker_walk --logdir ./logdir/dmc_walker_walk
```
Monitor results:
```
tensorboard --logdir ./logdir
```

## Benchmarks
So far, the following benchmarks can be used for testing.
| Environment        | Observation | Action | Budget | Description |
|-------------------|---|---|---|-----------------------|
| [DMC Proprio](https://github.com/deepmind/dm_control) | State | Continuous | 500K | DeepMind Control Suite with low-dimensional inputs. |
| [DMC Vision](https://github.com/deepmind/dm_control) | Image | Continuous |1M| DeepMind Control Suite with high-dimensional images inputs. |
| [Atari 100k](https://github.com/openai/atari-py) | Image | Discrete |400K| 26 Atari games. |
| [Crafter](https://github.com/danijar/crafter) | Image | Discrete |1M| Survival environment to evaluates diverse agent abilities.|
| [Minecraft](https://github.com/minerllabs/minerl) | Image and State |Discrete |100M| Vast 3D open world.|
| [Memory Maze](https://github.com/jurgisp/memory-maze) | Image |Discrete |100M| 3D mazes to evaluate RL agents' long-term memory.|

## Results
#### DMC Proprio
![dmcproprio](https://github.com/NM512/dreamerv3-torch/assets/70328564/7f6e47a5-3235-4bc4-bef9-15ff96782d5e)
#### DMC Vision
![dmcvision](https://github.com/NM512/dreamerv3-torch/assets/70328564/b710d217-2428-4fa0-8471-55e15ec5aa43)
#### Atari 100k
![atari100k](https://github.com/NM512/dreamerv3-torch/assets/70328564/0da6d899-d91d-44b4-a8c4-d5b37413aa11)

## Acknowledgments
This code is heavily inspired by the following works:
- danijar's Dreamer-v3 jax implementation: https://github.com/danijar/dreamerv3
- danijar's Dreamer-v2 tensorflow implementation: https://github.com/danijar/dreamerv2
- jsikyoon's Dreamer-v2 pytorch implementation: https://github.com/jsikyoon/dreamer-torch
- RajGhugare19's Dreamer-v2 pytorch implementation: https://github.com/RajGhugare19/dreamerv2
- denisyarats's DrQ-v2 original implementation: https://github.com/facebookresearch/drqv2
