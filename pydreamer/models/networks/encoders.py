from typing import Optional, Union
import torch
import torch.nn as nn
import torch.distributions as D

from .. math_functions import *
from .common import *
from .. import tools_v3


class MultiEncoder_v2(nn.Module):

    def __init__(self, shapes,conf):
        super().__init__()
        self.wm_type=conf.wm_type
        self.reward_input = conf.reward_input
        

        # if conf.image_encoder == 'cnn':
        #     self.encoder_image = ConvEncoder(in_channels=encoder_channels,
        #                                      cnn_depth=conf.cnn_depth)
        # elif conf.image_encoder == 'dense':
        #     if conf.wm_type=='v2':
        #         self.encoder_image = DenseEncoder(in_dim=conf.image_size * conf.image_size * encoder_channels,
        #                                       out_dim=256,
        #                                       hidden_layers=conf.image_encoder_layers,
        #                                       layer_norm=conf.layer_norm)
        # elif not conf.image_encoder:
        #     self.encoder_image = None
        # else:
        #     assert False, conf.image_encoder
            
        #     # vecons_size=0
        # if conf.vecobs_size:
        #     self.encoder_vecobs = MLP_v2(conf.vecobs_size, 256, hidden_dim=400, hidden_layers=2, layer_norm=conf.layer_norm)
        # else:
        #     self.encoder_vecobs = None

        # assert self.encoder_image or self.encoder_vecobs, "Either image_encoder or vecobs_size should be set"
        # self.out_dim = ((self.encoder_image.out_dim if self.encoder_image else 0) +
        #                 (self.encoder_vecobs.out_dim if self.encoder_vecobs else 0))
        
        excluded = ("reset", "is_last", "terminal", "reward")
        shapes = {
            k: v
            for k, v in shapes.items()
            if k not in excluded and not k.startswith("log_")
        }
        self.cnn_shapes = {
            k: v for k, v in shapes.items() if len(v) == 3 and re.match(conf.encoder["cnn_keys"], k)
        }
        self.mlp_shapes = {
            k: v
            for k, v in shapes.items()
            if len(v) in (1, 2) and re.match(conf.encoder["mlp_keys"], k)
        }
        print("Encoder CNN shapes:", self.cnn_shapes)
        print("Encoder MLP shapes:", self.mlp_shapes)
        
        if conf.image_encoder == 'cnn':
            input_ch = sum([v[-1] for v in self.cnn_shapes.values()])
            if conf.reward_input:
                    input_ch+=2  # + reward, terminal
            # else:
            #         encoder_channels = conf.image_channels
            input_shape = tuple(self.cnn_shapes.values())[0][:2] + (input_ch,)
            self.encoder_image = ConvEncoder(
                input_shape, self.wm_type,conf.encoder["cnn_depth"], conf.encoder["act"], conf.encoder["norm"], conf.encoder["kernel_size"], 
                conf.encoder["minres"]
            )
            # self.encoder_image = ConvEncoder(in_channels=encoder_channels,
            #                                  cnn_depth=conf.cnn_depth)
        
        elif conf.image_encoder == 'dense':
            input_size = sum([sum(v) for v in self.mlp_shapes.values()])
            self.encoder_image = DenseEncoder(in_dim=input_size,
                                            out_dim=256,
                                            hidden_layers=conf.image_encoder_layers,
                                            layer_norm=conf.layer_norm)
        elif not conf.image_encoder:
            self.encoder_image = None
        else:
            assert False, conf.image_encoder
            
            # vecons_size=0
        if conf.vecobs_size:
            self.encoder_vecobs = MLP_v2(conf.vecobs_size, 256, hidden_dim=400, hidden_layers=2, layer_norm=conf.layer_norm)
        else:
            self.encoder_vecobs = None

        assert self.encoder_image or self.encoder_vecobs, "Either image_encoder or vecobs_size should be set"
        self.out_dim = ((self.encoder_image.out_dim if self.encoder_image else 0) +
                        (self.encoder_vecobs.out_dim if self.encoder_vecobs else 0))
        

    def forward(self, obs: Dict[str, Tensor]) -> TensorTBE:
        # TODO:
        #  1) Make this more generic, e.g. working without image input or without vecobs
        #  2) Treat all inputs equally, adding everything via linear layer to embed_dim

        embeds = []

        if self.encoder_image:
            image = obs['image']
            T, B, C, H, W = image.shape
            if self.reward_input:
                reward = obs['reward']
                terminal = obs['terminal']
                reward_plane = reward.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand((T, B, 1, H, W))
                terminal_plane = terminal.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand((T, B, 1, H, W))
                image = torch.cat([image,  # (T,B,C+2,H,W)
                                reward_plane.to(image.dtype),
                                terminal_plane.to(image.dtype)], dim=-3)

            embed_image = self.encoder_image.forward(image)  # (T,B,E)
            embeds.append(embed_image)

        if self.encoder_vecobs:
            embed_vecobs = self.encoder_vecobs(obs['vecobs'])
            embeds.append(embed_vecobs)

        embed = torch.cat(embeds, dim=-1)  # (T,B,E+256)
        return embed
    
class MultiEncoder_v3(nn.Module):
    def __init__(
        self,
        shapes,
        conf,
    ):
        super(MultiEncoder_v3, self).__init__()
        self.wm_type=conf.wm_type
        
        
        excluded = ("reset", "is_last", "terminal", "reward")
        shapes = {
            k: v
            for k, v in shapes.items()
            if k not in excluded and not k.startswith("log_")
        }
        self.cnn_shapes = {
            k: v for k, v in shapes.items() if len(v) == 3 and re.match(conf.encoder["cnn_keys"], k)
        }
        self.mlp_shapes = {
            k: v
            for k, v in shapes.items()
            if len(v) in (1, 2) and re.match(conf.encoder["mlp_keys"], k)
        }
        print("Encoder CNN shapes:", self.cnn_shapes)
        print("Encoder MLP shapes:", self.mlp_shapes)

        self.out_dim = 0
        if self.cnn_shapes:
            input_ch = sum([v[-1] for v in self.cnn_shapes.values()])
            input_shape = tuple(self.cnn_shapes.values())[0][:2] + (input_ch,)
            self._cnn = ConvEncoder(
                input_shape, self.wm_type,conf.encoder["cnn_depth"], conf.encoder["act"], conf.encoder["norm"], conf.encoder["kernel_size"], 
                conf.encoder["minres"]
            )
            self.out_dim += self._cnn.out_dim
        if self.mlp_shapes:
            input_size = sum([sum(v) for v in self.mlp_shapes.values()])
            self._mlp = MLP_v3(
                input_size,
                None,
               conf.encoder["mlp_layers"],
                conf.encoder["mlp_units"],
                conf.encoder["act"],
                conf.encoder["norm"],
                symlog_inputs=conf.encoder["symlog_inputs"],
            )
            self.out_dim += conf.encoder["mlp_units"]

    def forward(self, obs):
        outputs = []
        if self.cnn_shapes:
            inputs = torch.cat([obs[k] for k in self.cnn_shapes], -1)
            outputs.append(self._cnn(inputs))
        if self.mlp_shapes:
            inputs = torch.cat([obs[k] for k in self.mlp_shapes], -1)
            outputs.append(self._mlp(inputs))
        outputs = torch.cat(outputs, -1)
        return outputs


class ConvEncoder(nn.Module):

    def __init__(self, input_shape, wm_type,cnn_depth=32, act="SiLU",norm="LayerNorm",minres=4,kernel_size=4):
        super().__init__()
        self.wm_type=wm_type
        # if wm_type=="v2":
        h,w,input_ch= input_shape
        activation=nn.ELU
        self.out_dim = cnn_depth * 32
        
        stride = 2
        #中间层的channer数
        d = cnn_depth
        self.layers = nn.Sequential(
            nn.Conv2d(input_ch, d, kernel_size, stride),
            activation(),
            nn.Conv2d(d, d * 2, kernel_size, stride),
            activation(),
            nn.Conv2d(d * 2, d * 4, kernel_size, stride),
            activation(),
            nn.Conv2d(d * 4, d * 8, kernel_size, stride),
            activation(),
            nn.Flatten()
        )
        # elif wm_type=="v3":
        #     h, w, input_ch = input_shape
        #     act = getattr(torch.nn, act)
        #     norm = getattr(torch.nn, norm)
        #     layers = []
        #     for i in range(int(np.log2(h) - np.log2(minres))):
        #         if i == 0:
        #             in_dim = input_ch
        #         else:
        #             in_dim = 2 ** (i - 1) * cnn_depth
        #         out_dim = 2**i * cnn_depth
        #         layers.append(
        #             Conv2dSame(
        #                 in_channels=in_dim,
        #                 out_channels=out_dim,
        #                 kernel_size=kernel_size,
        #                 stride=2,
        #                 bias=False,
        #             )
        #         )
        #         layers.append(ChLayerNorm(out_dim))
        #         layers.append(act())
        #         h, w = h // 2, w // 2

            # self.out_dim = out_dim * h * w
            # self.layers = nn.Sequential(*layers)
            # self.layers.apply(tools_v3.weight_init)

    def forward(self, x):
        # if self.wm_type=='v2':
            #（T,B,C,H,W）→（B*T,C,H,W)
            x, bd = flatten_batch(x, 3)
            y = self.layers(x)
            #（B*T,C,H,W）→（T,B,C,H,W)
            y = unflatten_batch(y, bd)
            return y
        # elif self.wm_type=='v3':
        #     # # (batch, time, h, w, ch) -> (batch * time, h, w, ch)
        #     # x = x.reshape((-1,) + tuple(x.shape[-3:]))
        #     # # (batch * time, h, w, ch) -> (batch * time, ch, h, w)
        #     # # 因为数据来源于v2，所以已经是ch,h,w了
        #     # # x = x.permute(0, 3, 1, 2)
        #     # x = self.layers(x)
        #     # # (batch * time, ...) -> (batch * time, -1)
        #     # x = x.reshape([x.shape[0], np.prod(x.shape[1:])])
        #     # # (batch * time, -1) -> (batch, time, -1)
        #     # return x.reshape(list(x.shape[:-3]) + [x.shape[-1]])
        #       #（B,T,C,H,W）→（B*T,C,H,W)
        #     x, bd = flatten_batch(x, 3)
        #     y = self.layers(x)
        #     #（B*T,C,H,W）→（B,T,C,H,W)
        #     y = unflatten_batch(y, bd)
        #     return y.reshape(y.shape[0], y.shape[1], -1)

## Add a new symlog for MLP encoder
class DenseEncoder(nn.Module):

    def __init__(self, in_dim, out_dim=256, activation=nn.ELU, hidden_dim=400, hidden_layers=2, layer_norm=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        norm = nn.LayerNorm if layer_norm else NoNorm
        layers = [nn.Flatten()]
        layers += [
            nn.Linear(in_dim, hidden_dim),
            norm(hidden_dim, eps=1e-3),
            activation()]
        for _ in range(hidden_layers - 1):
            layers += [
                nn.Linear(hidden_dim, hidden_dim),
                norm(hidden_dim, eps=1e-3),
                activation()]
        layers += [
            nn.Linear(hidden_dim, out_dim),
            activation()]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x, bd = flatten_batch(x, 3)
        y = self.model(x)
        y = unflatten_batch(y, bd)
        return y
