from typing import Optional, Union
import torch
import torch.nn as nn
import torch.distributions as D

from ..math_functions import *
from .common import *
from .. import tools_v3


class MultiDecoder(nn.Module):

    def __init__(self, features_dim, conf):
        super().__init__()
        self.image_weight = conf.image_weight
        self.vecobs_weight = conf.vecobs_weight
        self.reward_weight = conf.reward_weight
        self.terminal_weight = conf.terminal_weight

        if conf.image_decoder == 'cnn':
            self.image = ConvDecoder(in_dim=features_dim,
                                     out_channels=conf.image_channels,
                                     cnn_depth=conf.cnn_depth)
        elif conf.image_decoder == 'dense':
            self.image = CatImageDecoder(in_dim=features_dim,
                                         out_shape=(conf.image_channels, conf.image_size, conf.image_size),
                                         hidden_layers=conf.image_decoder_layers,
                                         layer_norm=conf.layer_norm,
                                         min_prob=conf.image_decoder_min_prob)
        elif not conf.image_decoder:
            self.image = None
        else:
            assert False, conf.image_decoder
        if conf.wm_type=='v2':
            if conf.reward_decoder_categorical:
                self.reward = DenseCategoricalSupportDecoder(
                    in_dim=features_dim,
                    support=clip_rewards_np(conf.reward_decoder_categorical, conf.clip_rewards),  # reward_decoder_categorical values are untransformed 
                    hidden_layers=conf.reward_decoder_layers,
                    layer_norm=conf.layer_norm)
            else:
                self.reward = DenseNormalDecoder(in_dim=features_dim, hidden_layers=conf.reward_decoder_layers, layer_norm=conf.layer_norm)

            self.terminal = DenseBernoulliDecoder(in_dim=features_dim, hidden_layers=conf.terminal_decoder_layers, layer_norm=conf.layer_norm)
        elif conf.wm_type=='v3':
            if conf.reward_head == "symlog_disc":
                self.reward = MLP_v3(
                    features_dim,  # pytorch version
                    (255,),
                    conf.reward_layers,
                    conf.units,
                    conf.act,
                    conf.norm,
                    dist=conf.reward_head,
                    outscale=0.0,
                    device=conf.device,
                )
            else:
                self.reward = MLP_v3(
                    features_dim,  # pytorch version
                    [],
                    conf.reward_layers,
                    conf.units,
                    conf.act,
                    conf.norm,
                    dist=conf.reward_head,
                    outscale=0.0,
                    device=conf.device,
                )
            self.terminal = MLP_v3(
                features_dim,  # pytorch version
                [],
                conf.terminal_layers,
                conf.units,
                conf.act,
                conf.norm,
                dist="binary",
                device=conf.device,
            )

        if conf.vecobs_size:
            self.vecobs = DenseNormalDecoder(in_dim=features_dim, out_dim=conf.vecobs_size, hidden_layers=4, layer_norm=conf.layer_norm)
        else:
            self.vecobs = None

    def training_step(self,
                      features: TensorTBIF,
                      obs: Dict[str, Tensor],
                      extra_metrics: bool = False
                      ) -> Tuple[TensorTBI, Dict[str, Tensor], Dict[str, Tensor]]:
        tensors = {}
        metrics = {}
        loss_reconstr = 0

        if self.image:
            loss_image_tbi, loss_image, image_rec = self.image.training_step(features, obs['image'])
            loss_reconstr += self.image_weight * loss_image_tbi
            metrics.update(loss_image=loss_image.detach().mean())
            tensors.update(loss_image=loss_image.detach(),
                           image_rec=image_rec.detach())

        if self.vecobs:
            loss_vecobs_tbi, loss_vecobs, vecobs_rec = self.vecobs.training_step(features, obs['vecobs'])
            loss_reconstr += self.vecobs_weight * loss_vecobs_tbi
            metrics.update(loss_vecobs=loss_vecobs.detach().mean())
            tensors.update(loss_vecobs=loss_vecobs.detach(),
                           vecobs_rec=vecobs_rec.detach())

        loss_reward_tbi, loss_reward, reward_rec = self.reward.training_step(features, obs['reward'])
        loss_reconstr += self.reward_weight * loss_reward_tbi
        metrics.update(loss_reward=loss_reward.detach().mean())
        tensors.update(loss_reward=loss_reward.detach(),
                       reward_rec=reward_rec.detach())

        loss_terminal_tbi, loss_terminal, terminal_rec = self.terminal.training_step(features, obs['terminal'])
        loss_reconstr += self.terminal_weight * loss_terminal_tbi
        metrics.update(loss_terminal=loss_terminal.detach().mean())
        tensors.update(loss_terminal=loss_terminal.detach(),
                       terminal_rec=terminal_rec.detach())

        if extra_metrics:
            if isinstance(self.reward, DenseCategoricalSupportDecoder):
                # TODO: logic should be moved to appropriate decoder
                reward_cat = self.reward.to_categorical(obs['reward'])
                for i in range(len(self.reward.support)):
                    # Logprobs for specific categorical reward values
                    mask_rewardp = reward_cat == i  # mask where categorical reward has specific value
                    loss_rewardp = loss_reward * mask_rewardp / mask_rewardp  # set to nan where ~mask
                    metrics[f'loss_reward{i}'] = nanmean(loss_rewardp)  # index by support bucket, not by value
                    tensors[f'loss_reward{i}'] = loss_rewardp
            else:
                for sig in [-1, 1]:
                    # Logprobs for positive and negative rewards
                    mask_rewardp = torch.sign(obs['reward']) == sig  # mask where reward is positive or negative
                    loss_rewardp = loss_reward * mask_rewardp / mask_rewardp  # set to nan where ~mask
                    metrics[f'loss_reward{sig}'] = nanmean(loss_rewardp)
                    tensors[f'loss_reward{sig}'] = loss_rewardp

            mask_terminal1 = obs['terminal'] > 0  # mask where terminal is 1
            loss_terminal1 = loss_terminal * mask_terminal1 / mask_terminal1  # set to nan where ~mask
            metrics['loss_terminal1'] = nanmean(loss_terminal1)
            tensors['loss_terminal1'] = loss_terminal1

        return loss_reconstr, metrics, tensors
class MultiDecoder_v3(nn.Module):
    def __init__(
        self,
        feat_size,
        shapes,
        mlp_keys,
        cnn_keys,
        act,
        norm,
        cnn_depth,
        kernel_size,
        minres,
        mlp_layers,
        mlp_units,
        cnn_sigmoid,
        image_dist,
        vector_dist,
    ):
        super().__init__()
        ## image decoder part
        excluded = ("reset", "is_last", "terminal", "reward")
        shapes = {k: v for k, v in shapes.items() if k not in excluded}
        self.cnn_shapes = {
            k: v for k, v in shapes.items() if len(v) == 3 and re.match(cnn_keys, k)
        }
        self.mlp_shapes = {
            k: v
            for k, v in shapes.items()
            if len(v) in (1, 2) and re.match(mlp_keys, k)
        }
        print("Image Decoder CNN shapes:", self.cnn_shapes)
        print("Image Decoder MLP shapes:", self.mlp_shapes)

        if self.cnn_shapes:
            some_shape = list(self.cnn_shapes.values())[0]
            shape = (sum(x[-1] for x in self.cnn_shapes.values()),) + some_shape[:-1]
            self._cnn = ConvDecoder_v3(
                feat_size,
                shape,
                cnn_depth,
                act,
                norm,
                kernel_size,
                minres,
                cnn_sigmoid=cnn_sigmoid,
            )
        if self.mlp_shapes:
            self._mlp = MLP_v3(
                feat_size,
                self.mlp_shapes,
                mlp_layers,
                mlp_units,
                act,
                norm,
                vector_dist,
            )
        self._image_dist = image_dist

    def forward(self, features):
        dists = {}
        if self.cnn_shapes:
            feat = features
            outputs = self._cnn(feat)
            split_sizes = [v[-1] for v in self.cnn_shapes.values()]
            # outputs = torch.split(outputs, split_sizes, -1)
            outputs = torch.split(outputs, split_sizes, -3)
            dists.update(
                {
                    key: self._make_image_dist(output)
                    for key, output in zip(self.cnn_shapes.keys(), outputs)
                }
            )
        if self.mlp_shapes:
            dists.update(self._mlp(features))
        return dists

    def _make_image_dist(self, mean):
        if self._image_dist == "normal":
            return tools_v3.ContDist(
                torchd.independent.Independent(torchd.normal.Normal(mean, 1), 3)
            )
        if self._image_dist == "mse":
            return tools_v3.MSEDist(mean)
        raise NotImplementedError(self._image_dist)

class ConvDecoder(nn.Module):

    def __init__(self,
                 in_dim,
                 out_channels=3,
                 cnn_depth=32,
                 mlp_layers=0,
                 layer_norm=True,
                 activation=nn.ELU
                 ):
        super().__init__()
        self.in_dim = in_dim
        kernels = (5, 5, 6, 6)
        stride = 2
        d = cnn_depth
        if mlp_layers == 0:
            layers = [
                nn.Linear(in_dim, d * 32),  # No activation here in DreamerV2
            ]
        else:
            hidden_dim = d * 32
            norm = nn.LayerNorm if layer_norm else NoNorm
            layers = [
                nn.Linear(in_dim, hidden_dim),
                norm(hidden_dim, eps=1e-3),
                activation()
            ]
            for _ in range(mlp_layers - 1):
                layers += [
                    nn.Linear(hidden_dim, hidden_dim),
                    norm(hidden_dim, eps=1e-3),
                    activation()]

        self.model = nn.Sequential(
            # FC
            *layers,
            nn.Unflatten(-1, (d * 32, 1, 1)),
            # Deconv
            nn.ConvTranspose2d(d * 32, d * 4, kernels[0], stride),
            activation(),
            nn.ConvTranspose2d(d * 4, d * 2, kernels[1], stride),
            activation(),
            nn.ConvTranspose2d(d * 2, d, kernels[2], stride),
            activation(),
            nn.ConvTranspose2d(d, out_channels, kernels[3], stride))

    def forward(self, x: Tensor) -> Tensor:
        x, bd = flatten_batch(x)
        y = self.model(x)
        y = unflatten_batch(y, bd)
        return y

    def loss(self, output: Tensor, target: Tensor) -> Tensor:
        output, bd = flatten_batch(output, 3)
        target, _ = flatten_batch(target, 3)
        loss = 0.5 * torch.square(output - target).sum(dim=[-1, -2, -3])  # MSE
        return unflatten_batch(loss, bd)

    def training_step(self, features: TensorTBIF, target: TensorTBCHW) -> Tuple[TensorTBI, TensorTB, TensorTBCHW]:
        assert len(features.shape) == 4 and len(target.shape) == 5
        I = features.shape[2]
        target = insert_dim(target, 2, I)  # Expand target with iwae_samples dim, because features have it

        decoded = self.forward(features)
        loss_tbi = self.loss(decoded, target)
        loss_tb = -logavgexp(-loss_tbi, dim=2)  # TBI => TB
        decoded = decoded.mean(dim=2)  # TBICHW => TBCHW

        assert len(loss_tbi.shape) == 3 and len(decoded.shape) == 5
        return loss_tbi, loss_tb, decoded
class ConvDecoder_v3(nn.Module):
    def __init__(
        self,
        feat_size,
        shape=(3, 64, 64),
        depth=32,
        act=nn.ELU,
        norm=nn.LayerNorm,
        kernel_size=4,
        minres=4,
        outscale=1.0,
        cnn_sigmoid=False,
    ):
        super(ConvDecoder_v3, self).__init__()
        act = getattr(torch.nn, act)
        norm = getattr(torch.nn, norm)
        self._shape = shape
        self._cnn_sigmoid = cnn_sigmoid
        layer_num = int(np.log2(shape[1]) - np.log2(minres))
        self._minres = minres
        self._embed_size = minres**2 * depth * 2 ** (layer_num - 1)

        self._linear_layer = nn.Linear(feat_size, self._embed_size)
        self._linear_layer.apply(tools_v3.weight_init)
        in_dim = self._embed_size // (minres**2)

        layers = []
        h, w = minres, minres
        for i in range(layer_num):
            out_dim = self._embed_size // (minres**2) // (2 ** (i + 1))
            bias = False
            initializer = tools_v3.weight_init
            if i == layer_num - 1:
                out_dim = self._shape[0]
                act = False
                bias = True
                norm = False
                initializer = tools_v3.uniform_weight_init(outscale)

            if i != 0:
                in_dim = 2 ** (layer_num - (i - 1) - 2) * depth
            pad_h, outpad_h = self.calc_same_pad(k=kernel_size, s=2, d=1)
            pad_w, outpad_w = self.calc_same_pad(k=kernel_size, s=2, d=1)
            layers.append(
                nn.ConvTranspose2d(
                    in_dim,
                    out_dim,
                    kernel_size,
                    2,
                    padding=(pad_h, pad_w),
                    output_padding=(outpad_h, outpad_w),
                    bias=bias,
                )
            )
            if norm:
                layers.append(ChLayerNorm(out_dim))
            if act:
                layers.append(act())
            [m.apply(initializer) for m in layers[-3:]]
            h, w = h * 2, w * 2

        self.layers = nn.Sequential(*layers)

    def calc_same_pad(self, k, s, d):
        val = d * (k - 1) - s + 1
        pad = math.ceil(val / 2)
        outpad = pad * 2 - val
        return pad, outpad

    def forward(self, features, dtype=None):
        x = self._linear_layer(features)
        # (batch, time, -1) -> (batch * time, h, w, ch)
        x = x.reshape(
            [-1, self._minres, self._minres, self._embed_size // self._minres**2]
        )
        # (batch, time, -1) -> (batch * time, ch, h, w)
        x = x.permute(0, 3, 1, 2)
        x = self.layers(x)
        # (batch, time, -1) -> (batch * time, ch, h, w) necessary???
        mean = x.reshape(features.shape[:-1] + self._shape)
        # (batch * time, ch, h, w) -> (batch * time, h, w, ch)
        # mean = mean.permute(0, 1, 3, 4, 2)
        if self._cnn_sigmoid:
            mean = F.sigmoid(mean) - 0.5
        return mean


class CatImageDecoder(nn.Module):
    """Dense decoder for categorical image, e.g. map"""

    def __init__(self, in_dim, out_shape=(33, 7, 7), activation=nn.ELU, hidden_dim=400, hidden_layers=2, layer_norm=True, min_prob=0):
        super().__init__()
        self.in_dim = in_dim
        self.out_shape = out_shape
        norm = nn.LayerNorm if layer_norm else NoNorm
        layers = []
        if hidden_layers >= 1:
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
                nn.Linear(hidden_dim, np.prod(out_shape)),
                nn.Unflatten(-1, out_shape)]
        else:  
            # hidden_layers == 0
            layers += [
                nn.Linear(in_dim, np.prod(out_shape)),
                nn.Unflatten(-1, out_shape)]
        self.model = nn.Sequential(*layers)
        self.min_prob = min_prob

    def forward(self, x: Tensor) -> Tensor:
        x, bd = flatten_batch(x)
        y = self.model(x)
        y = unflatten_batch(y, bd)
        return y

    def loss(self, output: Tensor, target: Tensor) -> Tensor:
        if len(output.shape) == len(target.shape):
            target = target.argmax(dim=-3)  # float(*,C,H,W) => int(*,H,W)
        assert target.dtype == torch.int64, 'Target should be categorical'
        output, bd = flatten_batch(output, len(self.out_shape))     # (*,C,H,W) => (B,C,H,W)
        target, _ = flatten_batch(target, len(self.out_shape) - 1)  # (*,H,W) => (B,H,W)

        if self.min_prob == 0:
            loss = F.nll_loss(F.log_softmax(output, 1), target, reduction='none')  # = F.cross_entropy()
        else:
            prob = F.softmax(output, 1)
            prob = (1.0 - self.min_prob) * prob + self.min_prob * (1.0 / prob.size(1))  # mix with uniform prob
            loss = F.nll_loss(prob.log(), target, reduction='none')

        if len(self.out_shape) == 3:
            loss = loss.sum(dim=[-1, -2])  # (*,H,W) => (*)
        assert len(loss.shape) == 1
        return unflatten_batch(loss, bd)

    def training_step(self, features: TensorTBIF, target: TensorTBCHW) -> Tuple[TensorTBI, TensorTB, TensorTBCHW]:
        assert len(features.shape) == 4 and len(target.shape) == 5
        I = features.shape[2]
        target = insert_dim(target, 2, I)  # Expand target with iwae_samples dim, because features have it

        logits = self.forward(features)
        loss_tbi = self.loss(logits, target)
        loss_tb = -logavgexp(-loss_tbi, dim=2)  # TBI => TB

        assert len(logits.shape) == 6   # TBICHW
        logits = logits - logits.logsumexp(dim=-3, keepdim=True)  # normalize C
        logits = torch.logsumexp(logits, dim=2)  # aggregate I => TBCHW
        logits = logits - logits.logsumexp(dim=-3, keepdim=True)  # normalize C again
        decoded = logits

        assert len(loss_tbi.shape) == 3 and len(decoded.shape) == 5
        return loss_tbi, loss_tb, decoded


class DenseBernoulliDecoder(nn.Module):

    def __init__(self, in_dim, hidden_dim=400, hidden_layers=2, layer_norm=True):
        super().__init__()
        self.model = MLP_v2(in_dim, 1, hidden_dim, hidden_layers, layer_norm)

    def forward(self, features: Tensor) -> D.Distribution:
        y = self.model.forward(features)
        p = D.Bernoulli(logits=y.float())
        return p

    def loss(self, output: D.Distribution, target: Tensor) -> Tensor:
        return -output.log_prob(target)

    def training_step(self, features: TensorTBIF, target: Tensor) -> Tuple[TensorTBI, TensorTB, TensorTB]:
        assert len(features.shape) == 4
        I = features.shape[2]
        target = insert_dim(target, 2, I)  # Expand target with iwae_samples dim, because features have it

        decoded = self.forward(features)
        loss_tbi = self.loss(decoded, target)
        loss_tb = -logavgexp(-loss_tbi, dim=2)  # TBI => TB
        decoded = decoded.mean.mean(dim=2)

        assert len(loss_tbi.shape) == 3
        assert len(loss_tb.shape) == 2
        assert len(decoded.shape) == 2
        return loss_tbi, loss_tb, decoded


class DenseNormalDecoder(nn.Module):

    def __init__(self, in_dim, out_dim=1, hidden_dim=400, hidden_layers=2, layer_norm=True, std=0.3989422804):
        super().__init__()
        self.model = MLP_v2(in_dim, out_dim, hidden_dim, hidden_layers, layer_norm)
        self.std = std
        self.out_dim = out_dim

    def forward(self, features: Tensor) -> D.Distribution:
        y = self.model.forward(features)
        p = D.Normal(loc=y, scale=torch.ones_like(y) * self.std)
        if self.out_dim > 1:
            p = D.independent.Independent(p, 1)  # Makes p.logprob() sum over last dim
        return p

    def loss(self, output: D.Distribution, target: Tensor) -> Tensor:
        var = self.std ** 2  # var cancels denominator, which makes loss = 0.5 (target-output)^2
        return -output.log_prob(target) * var

    def training_step(self, features: TensorTBIF, target: Tensor) -> Tuple[TensorTBI, TensorTB, Tensor]:
        assert len(features.shape) == 4
        I = features.shape[2]
        target = insert_dim(target, 2, I)  # Expand target with iwae_samples dim, because features have it

        decoded = self.forward(features)
        loss_tbi = self.loss(decoded, target)
        loss_tb = -logavgexp(-loss_tbi, dim=2)  # TBI => TB
        decoded = decoded.mean.mean(dim=2)

        assert len(loss_tbi.shape) == 3
        assert len(loss_tb.shape) == 2
        assert len(decoded.shape) == (2 if self.out_dim == 1 else 3)
        return loss_tbi, loss_tb, decoded


class DenseCategoricalSupportDecoder(nn.Module):
    """
    Represent continuous variable distribution by discrete set of support values.
    Useful for reward head, which can be e.g. [-10, 0, 1, 10]
    """

    def __init__(self, in_dim, support=[0.0, 1.0], hidden_dim=400, hidden_layers=2, layer_norm=True):
        assert isinstance(support, (list, np.ndarray))
        super().__init__()
        self.model = MLP_v2(in_dim, len(support), hidden_dim, hidden_layers, layer_norm)
        self.support = np.array(support).astype(float)
        self._support = nn.Parameter(torch.tensor(support).to(torch.float), requires_grad=False)

    def forward(self, features: Tensor) -> D.Distribution:
        y = self.model.forward(features)
        p = CategoricalSupport(logits=y.float(), sup=self._support.data)
        return p

    def loss(self, output: D.Distribution, target: Tensor) -> Tensor:
        target = self.to_categorical(target)
        return -output.log_prob(target)

    def to_categorical(self, target: Tensor) -> Tensor:
        # TODO: should interpolate between adjacent values, like in MuZero
        distances = torch.square(target.unsqueeze(-1) - self._support)
        return distances.argmin(-1)

    def training_step(self, features: TensorTBIF, target: Tensor) -> Tuple[TensorTBI, TensorTB, TensorTB]:
        assert len(features.shape) == 4
        I = features.shape[2]
        target = insert_dim(target, 2, I)  # Expand target with iwae_samples dim, because features have it

        decoded = self.forward(features)
        loss_tbi = self.loss(decoded, target)
        loss_tb = -logavgexp(-loss_tbi, dim=2)  # TBI => TB
        decoded = decoded.mean.mean(dim=2)

        assert len(loss_tbi.shape) == 3
        assert len(loss_tb.shape) == 2
        assert len(decoded.shape) == 2
        return loss_tbi, loss_tb, decoded
