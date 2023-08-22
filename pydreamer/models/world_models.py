from typing import Any, Tuple

import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F

from ..tools import *
from .actorcritic import *
from .networks.common import *
from .math_functions import *
from .networks.encoders import *
from .networks.decoders import *
from .networks.rssm_component import *
from .networks.rssm import *
# from .rssm_simplified import RSSMCore, RSSMCell
from .probes import *
from . import tools_v3
from .networks import *


class WorldModel_v2(nn.Module):

    def __init__(self, obs_space,conf):
        super().__init__()
        shapes = {k: tuple(v.shape) for k, v in obs_space.spaces.items()}
        self.deter_dim = conf.deter_dim
        self.stoch_dim = conf.stoch_dim
        self.stoch_discrete = conf.stoch_discrete
        self.kl_weight = conf.kl_weight
        self.aux_critic_weight = conf.aux_critic_weight
        
        ## parameter for different WM
        self.wm_type=conf.wm_type
        # if self.wm_type=='v2':
        self.kl_balance=conf.kl_balance
        # elif self.wm_type=='v3':
        #     self.kl_balance=conf.kl_balance
        #     self._step=step
        #     self.kl_free=conf.kl_free
        #     self.dyn_scale=conf.dyn_scale
        #     self.rep_scale=conf.rep_scale
        # Encoder

        self.encoder = MultiEncoder_v2(shapes,conf)
        
        # RSSM

        self.dynamics = RSSMCore(embed_dim=self.encoder.out_dim,
                             action_dim=conf.action_dim,
                             deter_dim=conf.deter_dim,
                             stoch_dim=conf.stoch_dim,
                             stoch_discrete=conf.stoch_discrete,
                             hidden_dim=conf.hidden_dim,
                             gru_layers=conf.gru_layers,
                             gru_type=conf.gru_type,
                             layer_norm=conf.layer_norm,
                             tidy=conf.tidy)


        # Decoders for image,rewards and cont

        features_dim = conf.deter_dim + conf.stoch_dim * (conf.stoch_discrete or 1)
        self.decoder = MultiDecoder_v2(features_dim, conf)
        
        
        # Auxiliary critic

        if conf.aux_critic:
            self.ac_aux = ActorCritic_v2(in_dim=features_dim,
                                      out_actions=conf.action_dim,
                                      layer_norm=conf.layer_norm,
                                      gamma=conf.gamma_aux,
                                      lambda_gae=conf.lambda_gae_aux,
                                      entropy_weight=conf.entropy,
                                      target_interval=conf.target_interval_aux,
                                      actor_grad=conf.actor_grad,
                                      actor_dist=conf.actor_dist,
                                      )
        else:
            self.ac_aux = None

        # Init

        for m in self.modules():
            init_weights_tf2(m)

    def init_state(self, batch_size: int) -> Tuple[Any, Any]:
        return self.dynamics.init_state(batch_size)

    def forward(self,
                obs: Dict[str, Tensor],
                in_state: Any
                ):
        loss, features, states, out_state, metrics, tensors = \
            self.training_step(obs, in_state, forward_only=True)
        return features, out_state

    def training_step(self,
                      obs: Dict[str, Tensor],
                      in_state: Any,
                      iwae_samples: int = 1,
                      do_open_loop=False,
                      do_image_pred=False,
                      forward_only=False
                      ):
        # Encoder

        embed = self.encoder(obs)

        # RSSM

        prior, post, post_samples, features, states, out_state = \
            self.dynamics.forward(embed,
                              obs['action'],
                              obs['reset'],
                              in_state,
                              iwae_samples=iwae_samples,
                              do_open_loop=do_open_loop)

        if forward_only:
            return torch.tensor(0.0), features, states, out_state, {}, {}

        # Decoder

        loss_reconstr, metrics, tensors = self.decoder.training_step(features, obs)

        # KL loss

        d = self.dynamics.zdistr
        dprior = d(prior)
        dpost = d(post)
        loss_kl_exact = D.kl.kl_divergence(dpost, dprior)  # (T,B,I)
        if iwae_samples == 1:
            # Analytic KL loss, standard for VAE
            if not self.kl_balance:
                loss_kl = loss_kl_exact
            else:
                loss_kl_postgrad = D.kl.kl_divergence(dpost, d(prior.detach()))
                loss_kl_priograd = D.kl.kl_divergence(d(post.detach()), dprior)
                # if self.wm_type=='v2':
                loss_kl = (1 - self.kl_balance) * loss_kl_postgrad + self.kl_balance * loss_kl_priograd
                # elif self.wm_type=='v3':
                    # kl_free = schedule(self.kl_free, self._step)
                    # dyn_scale = schedule(self.dyn_scale, self._step)
                    # rep_scale = schedule(self.rep_scale, self._step)
                    ## Do a clip
                    # rep_loss = torch.clip(loss_kl_postgrad, min=kl_free)
                    # dyn_loss = torch.clip(loss_kl_priograd, min=kl_free)
                    # loss_kl = dyn_scale * dyn_loss + rep_scale * rep_loss
        else:
            # Sampled KL loss, for IWAE
            z = post_samples.reshape(dpost.batch_shape + dpost.event_shape)
            loss_kl = dpost.log_prob(z) - dprior.log_prob(z)

        # Auxiliary critic loss

        if self.ac_aux:
            features_tb = features.select(2, 0)  # (T,B,I) => (T,B) - assume I=1
            (_, loss_critic_aux), metrics_ac, tensors_ac = \
                self.ac_aux.training_step(features_tb,
                                          obs['action'][1:],
                                          obs['reward'],
                                          obs['terminal'])
            metrics.update(loss_critic_aux=metrics_ac['loss_critic'],
                           policy_value_aux=metrics_ac['policy_value_im'])
            tensors.update(policy_value_aux=tensors_ac['value'])
        else:
            loss_critic_aux = 0.0

        # Total loss

        assert loss_kl.shape == loss_reconstr.shape
        loss_model_tbi = self.kl_weight * loss_kl + loss_reconstr
        loss_model = -logavgexp(-loss_model_tbi, dim=2)
        loss = loss_model.mean() + self.aux_critic_weight * loss_critic_aux

        # Metrics

        with torch.no_grad():
            loss_kl = -logavgexp(-loss_kl_exact, dim=2)  # Log exact KL loss even when using IWAE, it avoids random negative values
            entropy_prior = dprior.entropy().mean(dim=2)
            entropy_post = dpost.entropy().mean(dim=2)
            tensors.update(loss_kl=loss_kl.detach(),
                           entropy_prior=entropy_prior,
                           entropy_post=entropy_post)
            metrics.update(loss_model=loss_model.mean(),
                           loss_kl=loss_kl.mean(),
                           entropy_prior=entropy_prior.mean(),
                           entropy_post=entropy_post.mean())

        # Predictions

        if do_image_pred:
            with torch.no_grad():
                prior_samples = self.dynamics.zdistr(prior).sample().reshape(post_samples.shape)
                features_prior = self.dynamics.feature_replace_z(features, prior_samples)
                # Decode from prior(就是没有看到xt，凭借ht直接给出的预测)
                _, mets, tens = self.decoder.training_step(features_prior, obs, extra_metrics=True)
                metrics_logprob = {k.replace('loss_', 'logprob_'): v for k, v in mets.items() if k.startswith('loss_')}
                tensors_logprob = {k.replace('loss_', 'logprob_'): v for k, v in tens.items() if k.startswith('loss_')}
                tensors_pred = {k.replace('_rec', '_pred'): v for k, v in tens.items() if k.endswith('_rec')}
                metrics.update(**metrics_logprob)   # logprob_image, ...
                tensors.update(**tensors_logprob)  # logprob_image, ...
                tensors.update(**tensors_pred)  # image_pred, ...

        return loss, features, states, out_state, metrics, tensors
    
    
class WorldModel_v3(nn.Module):
    def __init__(self, obs_space, act_space, step, conf,device):
        super(WorldModel_v3, self).__init__()
        self._step = step
        self._use_amp = True if conf.precision == 16 else False
        self._conf = conf
        shapes = {k: tuple(v.shape) for k, v in obs_space.spaces.items()}
        self._device=device
        # Encoder
        self.encoder = MultiEncoder_v3(shapes, conf)
        
        # RSSM
        self.embed_size = self.encoder.out_dim
        self.dynamics = RSSM(
            conf.dyn_stoch,
            conf.dyn_deter,
            conf.dyn_hidden,
            conf.dyn_input_layers,
            conf.dyn_output_layers,
            conf.dyn_rec_depth,
            conf.dyn_shared,
            conf.dyn_discrete,
            conf.act,
            conf.norm,
            conf.dyn_mean_act,
            conf.dyn_std_act,
            conf.dyn_temp_post,
            conf.dyn_min_std,
            conf.dyn_cell,
            conf.unimix_ratio,
            conf.initial,
            #为啥原文件里没有这个
            # conf.num_actions,
            conf.action_dim,
            self.embed_size,
            conf.device,
        )
        # dECODERS FOR IMAGE,REWARDS and counts
        self.heads = nn.ModuleDict()
        if conf.dyn_discrete:
            feat_size = conf.dyn_stoch * conf.dyn_discrete + conf.dyn_deter
        else:
            feat_size = conf.dyn_stoch + conf.dyn_deter
        self.heads["decoder"] = MultiDecoder_v3(
            feat_size, shapes, **conf.decoder
        )
        if conf.reward_head == "symlog_disc":
            self.heads["reward"] = MLP_v3(
                feat_size,  # pytorch version
                (255,),
                conf.reward_layers,
                conf.units,
                conf.act,
                conf.norm,
                dist=conf.reward_head,
                outscale=0.0,
                device=self._device,
            )
        else:
            self.heads["reward"] = MLP_v3(
                feat_size,  # pytorch version
                [],
                conf.reward_layers,
                conf.units,
                conf.act,
                conf.norm,
                dist=conf.reward_head,
                outscale=0.0,
                device=self._device,
            )
        self.heads["cont"] = MLP_v3(
            feat_size,  # pytorch version
            [],
            conf.cont_layers,
            conf.units,
            conf.act,
            conf.norm,
            dist="binary",
            device=self._device,
        )
        for name in conf.grad_heads:
            assert name in self.heads, name
        # self._model_opt = tools_v3.Optimizer(
        #     "model",
        #     self.parameters(),
        #     conf.model_lr,
        #     conf.opt_eps,
        #     conf.grad_clip,
        #     conf.weight_decay,
        #     opt=conf.opt,
        #     use_amp=self._use_amp,
        # )
        self._scales = dict(reward=conf.reward_scale, cont=conf.cont_scale)
        
    def init_state(self, batch_size: int) -> Tuple[Any, Any]:
        return self.dynamics.init_state(batch_size)
    
    def forward(self,
                obs: Dict[str, Tensor],
                in_state: Any
                ):
        # TODO:v3好像不需要输入state，一开始h和z都是随机initial的；但是v2需要？
        model_loss,feat, post, context, metrics,tensors= \
            self.training_step(obs, forward_only=True)
        out_state={key: tensor[-1] for key, tensor in post.items()}
        return feat, out_state

    def training_step(self, data,do_image_pred=False,forward_only=False):
        # action (batch_size, batch_length, act_dim)
        # image (batch_size, batch_length, h, w, ch)
        # reward (batch_size, batch_length)
        # discount (batch_size, batch_length)
        data = self.preprocess(data,forward_only=True)
        with tools_v3.RequiresGrad(self):
            with torch.cuda.amp.autocast(self._use_amp):
                embed = self.encoder(data)
                post, prior = self.dynamics.observe(
                    embed, data["action"], data["reset"]
                )
                if forward_only:
                    post = {k: v.detach() for k, v in post.items()}
                    feat=self.dynamics.to_feature(post)
                    return torch.tensor(0.0), feat, post, {}, {},{}
                kl_free = tools_v3.schedule(self._conf.kl_free, self._step)
                dyn_scale = tools_v3.schedule(self._conf.dyn_scale, self._step)
                rep_scale = tools_v3.schedule(self._conf.rep_scale, self._step)
                kl_loss, kl_value, dyn_loss, rep_loss = self.dynamics.kl_loss(
                    post, prior, kl_free, dyn_scale, rep_scale
                )
                preds = {}
                for name, head in self.heads.items():
                    grad_head = name in self._conf.grad_heads
                    feat = self.dynamics.to_feature(post)
                    feat = feat if grad_head else feat.detach()
                    pred = head(feat)
                    if type(pred) is dict:
                        preds.update(pred)
                    else:
                        preds[name] = pred
                losses = {}
                for name, pred in preds.items():
                    like = pred.log_prob(data[name])
                    losses[name] = -torch.mean(like) * self._scales.get(name, 1.0)
                model_loss = sum(losses.values()) + kl_loss
        #     metrics = self._model_opt(model_loss, self.parameters())
        tensors = {}
        metrics = {}
        metrics["loss_model"] = model_loss.detach().cpu().numpy()
        metrics.update({f"{name}_loss": to_np(loss) for name, loss in losses.items()})
        metrics["kl_free"] = kl_free
        metrics["dyn_scale"] = dyn_scale
        metrics["rep_scale"] = rep_scale
        metrics["loss_dyn"] = to_np(dyn_loss)
        metrics["loss_rep"] = to_np(rep_loss)
        metrics["loss_kl"] = to_np(kl_loss)
        # metrics["kl"] = to_np(torch.mean(kl_value))
        with torch.cuda.amp.autocast(self._use_amp):
            metrics["prior_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(prior).entropy())
            )
            metrics["post_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(post).entropy())
            )
            context = dict(
                embed=embed,
                feat=self.dynamics.to_feature(post),
                kl=kl_value,
                postent=self.dynamics.get_dist(post).entropy(),
            )
        post = {k: v.detach() for k, v in post.items()}
        feat=self.dynamics.to_feature(post)
        
        # Predictions
        if do_image_pred:
            tensors=self.video_pred(data)
        return model_loss,feat, post, context, metrics,tensors

    def preprocess(self, obs,forward_only=False):
        obs = obs.copy()
        obs["image"] = torch.Tensor(obs["image"]) / 255.0 - 0.5
        # (batch_size, batch_length) -> (batch_size, batch_length, 1)
        obs["reward"] = torch.Tensor(obs["reward"]).unsqueeze(-1)
        if "discount" in obs:
            obs["discount"] *= self._conf.discount
            # (batch_size, batch_length) -> (batch_size, batch_length, 1)
            obs["discount"] = torch.Tensor(obs["discount"]).unsqueeze(-1)
        if "terminal" in obs:
            # this label is necessary to train cont_head
            obs["cont"] = torch.Tensor(1.0 - obs["terminal"]).unsqueeze(-1)
        else:
            raise ValueError('"terminal" was not found in observation.')
        if not forward_only:
            obs = {k: torch.Tensor(v).to(self._device) for k, v in obs.items()}
        return obs

    def video_pred(self, data):
        data = self.preprocess(data)
        embed = self.encoder(data)
        states, _ = self.dynamics.observe(
            embed[:5, :6], data["action"][:5, :6], data["reset"][:5, :6]
        )
        recon = self.heads["decoder"](self.dynamics.to_feature(states))["image"].mode()[
            :5
        ]
        reward_post = self.heads["reward"](self.dynamics.to_feature(states)).mode()[:5]
        # init = {k: v[:, -1] for k, v in states.items()}
        #init 结构为B*hidden_dim
        init={k: v[-1,:] for k, v in states.items()}
        prior = self.dynamics.imagine(data["action"][5:, :6], init)
        openl = self.heads["decoder"](self.dynamics.to_feature(prior))["image"].mode()
        reward_prior = self.heads["reward"](self.dynamics.to_feature(prior)).mode()
        # observed image is given until 5 steps
        model = torch.cat([recon[:5, :6], openl], 0)
        truth = data["image"][:,:6] + 0.5
        # model = model + 0.5
        error = (model - truth + 1.0) / 2.0
        
        tensors={}
        # train 中已有函数保存original data了
        # tensors.update(data)
        tensors["image_dream"]=model
        tensors["image_error"]=error
        

        # return torch.cat([truth, model, error], 2)
        return tensors