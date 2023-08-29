import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import copy
from torch import Tensor

from .math_functions import *
from .networks.common import *
from.tools_v3 import *

class RewardEMA(object):
    """running mean and std"""

    def __init__(self, device, alpha=1e-2):
        self.device = device
        self.values = torch.zeros((2,)).to(device)
        self.alpha = alpha
        self.range = torch.tensor([0.05, 0.95]).to(device)

    def __call__(self, x):
        flat_x = torch.flatten(x.detach())
        x_quantile = torch.quantile(input=flat_x, q=self.range)
        self.values = self.alpha * x_quantile + (1 - self.alpha) * self.values
        scale = torch.clip(self.values[1] - self.values[0], min=1.0)
        offset = self.values[0]
        return offset.detach(), scale.detach()

class ActorCritic(nn.Module):

    def __init__(self,
                 conf,world_model,device
                 ):
        super().__init__()
        ## feature_dim (h,z)
        self._conf=conf
        self.wm_type=conf.wm_type
        self._world_model=world_model
        self._device=device
        # self.action_dim = conf.action_dim
        # self.discount = conf.discount
        # self.lambda_ = conf.lambda_gae
        # self.entropy_weight = conf.actor_entropy
        # self.slow_value_target=conf.slow_value_target
        # self.slow_target_update=conf.slow_target_update
        # self.slow_target_fraction=conf.slow_target_fraction
        # self.actor_grad = conf.actor_grad
        # self.actor_dist = conf.actor_dist
        actor_out_dim = conf.action_dim if conf.actor_dist == 'onehot' else 2 * conf.action_dim
        feat_size = conf.deter_dim + conf.stoch_dim * (conf.stoch_discrete or 1)
        self.feat_size = feat_size
        hidden_layers=4
        
        self.actor = MLP_v2(feat_size, actor_out_dim, conf.hidden_dim, hidden_layers, conf.layer_norm)
        if self.wm_type=='v2':
            self.critic = MLP_v2(feat_size, 1,  conf.hidden_dim, hidden_layers, conf.layer_norm)
        elif self.wm_type=='v3':
            if self._conf.reward_EMA:
                self.reward_ema = RewardEMA(device=self._device)
            if conf.value_head == "symlog_disc":
                self.critic = MLP_v3(
                    feat_size,
                    (255,),
                    conf.value_layers,
                    conf.units,
                    conf.act,
                    conf.norm,
                    conf.value_head,
                    outscale=0.0,
                    device=self._device,
                )
            else:
                self.critic = MLP_v3(
                    feat_size,
                    [],
                    conf.value_layers,
                    conf.units,
                    conf.act,
                    conf.norm,
                    conf.value_head,
                    outscale=0.0,
                    device=self._device,
                )
        # self.critic_target = MLP_v2(feat_size, 1,  conf.hidden_dim, conf.hidden_layers, conf.layer_norm)
        # ## Here is a change! Orginally false, but I change it to true
        # # self.critic_target.requires_grad_(False)
        # self.critic_target.requires_grad_(True)
        if conf.slow_value_target:
            self._slow_value = copy.deepcopy(self.critic)
            self._updates = 0

    def forward_actor(self, features: Tensor) -> D.Distribution:
        y = self.actor.forward(features).float()  # .float() to force float32 on AMP
        
        if self._conf.actor_dist == 'onehot':
            return D.OneHotCategorical(logits=y)
        
        if self._conf.actor_dist == 'normal_tanh':
            return normal_tanh(y)

        if self._conf.actor_dist == 'tanh_normal':
            return tanh_normal(y)

        assert False, self._conf.actor_dist

    def forward_value(self, features: Tensor) -> Tensor:
        y = self.critic.forward(features)
        return y

    def training_step(self,
                      features: TensorJMF,
                      actions: TensorHMA,
                      rewards: TensorJM,
                      terminals: TensorJM,
                      log_only=False
                      ):
        """
        The ordering is as follows:
            features[0] 
            -> actions[0] -> rewards[1], terminals[1], features[1]
            -> actions[1] -> ...
            ...
            -> actions[H-1] -> rewards[H], terminals[H], features[H]
        """
        if not log_only:
            # 每轮都更新一点点
            # if self._updates % self.target_interval == 0:
                # self.update_critic_target()
            self._update_slow_target()
        self._updates += 1
        
        # reward1: TensorHM = rewards[1:]
        # terminal0: TensorHM = terminals[:-1]
        # terminal1: TensorHM = terminals[1:]
        # if self._conf.wm_type=='v3':
        #     reward1=reward1.squeeze(-1)
        #     terminal0=terminal0.squeeze(-1)
        #     terminal1=terminal1.squeeze(-1)
        
        # # GAE from https://arxiv.org/abs/1506.02438 eq (16)
        # #   advantage_gae[t] = advantage[t] + (discount lambda) advantage[t+1] + (discount lambda)^2 advantage[t+2] + ...

        # value_t: TensorJM = self._slow_value.forward(features)
        # value0t: TensorHM = value_t[:-1]
        # value1t: TensorHM = value_t[1:]
        # # TD error=r+\discount*V(s')-V(s)
        # advantage = - value0t + reward1 + self._conf.discount * (1.0 - terminal1) * value1t
        # advantage_gae = []
        # agae = None
        # # GAE的累加
        # for adv, term in zip(reversed(advantage.unbind()), reversed(terminal1.unbind())):
        #     if agae is None:
        #         agae = adv
        #     else:
        #         agae = adv + self._conf.lambda_gae * self._conf.discount * (1.0 - term) * agae
        #     advantage_gae.append(agae)
        # advantage_gae.reverse()
        # advantage_gae = torch.stack(advantage_gae)
        # # Note: if lambda=0, then advantage_gae=advantage, then value_target = advantage + value0t = reward + discount * value1t
        # value_target = advantage_gae + value0t

        # # When calculating losses, should ignore terminal states, or anything after, so:
        # #   reality_weight[i] = (1-terminal[0]) (1-terminal[1]) ... (1-terminal[i])
        # # Note this takes care of the case when initial state features[0] is terminal - it will get weighted by (1-terminals[0]).
        # reality_weight = (1 - terminal0).log().cumsum(dim=0).exp()
        
        actor_ent = self.forward_actor(features[:-1]).entropy()
        # state_ent = self._world_model.dynamics.get_dist(states).entropy()
        state_ent=0
        value_target, reality_weight, base = self._compute_target(
            features, actions, rewards, terminals,actor_ent, state_ent
        )

        # Critic loss
        
        loss_critic,critic_mets,tensors=self._compute_critic_loss(
        features,
        actions,
        value_target,
        reality_weight)

        # value: TensorJM = self.critic.forward(features)
        # value0: TensorHM = value[:-1]
        # loss_critic = 0.5 * torch.square(value_target.detach() - value0)
        # loss_critic = (loss_critic * reality_weight).mean()

        # Actor loss
        
        #actor_loss
        loss_actor, act_mets = self._compute_actor_loss(
            features,
            actions,
            value_target,
            actor_ent,
            state_ent,
            reality_weight,
            base,
        )

        # policy_distr = self.forward_actor(features[:-1])  # TODO: we could reuse this from dream()
        # if self._conf.actor_grad == 'reinforce':
        #     action_logprob = policy_distr.log_prob(actions)
        #     loss_policy = - action_logprob * advantage_gae.detach()
        # elif self._conf.actor_grad == 'dynamics':
        #     # loss_policy = - value_target
        #     loss_policy=- advantage_gae
        # else:
        #     assert False, self._conf.actor_grad

        # policy_entropy = policy_distr.entropy()
        # loss_actor = loss_policy - self._conf.actor_entropy * policy_entropy
        # loss_actor = (loss_actor * reality_weight).mean()
        # assert (loss_policy.requires_grad and loss_policy.requires_grad) or not loss_critic.requires_grad

        with torch.no_grad():
            metrics = dict(
                           policy_reward=rewards[:1].mean(),
                           policy_reward_std=rewards[:1].std(),
                           )
            metrics.update(**act_mets,**critic_mets)
            # tensors = dict(value=value.detach(),
            #                value_target=value_target.detach(),
            #                value_advantage=advantage.detach(),
            #                value_advantage_gae=advantage_gae.detach(),
            #                value_weight=reality_weight.detach(),
            #                )

        return (loss_actor, loss_critic), metrics, tensors
    
    def _compute_target(
        self, features, actions, reward, terminal,actor_ent, state_ent
    ):
        if self.wm_type=='v2':
            reward1: TensorHM = reward[1:]
            terminal0: TensorHM = terminal[:-1]
            terminal1: TensorHM = terminal[1:]
            # if self._conf.wm_type=='v3':
            #     reward1=reward1.squeeze(-1)
            #     terminal0=terminal0.squeeze(-1)
            #     terminal1=terminal1.squeeze(-1)

            # GAE from https://arxiv.org/abs/1506.02438 eq (16)
            #   advantage_gae[t] = advantage[t] + (discount lambda) advantage[t+1] + (discount lambda)^2 advantage[t+2] + ...

            value_t: TensorJM = self._slow_value.forward(features)
            value0t: TensorHM = value_t[:-1]
            value1t: TensorHM = value_t[1:]
            # TD error=r+\discount*V(s')-V(s)
            advantage = - value0t + reward1 + self._conf.discount * (1.0 - terminal1) * value1t
            advantage_gae = []
            agae = None
            # GAE的累加
            for adv, term in zip(reversed(advantage.unbind()), reversed(terminal1.unbind())):
                if agae is None:
                    agae = adv
                else:
                    agae = adv + self._conf.lambda_gae * self._conf.discount * (1.0 - term) * agae
                advantage_gae.append(agae)
            advantage_gae.reverse()
            advantage_gae = torch.stack(advantage_gae)
            # Note: if lambda=0, then advantage_gae=advantage, then value_target = advantage + value0t = reward + discount * value1t
            value_target = advantage_gae + value0t

            # When calculating losses, should ignore terminal states, or anything after, so:
            #   reality_weight[i] = (1-terminal[0]) (1-terminal[1]) ... (1-terminal[i])
            # Note this takes care of the case when initial state features[0] is terminal - it will get weighted by (1-terminals[0]).
            # Note that this weights didn't consider discounts
            reality_weight = (1 - terminal0).log().cumsum(dim=0).exp()
            return value_target,reality_weight,value0t
        
        elif self.wm_type=='v3':
            ## discount
            if self._world_model.decoder.terminal is not None:
                print('terminal exists')
                # discount = self._conf.discount * self._world_model.decoder.terminal(inp).mean
                ## 注意现在这里是terminal，不是cont，所以要用1-！
                discount = self._conf.discount * (1-terminal)
            else:
                discount = self._conf.discount * torch.ones_like(reward)
            ## entropy
            if self._conf.future_entropy and self._conf.actor_entropy > 0:
                reward += self._conf.actor_entropy * actor_ent
            if self._conf.future_entropy and self._conf.actor_state_entropy > 0:
                reward += self._conf.actor_state_entropy * state_ent
            #valu_estimator
            value = self.critic(features).mode()
            target = lambda_return(
                reward[1:],
                value[:-1],
                discount[1:],
                bootstrap=value[-1],
                lambda_=self._conf.lambda_gae,
                axis=0,
            )
            weights = torch.cumprod(
                torch.cat([torch.ones_like(discount[:1]), discount[:-1]], 0), 0
            ).detach()
            return target, weights, value[:-1]

    # def update_critic_target(self):
    #     self.critic_target.load_state_dict(self.critic.state_dict())  # type: ignore
    
    def _compute_actor_loss(
        self,
        features,
        actions,
        target,
        actor_ent,
        state_ent,
        reality_weight,
        base,
    ): 
        policy_distr = self.forward_actor(features[:-1])
        actor_metric = {}
        if self.wm_type=='v2':
              # TODO: we could reuse this from dream()
            advantage_gae=target-base
            if self._conf.actor_grad == 'reinforce':
                action_logprob = policy_distr.log_prob(actions)
                loss_policy = - action_logprob * advantage_gae.detach()
            elif self._conf.actor_grad == 'dynamics':
                # loss_policy = - value_target
                loss_policy=- advantage_gae
            else:
                assert False, self._conf.actor_grad

            loss_actor = loss_policy - self._conf.actor_entropy * actor_ent
            loss_actor = (loss_actor * reality_weight).mean()
        
        elif self.wm_type=='v3':
            # Q-val for actor is not transformed using symlog
            target = torch.stack(target, dim=1)
            ## 做一些v3特有的处理
            if self._conf.reward_EMA:
                offset, scale = self.reward_ema(target)
                normed_target = (target - offset) / scale
                normed_base = (base - offset) / scale
                adv = normed_target - normed_base
                actor_metric.update(tensorstats(normed_target, "normed_target"))
                values = self.reward_ema.values
                actor_metric["EMA_005"] = to_np(values[0])
                actor_metric["EMA_095"] = to_np(values[1])

            if self._conf.actor_grad == "dynamics":
                loss_policy = -adv
            elif self._conf.actor_grad == "reinforce":
                # actor_target = (
                #     policy.log_prob(actions)[:, :, None]
                #     * (target - self.critic(features[:-1]).mode()).detach()
                # )
                action_logprob = policy_distr.log_prob(actions)
                # 注意这里减的是critic，不是slow-critic
                loss_policy = -action_logprob* (target - self.critic(features[:-1]).mode()).detach()   
            elif self._conf.actor_grad == "both":
                # actor_target = (
                #     # policy.log_prob(actions)[:, :, None]
                #     policy.log_prob(actions)
                #     * (target - self.critic(features[:-1]).mode()).detach()
                # )
                action_logprob = policy_distr.log_prob(actions)
                loss_policy = -action_logprob* (target - self.critic(features[:-1]).mode()).detach()
                mix = self._conf.imag_gradient_mix()
                # loss_policy = mix * (-target) + (1 - mix) * loss_policy
                loss_policy = mix * (-adv) + (1 - mix) * loss_policy
                actor_metric["imag_gradient_mix"] = mix
            else:
                raise NotImplementedError(self._conf.actor_grad)
            if not self._conf.future_entropy and (self._conf.actor_entropy > 0):
                #第三维度调整
                # actor_entropy = self._conf.actor_entropy * actor_ent[:, :, None]
                actor_entropy = self._conf.actor_entropy * actor_ent
                loss_policy -= actor_entropy
            if not self._conf.future_entropy and (self._conf.actor_state_entropy > 0):
                state_entropy = self._conf.actor_state_entropy * state_ent[:-1]
                loss_policy -= state_entropy
                actor_metric["actor_state_entropy"] = to_np(torch.mean(state_entropy))
            loss_actor = torch.mean(reality_weight[:-1] * loss_policy)
            
        actor_metric["policy_entropy"] = to_np(torch.mean(actor_ent))
        actor_metric["loss_actor"] = loss_actor.detach().cpu().numpy()
        return loss_actor, actor_metric
        
    def _compute_critic_loss(
        self,
        features,
        actions,
        value_target,
        reality_weight,
    ): 
        critic_metric={}
        # Critic loss
        if self.wm_type=='v2':
            value: TensorJM = self.critic.forward(features)
            value: TensorHM = value[:-1]
            loss_critic = 0.5 * torch.square(value_target.detach() - value)
            loss_critic = (loss_critic * reality_weight).mean()
            critic_metric['policy_value']=value[0].mean().detach().cpu(),  # Value of real states
            critic_metric['policy_value_im']=value.mean().detach().cpu(),  # Value of imagined states
        
        elif self.wm_type=='v3':
            value = self.critic(features[:-1].detach())
            value_target = torch.stack(value_target, dim=1)
            # (time, batch, 1), (time, batch, 1) -> (time, batch)
            loss_critic = -value.log_prob(value_target.detach())
            slow_target = self._slow_value(features[:-1].detach())
            if self._conf.slow_value_target:
                loss_critic = loss_critic - value.log_prob(
                    slow_target.mode().detach()
                )
            if self._conf.value_decay:
                loss_critic += self._conf.value_decay * value.mode()
            # (time, batch, 1), (time, batch, 1) -> (1,)
            # 第三维度调整
            # value_loss = torch.mean(weights[:-1] * value_loss[:, :, None])
            loss_critic = torch.mean(reality_weight[:-1] * loss_critic)
            # print("loss_critic",value_loss)

            critic_metric.update(tensorstats(value.mode().detach(), "policy_value"))
            critic_metric.update(tensorstats(value_target, "value_target"))
            # critic_metric.update(tensorstats(rewards, "imag_reward"))
            if self._conf.actor_dist in ["onehot"]:
                critic_metric.update(
                    tensorstats(
                        torch.argmax(actions, dim=-1).float(), "actions"
                    )
                )
            else:
                critic_metric.update(tensorstats(actions, "actions"))
            
        # with RequiresGrad(self):
            # metrics.update(self._actor_opt(actor_loss, self.actor.parameters()))
            # metrics.update(self._value_opt(value_loss, self.value.parameters()))
        
        critic_metric["loss_critic"] =loss_critic.detach().cpu().numpy()
        # critic_metric["loss_critic"] = value_loss.detach().cpu().numpy()
        tensors = dict(value=value.mode(),
                        value_weight=reality_weight.detach(),
                        )
        return loss_critic,critic_metric,tensors
            
        
    def _update_slow_target(self):
        if self._conf.slow_value_target:
            if self._updates % self._conf.slow_target_update == 0:
                mix = self._conf.slow_target_fraction
                for s, d in zip(self.critic.parameters(), self._slow_value.parameters()):
                    d.data = mix * s.data + (1 - mix) * d.data
            self._updates += 1
    
    
    def adversarial_attack(self, state, epsilon=0.2, iters=10):
        # Note that 'state' should be a PyTorch tensor
        
        # Create a copy of the state and require gradient computation
        adversarial_state = state.clone().detach().requires_grad_(True)

        for _ in range(iters):
            # Zero-out all the gradients
            self.critic.zero_grad()

            # Compute Q values
            Q = self.critic(adversarial_state)

            # Compute maximum Q values over actions
            # Qmax = Q.max(dim=1)[0]
            # Qmax = Q.max(dim=0)[0]

            # Compute the gradients
            Q.backward(torch.ones_like(Q))

            # Add an adversarial perturbation
            with torch.no_grad():
                
                adversarial_state += epsilon * adversarial_state.grad.sign()

            # Clamp the adversarial state to the valid range
            adversarial_state = torch.clamp(adversarial_state, state.min(), state.max())

            # Detach the adversarial state
            adversarial_state = adversarial_state.detach().requires_grad_(True)

        return adversarial_state

    def evaluate_with_adversarial_attack(self, features: TensorJMF):
        # Create adversarial states
        features_adv = self.adversarial_attack(features)

        # Feed the adversarial states to the actor and get the actions
        policy_distr = self.forward_actor(features_adv)
    
        # Sample an action from the distribution
        action = policy_distr.sample()
    
        return action


class ActorCritic_v3(nn.Module):
    def __init__(self, conf,world_model,device,stop_grad_actor=True, reward=None):
        # super(ActorCritic_v3, self).__init__()
        super().__init__()
        # self._use_amp = True if conf.precision == 16 else False
        self._conf = conf
        self._world_model = world_model
        self._stop_grad_actor = stop_grad_actor
        # self._reward = reward
        # self._discrete=conf.dyn_discrete
        self._device=device
        actor_out_dim = conf.action_dim if conf.actor_dist == 'onehot' else 2 * conf.action_dim
        if conf.dyn_discrete:
            feat_size = conf.deter_dim + conf.stoch_dim * (conf.stoch_discrete or 1)
        else:
            feat_size = conf.deter_dim + conf.stoch_dim
        hidden_layers=4
        
        # 给出actor和critic
        self.actor = MLP_v2(feat_size, actor_out_dim, 
                            conf.hidden_dim, hidden_layers, conf.layer_norm)
        # self.actor = ActionHead(
        #     feat_size,
        #     conf.action_dim,
        #     conf.actor_layers,
        #     conf.units,
        #     conf.act,
        #     conf.norm,
        #     conf.actor_dist,
        #     conf.actor_init_std,
        #     conf.actor_min_std,
        #     conf.actor_max_std,
        #     conf.actor_temp,
        #     outscale=1.0,
        #     unimix_ratio=conf.action_unimix_ratio,
        # )
        if conf.value_head == "symlog_disc":
            self.critic = MLP_v3(
                feat_size,
                (255,),
                conf.value_layers,
                conf.units,
                conf.act,
                conf.norm,
                conf.value_head,
                outscale=0.0,
                device=self._device,
            )
        else:
            self.critic = MLP_v3(
                feat_size,
                [],
                conf.value_layers,
                conf.units,
                conf.act,
                conf.norm,
                conf.value_head,
                outscale=0.0,
                device=self._device,
            )
        if conf.slow_value_target:
            self._slow_value = copy.deepcopy(self.critic)
            self._updates = 0
        # kw = dict(wd=conf.weight_decay, opt=conf.opt, use_amp=self._use_amp)
        # self._actor_opt = Optimizer(
        #     "actor",
        #     self.actor.parameters(),
        #     conf.actor_lr,
        #     conf.ac_opt_eps,
        #     conf.actor_grad_clip,
        #     **kw,
        # )
        # self._value_opt = Optimizer(
        #     "value",
        #     self.value.parameters(),
        #     conf.value_lr,
        #     conf.ac_opt_eps,
        #     conf.value_grad_clip,
        #     **kw,
        # )
        if self._conf.reward_EMA:
            self.reward_ema = RewardEMA(device=self._device)
    
    def forward_actor(self, features: Tensor) -> D.Distribution:
        y = self.actor.forward(features).float()  # .float() to force float32 on AMP
        
        if self._conf.actor_dist == 'onehot':
            return D.OneHotCategorical(logits=y)
        
        if self._conf.actor_dist == 'normal_tanh':
            return normal_tanh(y)

        if self._conf.actor_dist == 'tanh_normal':
            return tanh_normal(y)
        print(self._conf.actor_dist)

        assert False, self._conf.actor_dist

    def training_step(
        self,
        # start,
        # action=None,
        # reward=None,
        # imagine=None,
        # tape=None,
        # repeats=None,
        features: TensorJMF,
        actions: TensorHMA,
        rewards: TensorJM,
        terminals: TensorJM,
        # states,
        objective=None,
        log_only=False
    ):
        # objective = objective or self._reward
        self._update_slow_target()
        metrics = {}

        # with RequiresGrad(self.actor):
        # with torch.cuda.amp.autocast(self._use_amp):
            # features, states, actions = self._imagine(
            #     start, self.actor, self._conf.imag_horizon, repeats
            # )
            ##这个reward是通过wm送过来的
            # reward = objective(features, states, actions)
        actor_ent = self.forward_actor(features[:-1]).entropy()
        # state_ent = self._world_model.dynamics.get_dist(states).entropy()
        # 暂时没影响，因为这个现在是0
        # state_ent=self.get_dist(states).entropy()
        state_ent=0
        # this target is not scaled
        # slow is flag to indicate whether slow_target is used for lambda-return
        target, weights, base = self._compute_target(
            features, actions, rewards, actor_ent, state_ent
        )
        
        #actor_loss
        actor_loss, mets = self._compute_actor_loss(
            features,
            actions,
            target,
            actor_ent,
            state_ent,
            weights,
            base,
        )
        metrics.update(mets)
        value_input = features

        # with RequiresGrad(self.critic):
        # with torch.cuda.amp.autocast(self._use_amp):
        value = self.critic(value_input[:-1].detach())
        target = torch.stack(target, dim=1)
        # (time, batch, 1), (time, batch, 1) -> (time, batch)
        # value_loss
        value_loss = -value.log_prob(target.detach())
        slow_target = self._slow_value(value_input[:-1].detach())
        if self._conf.slow_value_target:
            value_loss = value_loss - value.log_prob(
                slow_target.mode().detach()
            )
        if self._conf.value_decay:
            value_loss += self._conf.value_decay * value.mode()
        # (time, batch, 1), (time, batch, 1) -> (1,)
        # 第三维度调整
        # value_loss = torch.mean(weights[:-1] * value_loss[:, :, None])
        value_loss = torch.mean(weights[:-1] * value_loss)
        # print("loss_critic",value_loss)

        metrics.update(tensorstats(value.mode().detach(), "policy_value"))
        metrics.update(tensorstats(target, "target"))
        metrics.update(tensorstats(rewards, "imag_reward"))
        if self._conf.actor_dist in ["onehot"]:
            metrics.update(
                tensorstats(
                    torch.argmax(actions, dim=-1).float(), "actions"
                )
            )
        else:
            metrics.update(tensorstats(actions, "actions"))
        metrics["policy_entropy"] = to_np(torch.mean(actor_ent))
        # with RequiresGrad(self):
            # metrics.update(self._actor_opt(actor_loss, self.actor.parameters()))
            # metrics.update(self._value_opt(value_loss, self.value.parameters()))
        metrics["loss_actor"] = actor_loss.detach().cpu().numpy()
        metrics["loss_critic"] = value_loss.detach().cpu().numpy()
        tensors = dict(value=value.mode(),
                        value_weight=weights.detach(),
                        )
        return (actor_loss,value_loss), metrics,tensors

    # def _imagine(self, start, policy, horizon, repeats=None):
    #     dynamics = self._world_model.dynamics
    #     if repeats:
    #         raise NotImplemented("repeats is not implemented in this version")
    #     flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
    #     start = {k: flatten(v) for k, v in start.items()}

    #     def step(prev, _):
    #         state, _, _ = prev
    #         feat = dynamics.to_feature(state)
    #         inp = feat.detach() if self._stop_grad_actor else feat
    #         action = policy(inp).sample()
    #         succ = dynamics.img_step(state, action, sample=self._conf.imag_sample)
    #         return succ, feat, action

    #     succ, feats, actions = static_scan(
    #         step, [torch.arange(horizon)], (start, None, None)
    #     )
    #     states = {k: torch.cat([start[k][None], v[:-1]], 0) for k, v in succ.items()}
    #     if repeats:
    #         raise NotImplemented("repeats is not implemented in this version")

    #     return feats, states, actions
    
    # def get_dist(self, state, dtype=None):
    #     if self._conf.discrete:
    #         logit = state["logit"]
    #         dist = torchd.independent.Independent(
    #             OneHotDist(logit, unimix_ratio=self._conf.action_unimix_ratio), 1
    #         )
    #     else:
    #         mean, std = state["mean"], state["std"]
    #         dist = ContDist(
    #             torchd.independent.Independent(torchd.normal.Normal(mean, std), 1)
    #         )
    #     return dist

    def _compute_target(
        self, features, actions, reward, actor_ent, state_ent
    ):
        ## discount
        # if "terminal" in self._world_model.heads:
        #     # print('terminal exxists')
        #     # inp = self._world_model.dynamics.to_feature(states)
        #     # discount = self._conf.discount * self._world_model.heads["cont"](inp).mean
        #     discount = self._conf.discount * self._world_model.heads["terminal"](features).mean
        # else:
        discount = self._conf.discount * torch.ones_like(reward)
        ## entropy
        if self._conf.future_entropy and self._conf.actor_entropy > 0:
            reward += self._conf.actor_entropy * actor_ent
        if self._conf.future_entropy and self._conf.actor_state_entropy > 0:
            reward += self._conf.actor_state_entropy * state_ent
        #valu_estimator
        value = self.critic(features).mode()
        target = lambda_return(
            reward[1:],
            value[:-1],
            discount[1:],
            bootstrap=value[-1],
            lambda_=self._conf.lambda_gae,
            axis=0,
        )
        weights = torch.cumprod(
            torch.cat([torch.ones_like(discount[:1]), discount[:-1]], 0), 0
        ).detach()
        return target, weights, value[:-1]

    def _compute_actor_loss(
        self,
        features,
        actions,
        target,
        actor_ent,
        state_ent,
        weights,
        base,
    ):
        metrics = {}
        inp = features[:-1].detach() if self._stop_grad_actor else features[:-1]
        policy = self.forward_actor(inp)
        actor_ent = policy.entropy()
        # Q-val for actor is not transformed using symlog
        target = torch.stack(target, dim=1)
        if self._conf.reward_EMA:
            offset, scale = self.reward_ema(target)
            normed_target = (target - offset) / scale
            normed_base = (base - offset) / scale
            adv = normed_target - normed_base
            metrics.update(tensorstats(normed_target, "normed_target"))
            values = self.reward_ema.values
            metrics["EMA_005"] = to_np(values[0])
            metrics["EMA_095"] = to_np(values[1])

        if self._conf.actor_grad == "dynamics":
            actor_target = adv
        elif self._conf.actor_grad == "reinforce":
            # actor_target = (
            #     policy.log_prob(actions)[:, :, None]
            #     * (target - self.critic(features[:-1]).mode()).detach()
            # )
            actor_target = (
                policy.log_prob(actions)
                * (target - self.critic(features[:-1]).mode()).detach()
            )
        elif self._conf.actor_grad == "both":
            actor_target = (
                # policy.log_prob(actions)[:, :, None]
                policy.log_prob(actions)
                * (target - self.critic(features[:-1]).mode()).detach()
            )
            mix = self._conf.imag_gradient_mix()
            actor_target = mix * target + (1 - mix) * actor_target
            metrics["imag_gradient_mix"] = mix
        else:
            raise NotImplementedError(self._conf.actor_grad)
        if not self._conf.future_entropy and (self._conf.actor_entropy > 0):
            #第三维度调整
            # actor_entropy = self._conf.actor_entropy * actor_ent[:, :, None]
            actor_entropy = self._conf.actor_entropy * actor_ent
            actor_target += actor_entropy
        if not self._conf.future_entropy and (self._conf.actor_state_entropy > 0):
            state_entropy = self._conf.actor_state_entropy * state_ent[:-1]
            actor_target += state_entropy
            metrics["actor_state_entropy"] = to_np(torch.mean(state_entropy))
        actor_loss = -torch.mean(weights[:-1] * actor_target)
        return actor_loss, metrics

    def _update_slow_target(self):
        if self._conf.slow_value_target:
            if self._updates % self._conf.slow_target_update == 0:
                mix = self._conf.slow_target_fraction
                for s, d in zip(self.critic.parameters(), self._slow_value.parameters()):
                    d.data = mix * s.data + (1 - mix) * d.data
            self._updates += 1