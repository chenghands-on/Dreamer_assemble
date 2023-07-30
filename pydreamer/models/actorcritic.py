import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from torch import Tensor

from .functions import *
from .common import *
from.tools_v3 import *

class ActorCritic_v2(nn.Module):

    def __init__(self,
                 in_dim,
                 out_actions,
                 hidden_dim=400,
                 hidden_layers=4,
                 layer_norm=True,
                 gamma=0.999,
                 lambda_gae=0.95,
                 entropy_weight=1e-3,
                 target_interval=100,
                 actor_grad='reinforce',
                 actor_dist='onehot'
                 ):
        super().__init__()
        self.in_dim = in_dim
        self.out_actions = out_actions
        self.gamma = gamma
        self.lambda_ = lambda_gae
        self.entropy_weight = entropy_weight
        self.target_interval = target_interval
        self.actor_grad = actor_grad
        self.actor_dist = actor_dist

        actor_out_dim = out_actions if actor_dist == 'onehot' else 2 * out_actions
        self.actor = MLP(in_dim, actor_out_dim, hidden_dim, hidden_layers, layer_norm)
        self.critic = MLP(in_dim, 1, hidden_dim, hidden_layers, layer_norm)
        self.critic_target = MLP(in_dim, 1, hidden_dim, hidden_layers, layer_norm)
        ## Here is a change!
        # self.critic_target.requires_grad_(False)
        self.critic_target.requires_grad_(True)
        self.train_steps = 0

    def forward_actor(self, features: Tensor) -> D.Distribution:
        y = self.actor.forward(features).float()  # .float() to force float32 on AMP
        
        if self.actor_dist == 'onehot':
            return D.OneHotCategorical(logits=y)
        
        if self.actor_dist == 'normal_tanh':
            return normal_tanh(y)

        if self.actor_dist == 'tanh_normal':
            return tanh_normal(y)

        assert False, self.actor_dist

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
            if self.train_steps % self.target_interval == 0:
                self.update_critic_target()
            self.train_steps += 1

        reward1: TensorHM = rewards[1:]
        terminal0: TensorHM = terminals[:-1]
        terminal1: TensorHM = terminals[1:]

        # GAE from https://arxiv.org/abs/1506.02438 eq (16)
        #   advantage_gae[t] = advantage[t] + (gamma lambda) advantage[t+1] + (gamma lambda)^2 advantage[t+2] + ...

        value_t: TensorJM = self.critic_target.forward(features)
        value0t: TensorHM = value_t[:-1]
        value1t: TensorHM = value_t[1:]
        advantage = - value0t + reward1 + self.gamma * (1.0 - terminal1) * value1t
        advantage_gae = []
        agae = None
        for adv, term in zip(reversed(advantage.unbind()), reversed(terminal1.unbind())):
            if agae is None:
                agae = adv
            else:
                agae = adv + self.lambda_ * self.gamma * (1.0 - term) * agae
            advantage_gae.append(agae)
        advantage_gae.reverse()
        advantage_gae = torch.stack(advantage_gae)
        # Note: if lambda=0, then advantage_gae=advantage, then value_target = advantage + value0t = reward + gamma * value1t
        value_target = advantage_gae + value0t

        # When calculating losses, should ignore terminal states, or anything after, so:
        #   reality_weight[i] = (1-terminal[0]) (1-terminal[1]) ... (1-terminal[i])
        # Note this takes care of the case when initial state features[0] is terminal - it will get weighted by (1-terminals[0]).
        reality_weight = (1 - terminal0).log().cumsum(dim=0).exp()

        # Critic loss

        value: TensorJM = self.critic.forward(features)
        value0: TensorHM = value[:-1]
        loss_critic = 0.5 * torch.square(value_target.detach() - value0)
        loss_critic = (loss_critic * reality_weight).mean()

        # Actor loss

        policy_distr = self.forward_actor(features[:-1])  # TODO: we could reuse this from dream()
        if self.actor_grad == 'reinforce':
            action_logprob = policy_distr.log_prob(actions)
            loss_policy = - action_logprob * advantage_gae.detach()
        elif self.actor_grad == 'dynamics':
            loss_policy = - value_target
        else:
            assert False, self.actor_grad

        policy_entropy = policy_distr.entropy()
        loss_actor = loss_policy - self.entropy_weight * policy_entropy
        loss_actor = (loss_actor * reality_weight).mean()
        assert (loss_policy.requires_grad and policy_entropy.requires_grad) or not loss_critic.requires_grad

        with torch.no_grad():
            metrics = dict(loss_critic=loss_critic.detach(),
                           loss_actor=loss_actor.detach(),
                           policy_entropy=policy_entropy.mean(),
                           policy_value=value0[0].mean(),  # Value of real states
                           policy_value_im=value0.mean(),  # Value of imagined states
                           policy_reward=reward1.mean(),
                           policy_reward_std=reward1.std(),
                           )
            tensors = dict(value=value.detach(),
                           value_target=value_target.detach(),
                           value_advantage=advantage.detach(),
                           value_advantage_gae=advantage_gae.detach(),
                           value_weight=reality_weight.detach(),
                           )

        return (loss_actor, loss_critic), metrics, tensors

    def update_critic_target(self):
        self.critic_target.load_state_dict(self.critic.state_dict())  # type: ignore

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
    def __init__(self, config,stop_grad_actor=True, reward=None):
        super(ActorCritic_v3, self).__init__()
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        # self._world_model = world_model
        self._stop_grad_actor = stop_grad_actor
        self._reward = reward
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
        # 给出actor和critic
        self.actor = networks.ActionHead(
            feat_size,
            config.num_actions,
            config.actor_layers,
            config.units,
            config.act,
            config.norm,
            config.actor_dist,
            config.actor_init_std,
            config.actor_min_std,
            config.actor_max_std,
            config.actor_temp,
            outscale=1.0,
            unimix_ratio=config.action_unimix_ratio,
        )
        if config.value_head == "symlog_disc":
            self.value = networks.MLP(
                feat_size,
                (255,),
                config.value_layers,
                config.units,
                config.act,
                config.norm,
                config.value_head,
                outscale=0.0,
                device=config.device,
            )
        else:
            self.value = networks.MLP(
                feat_size,
                [],
                config.value_layers,
                config.units,
                config.act,
                config.norm,
                config.value_head,
                outscale=0.0,
                device=config.device,
            )
        if config.slow_value_target:
            self._slow_value = copy.deepcopy(self.value)
            self._updates = 0
        kw = dict(wd=config.weight_decay, opt=config.opt, use_amp=self._use_amp)
        self._actor_opt = Optimizer(
            "actor",
            self.actor.parameters(),
            config.actor_lr,
            config.ac_opt_eps,
            config.actor_grad_clip,
            **kw,
        )
        self._value_opt = Optimizer(
            "value",
            self.value.parameters(),
            config.value_lr,
            config.ac_opt_eps,
            config.value_grad_clip,
            **kw,
        )
        if self._config.reward_EMA:
            self.reward_ema = RewardEMA(device=self._config.device)

    def training_step(
        self,
        # start,
        # objective=None,
        # action=None,
        # reward=None,
        # imagine=None,
        # tape=None,
        # repeats=None,
        features: TensorJMF,
        actions: TensorHMA,
        rewards: TensorJM,
        terminals: TensorJM,
        log_only=False
    ):
        objective = objective or self._reward
        self._update_slow_target()
        metrics = {}

        with RequiresGrad(self.actor):
            with torch.cuda.amp.autocast(self._use_amp):
                # features, states, actions = self._imagine(
                #     start, self.actor, self._config.imag_horizon, repeats
                # )
                reward = objective(features, states, actions)
                actor_ent = self.actor(features).entropy()
                state_ent = self._world_model.dynamics.get_dist(states).entropy()
                # this target is not scaled
                # slow is flag to indicate whether slow_target is used for lambda-return
                target, weights, base = self._compute_target(
                    features, states, actions, reward, actor_ent, state_ent
                )
                
                #actor_loss
                actor_loss, mets = self._compute_actor_loss(
                    features,
                    states,
                    actions,
                    target,
                    actor_ent,
                    state_ent,
                    weights,
                    base,
                )
                metrics.update(mets)
                value_input = features

        with RequiresGrad(self.value):
            with torch.cuda.amp.autocast(self._use_amp):
                value = self.value(value_input[:-1].detach())
                target = torch.stack(target, dim=1)
                # (time, batch, 1), (time, batch, 1) -> (time, batch)
                # value_loss
                value_loss = -value.log_prob(target.detach())
                slow_target = self._slow_value(value_input[:-1].detach())
                if self._config.slow_value_target:
                    value_loss = value_loss - value.log_prob(
                        slow_target.mode().detach()
                    )
                if self._config.value_decay:
                    value_loss += self._config.value_decay * value.mode()
                # (time, batch, 1), (time, batch, 1) -> (1,)
                value_loss = torch.mean(weights[:-1] * value_loss[:, :, None])

        metrics.update(tensorstats(value.mode(), "value"))
        metrics.update(tensorstats(target, "target"))
        metrics.update(tensorstats(reward, "imag_reward"))
        if self._config.actor_dist in ["onehot"]:
            metrics.update(
                tensorstats(
                    torch.argmax(actions, dim=-1).float(), "actions"
                )
            )
        else:
            metrics.update(tensorstats(actions, "actions"))
        metrics["actor_entropy"] = to_np(torch.mean(actor_ent))
        with RequiresGrad(self):
            # metrics.update(self._actor_opt(actor_loss, self.actor.parameters()))
            # metrics.update(self._value_opt(value_loss, self.value.parameters()))
            metrics["actor_loss"] = actor_loss.detach().cpu().numpy()
            metrics["value_loss"] = value_loss.detach().cpu().numpy()
        return (actor_loss,value_loss),features, states, actions, weights, metrics

    def _imagine(self, start, policy, horizon, repeats=None):
        dynamics = self._world_model.dynamics
        if repeats:
            raise NotImplemented("repeats is not implemented in this version")
        flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
        start = {k: flatten(v) for k, v in start.items()}

        def step(prev, _):
            state, _, _ = prev
            feat = dynamics.get_feat(state)
            inp = feat.detach() if self._stop_grad_actor else feat
            action = policy(inp).sample()
            succ = dynamics.img_step(state, action, sample=self._config.imag_sample)
            return succ, feat, action

        succ, feats, actions = static_scan(
            step, [torch.arange(horizon)], (start, None, None)
        )
        states = {k: torch.cat([start[k][None], v[:-1]], 0) for k, v in succ.items()}
        if repeats:
            raise NotImplemented("repeats is not implemented in this version")

        return feats, states, actions

    def _compute_target(
        self, features, states, actions, reward, actor_ent, state_ent
    ):
        if "cont" in self._world_model.heads:
            inp = self._world_model.dynamics.get_feat(states)
            discount = self._config.discount * self._world_model.heads["cont"](inp).mean
        else:
            discount = self._config.discount * torch.ones_like(reward)
        if self._config.future_entropy and self._config.actor_entropy() > 0:
            reward += self._config.actor_entropy() * actor_ent
        if self._config.future_entropy and self._config.actor_state_entropy() > 0:
            reward += self._config.actor_state_entropy() * state_ent
        value = self.value(features).mode()
        target = lambda_return(
            reward[1:],
            value[:-1],
            discount[1:],
            bootstrap=value[-1],
            lambda_=self._config.discount_lambda,
            axis=0,
        )
        weights = torch.cumprod(
            torch.cat([torch.ones_like(discount[:1]), discount[:-1]], 0), 0
        ).detach()
        return target, weights, value[:-1]

    def _compute_actor_loss(
        self,
        features,
        states,
        actions,
        target,
        actor_ent,
        state_ent,
        weights,
        base,
    ):
        metrics = {}
        inp = features.detach() if self._stop_grad_actor else features
        policy = self.actor(inp)
        actor_ent = policy.entropy()
        # Q-val for actor is not transformed using symlog
        target = torch.stack(target, dim=1)
        if self._config.reward_EMA:
            offset, scale = self.reward_ema(target)
            normed_target = (target - offset) / scale
            normed_base = (base - offset) / scale
            adv = normed_target - normed_base
            metrics.update(tensorstats(normed_target, "normed_target"))
            values = self.reward_ema.values
            metrics["EMA_005"] = to_np(values[0])
            metrics["EMA_095"] = to_np(values[1])

        if self._config.imag_gradient == "dynamics":
            actor_target = adv
        elif self._config.imag_gradient == "reinforce":
            actor_target = (
                policy.log_prob(actions)[:-1][:, :, None]
                * (target - self.value(features[:-1]).mode()).detach()
            )
        elif self._config.imag_gradient == "both":
            actor_target = (
                policy.log_prob(actions)[:-1][:, :, None]
                * (target - self.value(features[:-1]).mode()).detach()
            )
            mix = self._config.imag_gradient_mix()
            actor_target = mix * target + (1 - mix) * actor_target
            metrics["imag_gradient_mix"] = mix
        else:
            raise NotImplementedError(self._config.imag_gradient)
        if not self._config.future_entropy and (self._config.actor_entropy() > 0):
            actor_entropy = self._config.actor_entropy() * actor_ent[:-1][:, :, None]
            actor_target += actor_entropy
        if not self._config.future_entropy and (self._config.actor_state_entropy() > 0):
            state_entropy = self._config.actor_state_entropy() * state_ent[:-1]
            actor_target += state_entropy
            metrics["actor_state_entropy"] = to_np(torch.mean(state_entropy))
        actor_loss = -torch.mean(weights[:-1] * actor_target)
        return actor_loss, metrics

    def _update_slow_target(self):
        if self._config.slow_value_target:
            if self._updates % self._config.slow_target_update == 0:
                mix = self._config.slow_target_fraction
                for s, d in zip(self.value.parameters(), self._slow_value.parameters()):
                    d.data = mix * s.data + (1 - mix) * d.data
            self._updates += 1