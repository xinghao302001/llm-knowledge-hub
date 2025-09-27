from typing import Tuple
from dataclasses import dataclass
import argparse
import time

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

try:
    import gymnasium as gym  ## 连续动作示例
except Exception:
    import gym  ## 离散动作示例


def set_seed(seed: int):
    """设置随机种子，便于复现。"""
    import random

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class PPOConfig:
    env_id: str = "CartPole-v1"
    total_steps: int = 200_000
    rollout_steps: int = 2048  # 每次采样的交互步数
    gamma: float = 0.99  # 折扣因子
    gae_lambda: float = 0.95  # GAE 中的 λ
    pi_lr: float = 3e-4  # 策略学习率
    v_lr: float = 1e-3  # 价值函数学习率
    clip_ratio: float = 0.2  # PPO 的剪切阈值 ε
    train_iters: int = 10  # 每个采样阶段，优化的迭代次数（多个 epoch）
    minibatch_size: int = 32  # mini-batch 大小
    vf_coef: float = 0.5  # 值函数损失权重
    ent_coef: float = 0.01  # 熵正则权重（鼓励探索）
    max_grad_norm: float = 0.5  # 梯度裁剪
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class RolloutBuffer:
    """用于存放采样得到的轨迹：状态，动作，对数概率，奖励，是否终止，价值估计等"""

    def __init__(self):
        self.clear()

    def add(self, state, action, logprob, reward, is_terminal, value):
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.values.append(value)
        self.rewards.append(reward)
        self.is_terminals.append(is_terminal)

    def clear(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.values = []
        self.is_terminals = []
        self.advantages = []
        self.returns = []


# # ============================
# Actor - Critic
# ============================
class ActorCritic(nn.Module):

    def __init__(
        self,
        obs_dim: int,
        action_space: int,
        hidden_size: int = 64,
        n_shared_layers: int = 2,
        is_discrete: bool = True,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_space = action_space
        self.hidden_size = hidden_size
        self.n_shared_layers = n_shared_layers
        self.is_discrete = is_discrete  # 离散动作空间

        # shared-head
        layers = []
        last = obs_dim
        for _ in range(self.n_shared_layers):
            layers += [nn.Linear(last, self.hidden_size), nn.Tanh()]
            last = self.hidden_size
        self.backbone = nn.Sequential(*layers)

        # actor
        if self.is_discrete:
            act_dim = self.action_space.n
            self.pi_head = nn.Linear(last, act_dim)
            self.log_std = None  # 离散动作不需要方差 ??
        else:
            act_dim = self.action_space
            self.pi_head = nn.Linear(last, act_dim)
            # 使用可学习的对数标准差（全局参数，亦可做成 state-dependent）
            self.log_std = nn.Parameter(torch.zeros(act_dim))

        # critic
        self.critic = nn.Linear(last, 1)
        # self.critic = nn.Sequential(
        #     nn.Linear(self.obs_dim, 64),
        #     nn.Tanh(),
        #     nn.Linear(64, 64),
        #     nn.Tanh(),
        #     nn.Linear(64, 1),
        # )
        self.apply(self.orthogonal_init)

    @staticmethod
    def orthogonal_init(m: nn.Module):
        """对线性层与卷积层做正交初始化，提升稳定性。"""
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain("tanh"))
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        feat = self.backbone(
            x
        )  ## [B, hidden_size] 两头（actor / critic）各接一个线性头
        logits_or_mu = self.pi_head(feat)  # # [B, act_dim]
        value = self.critic(feat).squeeze(-1)  # [B]
        return logits_or_mu, value  ## forward得到的是什么？可以不写吗？

    @torch.no_grad()
    def act(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """基于当前策略进行采样: 返回action, log_prob, state_value"""
        self.eval()
        logits_or_mu, value = self.forward(obs.unsqueeze(0))  # [1,2] , [1]

        if self.is_discrete:
            dist = torch.distributions.Categorical(logits=logits_or_mu)
            action = dist.sample()
            logp = dist.log_prob(action)
        else:
            mu = logits_or_mu.squeeze(0)
            std = self.log_std.exp()
            dist = torch.distributions.Normal(mu, std)
            action = dist.sample()
            logp = dist.log_prob(action).sum(-1)
        return action.squeeze(0), logp.squeeze(0), value.squeeze(0)

    def evaluate_actions(
            self, obs_b: torch.Tensor, act_b: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """在优化阶段, 对一个batch 的 (obs, act) 进行计算：- log_prob(a|s), entropy, V(s)"""
        logits_or_mu, value = self.forward(obs_b)
        if self.is_discrete:
            dist = torch.distributions.Categorical(logits=logits_or_mu)
            logp = dist.log_prob(act_b)
            entropy = dist.entropy()
        else:
            std = self.log_std.exp()
            dist = torch.distributions.Normal(logits_or_mu, std)
            logp = dist.log_prob(act_b).sum(-1)
            entropy = dist.entropy().sum(-1)
        return logp, entropy, value


# # ============================
# PPO Algorithm
# ============================
class PPOAgent:
    def __init__(
        self,
        cfg: PPOConfig,
        state_dim: int,
        action_space: int,
        is_discrete: bool = True,
    ):
        self.cfg = cfg
        self.state_dim = state_dim
        self.action_space = action_space
        self.is_discrete = is_discrete

        # ---- 构建Actor-Critic网络 ----
        self.actor_critic = ActorCritic(
            obs_dim=self.state_dim,
            action_space=self.action_space,
            is_discrete=self.is_discrete,
        )

        # ---- 构建优化器 ----
        actor_params = [
            p for n, p in self.actor_critic.named_parameters() if "critic" not in n
        ]
        critic_params = [
            p for n, p in self.actor_critic.named_parameters() if "critic" in n
        ]
        self.pi_optimizer = torch.optim.Adam(actor_params, lr=cfg.pi_lr)
        self.v_optimizer = torch.optim.Adam(critic_params, lr=cfg.v_lr)
        self.buffer = RolloutBuffer()

        # ----- loss 统计属性 ------
        self.stats = {
            "policy_loss": [],
            "value_loss": [],
            "entropy_loss": [],
            "total_loss": [],
        }

    def _compute_gae(self, last_value: float, done: bool):
        """基于 GAE(λ) 计算 advantage 与 return"""
        rewards = np.array(self.buffer.rewards, dtype=np.float32)
        values = np.array(
            self.buffer.values + [last_value], dtype=np.float32
        )  # 末尾补 V(s_T)
        is_terminals = np.array(self.buffer.is_terminals + [done], dtype=np.float32)

        gae = 0.0
        adv = np.zeros_like(rewards)
        for t in reversed(range(len(rewards))):
            # TD 残差：δ_t = r_t + γ * V(s_{t+1}) * (1 - done_{t}) - V(s_t)
            delta_t = (
                rewards[t]
                + self.cfg.gamma * values[t + 1] * (1 - is_terminals[t])
                - values[t]
            )
            # GAE 递推：A_t = δ_t + γλ(1-done_t) * A_{t+1}
            gae_t = (
                delta_t
                + self.cfg.gamma * self.cfg.gae_lambda * (1 - is_terminals[t]) * gae
            )
            adv[t] = gae_t

        ret = adv + values[:-1]  # 回报（用于训练 critic）

        # 标准化优势，数值更稳定
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        self.buffer.advantages = adv.tolist()
        self.buffer.returns = ret.tolist()

    def update(self):
        """使用 PPO 的剪切目标对策略与价值函数进行多轮优化。"""
        device = self.cfg.device
        old_obs = torch.tensor(
            np.array(self.buffer.states), dtype=torch.float32, device=device
        )

        if self.actor_critic.is_discrete:
            old_acts = torch.tensor(
                np.array(self.buffer.actions), dtype=torch.long, device=device
            )
        else:
            old_acts = torch.tensor(
                np.array(self.buffer.actions), dtype=torch.float32, device=device
            )

        old_logp = torch.tensor(
            np.array(self.buffer.logprobs), dtype=torch.float32, device=device
        )

        old_adv = torch.tensor(
            np.array(self.buffer.advantages), dtype=torch.float32, device=device
        )
        old_ret = torch.tensor(
            np.array(self.buffer.returns), dtype=torch.float32, device=device
        )

        n = old_obs.size(0)
        idxs = np.arange(n)
        for _ in range(self.cfg.train_iters):
            # Mini-batch training
            np.random.shuffle(idxs)
            for start in range(0, n, self.cfg.minibatch_size):
                end = start + self.cfg.minibatch_size
                mb_idx = idxs[start:end]  ## minibatch_idex

                mb_obs = old_obs[mb_idx]
                mb_acts = old_acts[mb_idx]
                mb_old_logp = old_logp[mb_idx]
                mb_adv = old_adv[mb_idx]
                mb_ret = old_ret[mb_idx]

                # 重新计算当前action下 logπ(a|s)、entropy、V(s)
                logp, entropy, value = self.actor_critic.evaluate_actions(
                    mb_obs, mb_acts
                )

                # 重要性采样比 r(θ) = exp(logπ_θ - logπ_θ_old)
                ratio = torch.exp(logp - mb_old_logp)

                # PPO 剪切目标：min(r*A, clip(r, 1-ε, 1+ε)*A)
                surr1 = ratio * mb_adv
                surr2 = (
                    torch.clamp(
                        ratio, 1.0 - self.cfg.clip_ratio, 1.0 + self.cfg.clip_ratio
                    )
                    * mb_adv
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # 值函数损失（可选剪切，本实现使用 MSE
                value_loss = F.mse_loss(value, mb_ret)

                # 熵正则（鼓励更大探索)
                entropy_loss = -entropy.mean()

                # total loss
                loss = (
                    policy_loss
                    + self.cfg.vf_coef * value_loss
                    + self.cfg.ent_coef * entropy_loss
                )
                self.stats["policy_loss"].append(policy_loss.item())
                self.stats["value_loss"].append(value_loss.item())
                self.stats["entropy_loss"].append(entropy_loss.item())
                self.stats["total_loss"].append(loss.item())

                # 反向传播 + 梯度裁剪
                self.pi_optimizer.zero_grad()
                self.v_optimizer.zero_grad()

                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.actor_critic.parameters(), self.cfg.max_grad_norm
                )

                # 注意：两个优化器更新相同的网络参数集合时需小心；可以各自 step。
                self.pi_optimizer.step()
                self.v_optimizer.step()

    def collect_rollout(self, env) -> Tuple[float, float]:
        """交互 rollout_steps 步，收集数据至 buffer与环境交互, 收集一批数据。"""
        self.buffer.clear()
        device = self.cfg.device

        # reset 兼容 gym/gymnasium 的返回
        reset_out = env.reset()
        if isinstance(reset_out, tuple):
            obs, _info = reset_out
        else:
            obs = reset_out

        ep_returns, ep_lens = [], []
        cur_ret, cur_len = 0.0, 0

        for _ in range(self.cfg.rollout_steps):
            # obs -> tensor，并在 act 内部加 batch 维（unsqueeze(0)）
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                action, logp, value = self.actor_critic.act(obs_t)

            if self.actor_critic.is_discrete:
                a_env = int(action.item())  # 离散：标量 int
                a_for_buffer = a_env  # buffer 中保存整型动作

            else:
                a_env = action.detach().cpu().numpy()  # 连续：ndarray(shape=[act_dim])
                a_for_buffer = a_env.copy()  # 避免后续被原地改动

            step_out = env.step(a_env)
            if len(step_out) == 5:
                next_obs, reward, terminated, truncated, info = step_out
                is_terminated = bool(terminated or truncated)
            else:
                next_obs, reward, is_terminated, info = step_out

            self.buffer.add(
                obs.copy(),
                a_for_buffer,
                float(logp.item()),
                float(reward),
                bool(is_terminated),
                float(value.item()),
            )

            # 统计当前 episode 的累计回报与长度
            cur_ret += float(reward)
            cur_len += 1
            obs = next_obs

            if is_terminated:
                ep_returns.append(cur_ret)
                ep_lens.append(cur_len)
                reset_out = env.reset()
                obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
                cur_ret, cur_len = 0.0, 0

        # 用于 GAE 的最后一个状态价值估计 V(s_T)
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
            # forward 返回 (logits_or_mu, value)，取 value 再 item()
            last_value = self.actor_critic.forward(obs_t.unsqueeze(0))[1].item()

        self._compute_gae(last_value=last_value, done=False)

        # 返回采样期统计
        avg_ret = float(np.mean(ep_returns)) if ep_returns else 0.0
        avg_len = float(np.mean(ep_lens)) if ep_lens else 0.0

        return avg_ret, avg_len


# =============== 训练入口 ===============


def train(cfg: PPOConfig):
    set_seed(cfg.seed)

    # 创建环境
    env = gym.make(cfg.env_id)
    state_dim = env.observation_space.shape[0]
    action_space = env.action_space
    print(f"Env: {cfg.env_id}, Obs: {env.observation_space}, Act: {env.action_space}")

    agent = PPOAgent(
        cfg=cfg,
        state_dim=state_dim,
        action_space=action_space,
        is_discrete=True,
    )

    steps = 0
    episode = 0
    t0 = time.time()
    while steps < cfg.total_steps:
        # 1) 采样数据
        avg_ret, avg_len = agent.collect_rollout(env=env)
        steps += cfg.rollout_steps

        # 2) 用 PPO 目标进行多轮优化
        agent.update()

        # 3) 计算并打印平均 loss
        avg_policy_loss = np.mean(agent.stats["policy_loss"])
        avg_value_loss = np.mean(agent.stats["value_loss"])
        avg_entropy_loss = np.mean(agent.stats["entropy_loss"])
        avg_total_loss = np.mean(agent.stats["total_loss"])

        # 清空统计
        for k in agent.stats:
            agent.stats[k] = []

        episode += 1
        elapsed = time.time() - t0
        print(
            f"[Epoch {episode}] "
            f"steps={steps} "
            f"avg_ret={avg_ret:.2f} "
            f"avg_len={avg_len:.1f} "
            f"pi_loss={avg_policy_loss:.3f} "
            f"v_loss={avg_value_loss:.3f} "
            f"ent_loss={avg_entropy_loss:.3f} "
            f"total_loss={avg_total_loss:.3f} "
            f"time={elapsed:.1f}s"
        )

    env.close()


# =============== 命令行参数 ===============


def parse_args() -> PPOConfig:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--env",
        type=str,
        default="CartPole-v1",
        help="环境 ID, 如 CartPole-v1 / Pendulum-v1",
    )
    p.add_argument("--total-steps", type=int, default=200000)
    p.add_argument("--rollout-steps", type=int, default=2048)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--pi-lr", type=float, default=3e-4)
    p.add_argument("--v-lr", type=float, default=1e-3)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--train-iters", type=int, default=10)
    p.add_argument("--minibatch-size", type=int, default=64)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args().__dict__

    cfg = PPOConfig(
        env_id=args["env"],
        total_steps=args["total_steps"],
        rollout_steps=args["rollout_steps"],
        gamma=args["gamma"],
        gae_lambda=args["gae_lambda"],
        pi_lr=args["pi_lr"],
        v_lr=args["v_lr"],
        clip_ratio=args["clip"],
        train_iters=args["train_iters"],
        minibatch_size=args["minibatch_size"],
        vf_coef=args["vf_coef"],
        ent_coef=args["ent_coef"],
        max_grad_norm=args["max_grad_norm"],
        seed=args["seed"],
    )
    return cfg


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
