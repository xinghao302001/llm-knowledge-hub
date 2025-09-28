from dataclasses import dataclass
from typing import List, Tuple, Union
import time

import numpy as np
import torch
import torch.nn as nn

try:
    import gymnasium as gym  # 兼容 gymnasium
except Exception:
    import gym


# =============== 基础配置 ===============
@dataclass
# [DPO-Δ] 配置改动：移除 PPO 的价值网络/GAE 相关项，新增 β（DPO强度）与基于“episode配对”的 mini-batch 尺寸。
class DPOConfig:
    env_id: str = "CartPole-v1"
    total_steps: int = 200_000
    rollout_steps: int = 4096  # 每轮交互步数上限（将切成若干个完整 episode）
    beta: float = 0.1  # DPO 温度系数  [DPO-Δ]
    pi_lr: float = 3e-4  # 策略学习率
    ent_coef: float = 0.0  # 可选熵正则，缺省 0
    max_grad_norm: float = 0.5
    train_iters: int = 5  # 每个采样阶段的优化轮数（用于DPO对比更新） [DPO-Δ]
    minibatch_pairs: int = (
        64  # mini-batch 中的 episode 对数量  [DPO-Δ] 以“正负episode对”为单位而非单步/GAE样本
    )
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# =============== 工具 ===============
def set_seed(seed: int):
    import random

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# # ============================
# Actor - Critic(Optional)
# ==============================
class ActorCritic(nn.Module):
    """与原 PPO 相同的Actor-Critic (此处 Critic 可选，不参与损失）。"""

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
        self.is_discrete = is_discrete

        layers = []
        last = obs_dim
        for _ in range(self.n_shared_layers):
            layers += [nn.Linear(last, self.hidden_size), nn.Tanh()]
            last = hidden_size
        self.backbone = nn.Sequential(*layers)

        if self.is_discrete:
            act_dim = self.action_space.n
            self.pi_head = nn.Linear(last, act_dim)
            self.log_std = None
        else:
            act_dim = self.action_space.shape[0]
            self.pi_head = nn.Linear(last, act_dim)
            self.log_std = nn.Parameter(torch.zeros(act_dim))

        self.critic = nn.Linear(last, 1)  # 不参与 DPO 损失/优化
        self.apply(self._ortho)

    @staticmethod
    def _ortho(m: nn.Module):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain("tanh"))
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        feat = self.backbone(x)
        logits_or_mu = self.pi_head(feat)
        value = self.critic(feat).squeeze(-1)
        return logits_or_mu, value

    @torch.no_grad()
    def act(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.eval()
        logits_or_mu, _ = self.forward(obs.unsqueeze(0))

        if self.is_discrete:
            dist = torch.distributions.Categorical(logits=logits_or_mu)
            action = dist.sample()
            logp = dist.log_prob(action)
            return int(action.item()), float(logp.item())
        else:
            mu = logits_or_mu.squeeze(0)
            std = self.log_std.exp()
            dist = torch.distributions.Normal(mu, std)
            action = dist.sample()
            logp = dist.log_prob(action).sum(-1)
            return action.cpu().numpy(), float(logp.item())

    def evaluate_actions(
        self, obs_b: torch.Tensor, act_b: torch.Tensor
    ) -> torch.Tensor:
        """给定批量 (s_t, a_t)，计算 log π(a_t|s_t)；离散/连续统一接口。"""
        logits_or_mu, _ = self.forward(obs_b)
        if self.is_discrete:
            dist = torch.distributions.Categorical(logits=logits_or_mu)
            logp = dist.log_prob(act_b)
            return logp  # [B]
        else:
            std = self.log_std.exp()
            dist = torch.distributions.Normal(logits_or_mu, std)
            logp = dist.log_prob(act_b).sum(-1)
            return logp  # [B]


# =============== 采样缓存 ===============
class Episode:
    def __init__(self):
        self.obs: List[np.ndarray] = []
        self.acts: List[Union[int, np.ndarray]] = []  # 离散存 int，连续存 np.ndarray
        self.rets: float = 0.0

    def length(self):
        return len(self.acts)


# [DPO-Δ]：按 episode存储（用于形成偏好对），正负样本进行采样
class RolloutBuffer:
    def __init__(self):
        self.episodes: List[Episode] = []
        self.clear()

    def clear(self):
        self.episodes = []

    def add_episode(self, epi: Episode):
        if epi.length() > 0:
            self.episodes.append(epi)


# # ============================
# DPO Algorithm
# ============================
#
class DPOAgent:
    """[DPO-Δ]: loss函数进行修改, 使用“冻结参考策略 + 对比损失”，无价值网络优化与 PPO 剪切比。"""

    def __init__(
        self,
        cfg: DPOConfig,
        state_dim: int,
        action_space: int,
        is_discrete: bool = True,
    ):
        self.cfg = cfg
        self.state_dim = state_dim
        self.action_space = action_space
        self.is_discrete = is_discrete
        self.device = self.cfg.device
        self.policy = ActorCritic(
            obs_dim=self.state_dim,
            action_space=self.action_space,
            is_discrete=self.is_discrete,
        ).to(self.device)

        ### 冻结参考策略：初始化为当前策略的拷贝
        import copy

        # [DPO-Δ] 冻结参考策略 π_ref：初始化为当前策略的拷贝
        self.ref_policy = copy.deepcopy(self.policy).to(self.device)
        for p in self.ref_policy.parameters():
            p.requires_grad_(False)
        self.ref_policy.eval()

        # [DPO-Δ] 仅优化策略参数（无 value 优化器、无 value_loss）
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=cfg.pi_lr)
        self.buffer = RolloutBuffer()

        # ----- loss 统计属性 ------
        self.stats = {
            "policy_loss": [],
            "total_loss": [],
        }

    def collect_rollout(self, env) -> Tuple[float, float, int]:
        """交互至达到 rollout_steps 或者 episode 数量足够；返回 avg_ret, avg_len, steps。"""
        self.buffer.clear()
        device = self.device

        # reset 兼容 gym/gymnasium
        reset_out = env.reset()
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out

        steps = 0
        ep_returns, ep_lens = [], []
        epi = Episode()

        while steps < self.cfg.rollout_steps:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
            act, logp = self.policy.act(obs_t)

            a_env = act
            if isinstance(a_env, torch.Tensor):
                if getattr(self, "is_discrete", False):
                    a_env = int(a_env.squeeze().item())
                else:
                    a_env = a_env.detach().cpu().numpy()

            # 与环境交互
            step_out = env.step(a_env)
            if len(step_out) == 5:
                next_obs, reward, terminated, truncated, info = step_out
                is_terminated = bool(terminated or truncated)
            else:
                next_obs, reward, is_terminated, info = step_out

            # 记录当前步的状态和动作
            epi.obs.append(np.array(obs, copy=True))
            epi.acts.append(np.array(a_env, copy=True))
            epi.rets += float(reward)

            steps += 1
            obs = next_obs

            if is_terminated:
                ep_returns.append(epi.rets)
                ep_lens.append(len(epi.acts))
                self.buffer.add_episode(epi)

                # 开启新 episode
                reset_out = env.reset()
                obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
                epi = Episode()

                # 若距离 rollout_steps 不足以再跑一个很短 episode，也允许提前退出
                if steps >= self.cfg.rollout_steps:
                    break

        avg_ret = float(np.mean(ep_returns)) if ep_returns else 0.0
        avg_len = float(np.mean(ep_lens)) if ep_lens else 0.0
        return avg_ret, avg_len, steps

    def _episode_logprob(self, model: ActorCritic, epi: Episode) -> torch.Tensor:
        """计算一条 episode 的 log p(y|x)：将每步 log π(a_t|s_t) 相加。"""
        obs_b = torch.tensor(
            np.stack(epi.obs, axis=0), dtype=torch.float32, device=self.device
        )  # [T, obs]
        if model.is_discrete:
            acts_b = torch.tensor(
                np.array(epi.acts), dtype=torch.long, device=self.device
            )  # [T]
        else:
            acts_b = torch.tensor(
                np.stack(epi.acts, axis=0), dtype=torch.float32, device=self.device
            )  # [T, act]
        logps = model.evaluate_actions(obs_b, acts_b)  # [T]
        return (
            logps.sum()
        )  # 标量 DPO 是整条轨迹对比，让「整条更优的轨迹」比差的轨迹有更高的整体概率

    # [DPO-Δ]：根据 episode 总回报构造 (chosen, rejected) 偏好对
    def _build_pairs(self) -> List[Tuple[int, int]]:
        """从已采集 episodes 中采样配对：高回报为正样本，低回报为负样本。
        每对 (pos_i, neg_i) 就是一个 (chosen, rejected) 对
        """

        n = len(self.buffer.episodes)
        if n < 2:
            return []
        # 根据回报排序后，按远近配对，可提升信息量
        idxs = list(range(n))
        ## 按照 episode 的总回报从高到低排序。
        idxs.sort(key=lambda i: self.buffer.episodes[i].rets, reverse=True)
        pairs = []
        half = n // 2
        for i in range(half):
            pos_i = idxs[i]
            neg_i = idxs[i + half]
            if self.buffer.episodes[pos_i].rets > self.buffer.episodes[neg_i].rets:
                pairs.append((pos_i, neg_i))
        return pairs

    def update(self):
        """DPO 对比式更新/CrossEntropy: 对 episode 对进行多轮优化。"""
        pairs = self._build_pairs()
        if not pairs:
            return 0.0, 0

        total_loss = 0.0
        n_steps = 0

        for _ in range(self.cfg.train_iters):
            np.random.shuffle(pairs)
            # mini-batch over pairs
            for start in range(0, len(pairs), self.cfg.minibatch_pairs):
                mb = pairs[start : start + self.cfg.minibatch_pairs]
                if not mb:
                    continue

                pos_logps_pi = []
                neg_logps_pi = []
                pos_logps_ref = []
                neg_logps_ref = []

                for pi_idx, ni_idx in mb:
                    epi_pos = self.buffer.episodes[pi_idx]
                    epi_neg = self.buffer.episodes[ni_idx]

                    # cur_policy 与 ref_policy
                    pos_logps_pi.append(self._episode_logprob(self.policy, epi_pos))
                    neg_logps_pi.append(self._episode_logprob(self.policy, epi_neg))
                    with torch.no_grad():
                        pos_logps_ref.append(
                            self._episode_logprob(self.ref_policy, epi_pos)
                        )
                        neg_logps_ref.append(
                            self._episode_logprob(self.ref_policy, epi_neg)
                        )

                pos_logps_pi = torch.stack(pos_logps_pi)  # [B]
                neg_logps_pi = torch.stack(neg_logps_pi)
                pos_logps_ref = torch.stack(pos_logps_ref)
                neg_logps_ref = torch.stack(neg_logps_ref)

                # [DPO-Δ] 计算 DPO 损失（Logistic）：-log σ(β[(m_π - m_ref)])
                beta = self.cfg.beta
                # [DPO-Δ] DPO 对比边际：当前策略对正/负样 episode 的序列对数似然差
                margin_pi = pos_logps_pi - neg_logps_pi
                # [DPO-Δ] 参考策略边际：用于相对比较，隐式控制 KL
                margin_ref = pos_logps_ref - neg_logps_ref
                logits = beta * (margin_pi - margin_ref)

                loss = -torch.log(torch.sigmoid(logits) + 1e-12).mean()
                self.stats["total_loss"].append(loss.item())
                # 可选熵正则 # TODO:
                if self.cfg.ent_coef > 0:
                    pass

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.cfg.max_grad_norm
                )
                self.optimizer.step()

                # with torch.no_grad():
                #     acc = (
                #         (margin_pi > 0).float().mean().item()
                #     )  # 当前策略是否把正样 episode 打分更高 ##TODO

                total_loss += float(loss.item())
                n_steps += 1

        avg_loss = total_loss / max(n_steps, 1)
        return avg_loss, len(pairs)


# =============== 训练入口（与 PPO 类似的主循环） ===============
def train(cfg: DPOConfig):
    set_seed(cfg.seed)
    env = gym.make(cfg.env_id)
    obs_dim = env.observation_space.shape[0]
    action_space = env.action_space
    print(f"Env: {cfg.env_id}, Obs: {env.observation_space}, Act: {env.action_space}")

    agent = DPOAgent(cfg, obs_dim, action_space)

    steps = 0
    epoch = 0
    t0 = time.time()

    while steps < cfg.total_steps:
        # 1) 采样：与环境交互，按 episode 存储
        avg_ret, avg_len, used = agent.collect_rollout(env)
        steps += used

        # 2) DPO 更新：使用 episode 对进行多轮优化
        avg_loss, n_pairs = agent.update()

        total_loss = np.mean(agent.stats["total_loss"])
        # 清空统计
        for k in agent.stats:
            agent.stats[k] = []
        epoch += 1
        elapsed = time.time() - t0
        print(
            f"[Epoch {epoch}] steps={steps} avg_ret={avg_ret:.2f} avg_len={avg_len:.1f} "
            f"pairs={n_pairs} total_loss={total_loss:.3f}  time={elapsed:.1f}s "
        )

    env.close()


# =============== CLI ===============
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--env", type=str, default="CartPole-v1")
    p.add_argument("--total-steps", type=int, default=200000)
    p.add_argument("--rollout-steps", type=int, default=4096)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--pi-lr", type=float, default=3e-4)
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--train-iters", type=int, default=5)
    p.add_argument("--minibatch-pairs", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args().__dict__

    cfg = DPOConfig(
        env_id=args["env"],
        total_steps=args["total_steps"],
        rollout_steps=args["rollout_steps"],
        beta=args["beta"],
        pi_lr=args["pi_lr"],
        ent_coef=args["ent_coef"],
        max_grad_norm=args["max_grad_norm"],
        train_iters=args["train_iters"],
        minibatch_pairs=args["minibatch_pairs"],
        seed=args["seed"],
    )

    train(cfg)
