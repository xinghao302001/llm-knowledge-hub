from typing import Tuple, List
from dataclasses import dataclass
import argparse
import time
import copy

import numpy as np
import torch
import torch.nn as nn

try:
    import gymnasium as gym  ## 连续动作示例
except Exception:
    import gym  ## 离散动作示例


def set_seed(seed: int):
    import random
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================
# GRPO Config
# =========================
@dataclass
class GRPOConfig:
    # 与 PPO 通用
    env_id: str = "CartPole-v1"
    total_steps: int = 200_000
    rollout_steps: int = 2048
    pi_lr: float = 3e-4
    clip_ratio: float = 0.2
    train_iters: int = 10
    minibatch_size: int = 64
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # [GRPO-Δ] 
    group_size: int = 4  # 每多少个 episode 构成一组
    norm_in_group: bool = True  # 是否按组内标准差再标准化
    kl_ref_coef: float = 0.02  # KL 正则系数 β
    freeze_ref_after_init: bool = True  # 引用策略是否固定为初始策略快照


# =========================
# Rollout Buffer
# =========================
class RolloutBuffer:
    """
    [GRPO-Δ] Buffer 中不再存 value / return / GAE，
    而是记录 episode 的分段索引，便于按 episode 求总回报，再按组构造相对优势。
    """

    def __init__(self):
        self.clear()

    def add(self, state, action, logprob, reward, is_terminal):
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.is_terminals.append(is_terminal)

    def clear(self):
        self.states: List[np.ndarray] = []
        self.actions: List = []
        self.logprobs: List[float] = []
        self.rewards: List[float] = []
        self.is_terminals: List[bool] = []
        self.episodes: List[Tuple[int, int]] = []
        self.g_advantages: List[float] = []  ## [GRPO-Δ] 组内相对优势
        self._ep_start: int = 0

    def end_episode(self):
        # 记录一个 episode 的 [start, end) 区间
        if not self._ep_start:
            self._ep_start = 0
        ep_end = len(self.states)
        self.episodes.append((self._ep_start, ep_end))
        self._ep_start = ep_end


# =========================
# Actor (no explicit Critic)
# =========================
class ActorCritic(nn.Module):
    """
    [GRPO-Δ] 与 PPO 的 ActorCritic 相比：
    - 移除 critic 头与正交初始化对 critic 的使用
    - 只保留策略头（logits / 均值），用于计算 logπ(a|s)
    """

    def __init__(
        self,
        obs_dim: int,
        action_space,
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
            last = self.hidden_size
        self.backbone = nn.Sequential(*layers)

        if self.is_discrete:
            act_dim = self.action_space.n
            self.pi_head = nn.Linear(last, act_dim)
            self.log_std = None
        else:
            act_dim = self.action_space
            self.pi_head = nn.Linear(last, act_dim)
            self.log_std = nn.Parameter(torch.zeros(act_dim))
        self.apply(self.orthogonal_init)

    @staticmethod
    def orthogonal_init(m: nn.Module):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain("tanh"))
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        feat = self.backbone(x)
        logits_or_mu = self.pi_head(feat)
        return logits_or_mu

    @torch.no_grad()
    def act(self, obs: torch.Tensor):
        self.eval()
        logits_or_mu = self.forward(obs.unsqueeze(0))  # [1, act_dim]
        if self.is_discrete:
            dist = torch.distributions.Categorical(logits=logits_or_mu)
            action = dist.sample()
            logp = dist.log_prob(action)
        else:
            mu = logits_or_mu.squeeze(0)  # TODO: if having no 1.dim
            std = self.log_std.exp()
            dist = torch.distributions.Normal(mu, std)
            action = dist.sample()
            logp = dist.log_prob(action).sum(-1)
        return action.squeeze(0), logp.squeeze(0)

    def evaluate_actions(
        self, obs_b: torch.Tensor, act_b: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """在优化阶段, 对一个batch 的 (obs, act) 进行计算：- log_prob(a|s), entropy, V(s)"""
        logits_or_mu = self.forward(obs_b)

        if self.is_discrete:
            dist = torch.distributions.Categorical(logits=logits_or_mu)
            logp = dist.log_prob(act_b)
            entropy = dist.entropy()
        else:
            std = self.log_std.exp()
            dist = torch.distributions.Normal(logits_or_mu, std)
            logp = dist.log_prob(act_b).sum(-1)
            entropy = dist.entropy().sum(-1)
        return logp, entropy


# =========================
# GRPO Agent
# =========================
class GRPOAgent:
    """
    [GRPO-Δ] 与PPO的关键差异总结:
    1) 无 critic, 无 GAE, (也无 v_loss、v_optimizer)
    2) 以 episode 为单位计算“分组相对优势”: A_i = R_i - mean(R_group) / std(R_group)
    3) 用 PPO 的剪切目标优化策略，但优势来自分组相对优势
    4) 额外引入引用策略 (ref_policy) 并加 KL(π_new || π_ref) 正则，约束策略漂移
    """

    def __init__(
        self,
        cfg: GRPOConfig,
        state_dim: int,
        action_space,
        is_discrete: bool = True,
    ):
        self.cfg = cfg
        self.state_dim = state_dim
        self.action_space = action_space
        self.is_discrete = is_discrete

        self.policy = ActorCritic(
            obs_dim=self.state_dim,
            action_space=action_space,
            is_discrete=self.is_discrete,
        ).to(self.cfg.device)

        # [GRPO-Δ] ref策略:训练初始的策略（可选冻结)
        self.ref_policy = copy.deepcopy(self.policy).to(self.cfg.device)
        if self.cfg.freeze_ref_after_init:
            for p in self.ref_policy.parameters():
                p.requires_grad = False

        # 只有pi优化器
        self.pi_optimizer = torch.optim.Adam(self.policy.parameters(), lr=cfg.pi_lr)
        self.buffer = RolloutBuffer()

        self.stats = {
            "policy_loss": [],
            "entropy_loss": [],
            "kl_ref": [],
            "total_loss": [],
            "avg_group_std": [],
        }

    def _finish_episodes_and_build_advantages(self):
        """
        [GRPO-Δ] 计算“分组相对优势”
        步骤：
            1. 先按 self.buffer.episodes 拿到每个 episode 的总回报 R_e
            2. 按顺序每 group_size 个 episode 构成一组，组内: A_e = R_e - mean(R_group) / (std+eps)
            3. 将 A_e 广播给该 episode 内的所有时间步
        """
        eps = 1e-8
        rewards = np.array(self.buffer.rewards, dtype=np.float32)
        episodes = self.buffer.episodes

        # 求每个 episode 的总回报
        ep_returns = []
        for start, end in episodes:
            ep_returns.append(float(np.sum(rewards[start:end])))

        # 分组构造
        adv = np.zeros_like(rewards, dtype=np.float32)
        group_size = self.cfg.group_size
        group_stds = []
        ## 按顺序每 group_size 个 episode 构成一组，组内: A_e = R_e - mean(R_group) / (std+eps)
        for g_start in range(0, len(episodes), group_size):
            g_end = min(g_start + group_size, len(episodes))
            idxs = list(range(g_start, g_end))
            group_rets = np.array([ep_returns[i] for i in idxs], dtype=np.float32)
            g_mean = float(group_rets.mean())
            g_std = float(group_rets.std() + eps)
            group_stds.append(g_std)

            # 组内每个 episode 的相对优势（episode 级别）
            if self.cfg.norm_in_group:
                ep_advs = (group_rets - g_mean) / g_std
            else:
                ep_advs = group_rets - g_mean

            # 广播到各自 episode 的全部时间步
            for k, ep_i in enumerate(idxs):
                start, end = episodes[ep_i]
                adv[start:end] = ep_advs[k]

        self.buffer.g_advantages = adv.tolist()
        if group_stds:
            self.stats["avg_group_std"].append(float(np.mean(group_stds)))

    def update(self):
        """[GRPO-Δ] 仅使用策略目标；优势来自“分组相对优势”；外加 KL(π||π_ref) 正则。"""
        devide = self.cfg.device
        old_obs = torch.tensor(
            np.array(self.buffer.states), dtype=torch.float32, device=self.cfg.device
        )

        if self.is_discrete:
            old_acts = torch.tensor(
                np.array(self.buffer.actions),
                dtype=torch.long,
                device=self.cfg.device,
            )
        else:
            old_acts = torch.tensor(
                np.array(self.buffer.actions),
                dtype=torch.float32,
                device=self.cfg.device,
            )
        old_logp = torch.tensor(
            np.array(self.buffer.logprobs), dtype=torch.float32, device=self.cfg.device
        )
        old_g_adv = torch.tensor(
            np.array(self.buffer.g_advantages),
            dtype=torch.float32,
            device=self.cfg.device,
        )
        old_g_adv = (old_g_adv - old_g_adv.mean()) / (old_g_adv.std() + 1e-8)

        n = old_obs.size(0)
        idxs = np.arange(n)

        for _ in range(self.cfg.train_iters):
            np.random.shuffle(idxs)
            for start in range(0, n, self.cfg.minibatch_size):
                end = start + self.cfg.minibatch_size
                mb_idx = idxs[start:end]

                mb_obs = old_obs[mb_idx]
                mb_acts = old_acts[mb_idx]
                mb_logp = old_logp[mb_idx]
                mb_g_adv = old_g_adv[mb_idx]

                # 当前策略的 logπ 与熵
                new_logp, new_entropy = self.policy.evaluate_actions(mb_obs, mb_acts)

                # [GRPO-Δ] 重要性比 & PPO 剪切仍然保留
                ratio = torch.exp(new_logp - mb_logp)
                surr1 = ratio * mb_g_adv
                surr2 = torch.clamp(
                    ratio, 1.0 - self.cfg.clip_ratio, 1.0 + self.cfg.clip_ratio
                ) * mb_g_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # [GRPO-Δ] 引用策略 KL：approx KL(π_new || π_ref) ≈ E[logπ_new - logπ_ref]
                with (
                    torch.no_grad()
                    if self.cfg.freeze_ref_after_init
                    else torch.enable_grad()
                ):
                    ref_logp, _ = self.ref_policy.evaluate_actions(mb_obs, mb_acts)
                
                kl_ref = (new_logp - ref_logp).mean()
                entropy_loss = -new_entropy.mean()

                loss = (
                    policy_loss + self.cfg.ent_coef * entropy_loss + self.cfg.kl_ref_coef * kl_ref
                )

                self.stats["policy_loss"].append(policy_loss.item())
                self.stats["entropy_loss"].append(entropy_loss.item())
                self.stats["kl_ref"].append(kl_ref.item())
                self.stats["total_loss"].append(loss.item())

                self.pi_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.cfg.max_grad_norm
                )
                self.pi_optimizer.step()

    def collect_rollout(self, env) -> Tuple[float, float]:
        """
        与 PPO 的采样主体类似，但：
        [GRPO-Δ] 在 episode 结束时调用 buffer.end_episode(), 训练前用 _finish_episodes_and_build_advantages() 构造“分组相对优势”.
        """
        self.buffer.clear()
        device = self.cfg.device
        reset_out = env.reset()
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out

        ep_returns, ep_lens = [], []
        cur_ret, cur_len = 0.0, 0
        
        steps = 0
        while steps < self.cfg.rollout_steps:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                action, logp = self.policy.act(obs_t)
            
            if self.is_discrete:
                a_env = int(action.item())
                a_for_buffer = a_env
            else:
                a_env = action.detach().cpu().numpy()
                a_for_buffer = a_env.copy()
            
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
            )

            cur_ret += float(reward)
            cur_len += 1
            obs = next_obs
            steps += 1

            if is_terminated:
                ep_returns.append(cur_ret)
                ep_lens.append(cur_len)
                self.buffer.end_episode() ## [GRPO-Δ]
                reset_out = env.reset()
                obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
                cur_ret, cur_len = 0.0, 0
        
        # 如果最后一个 episode 没终止，也将其记为一个 episode（便于成组)
        ##　len(self.buffer.episodes) == 0: → 表示 rollout 过程中一次 episode 都没被记录过（比如采样中途没有遇到过 done） → 必须手动把整个 buffer 作为一个 episode
        ##  self.buffer.episodes[-1][1] != len(self.buffer.states: -> 如果两者不相等，说明最后一个 episode 没有覆盖到 buffer 的最后一个 step。
        if len(self.buffer.episodes) == 0 or self.buffer.episodes[-1][1] != len(
            self.buffer.states
        ):
            self.buffer.end_episode()
        
        # [GRPO-Δ] 构造分组相对优势
        self._finish_episodes_and_build_advantages()

        # 返回采样期统计
        avg_ret = float(np.mean(ep_returns)) if ep_returns else 0.0
        avg_len = float(np.mean(ep_lens)) if ep_lens else 0.0
        return avg_ret, avg_len


# =========================
# Train
# =========================
def train(cfg: GRPOConfig):
    set_seed(cfg.seed)

    env = gym.make(cfg.env_id)
    state_dim = env.observation_space.shape[0]
    action_space = env.action_space
    print(f"Env: {cfg.env_id}, Obs: {env.observation_space}, Act: {env.action_space}")

    agent = GRPOAgent(
        cfg=cfg,
        state_dim=state_dim,
        action_space=action_space,
        is_discrete=True,
    )

    steps = 0
    epoch = 0
    t0 = time.time()
    while steps < cfg.total_steps:
        avg_ret, avg_len = agent.collect_rollout(env=env)
        steps += cfg.rollout_steps

        agent.update()

        avg_pi_loss = (
            np.mean(agent.stats["policy_loss"]) if agent.stats["policy_loss"] else 0.0
        )
        avg_ent_loss = (
            np.mean(agent.stats["entropy_loss"]) if agent.stats["entropy_loss"] else 0.0
        )
        avg_kl = np.mean(agent.stats["kl_ref"]) if agent.stats["kl_ref"] else 0.0
        avg_total = (
            np.mean(agent.stats["total_loss"]) if agent.stats["total_loss"] else 0.0
        )
        avg_gstd = (
            np.mean(agent.stats["avg_group_std"])
            if agent.stats["avg_group_std"]
            else 0.0
        )

        for k in agent.stats:
            agent.stats[k] = []

        epoch += 1
        elapsed = time.time() - t0
        print(
            f"[Epoch {epoch}] "
            f"steps={steps} "
            f"avg_ret={avg_ret:.2f} "
            f"avg_len={avg_len:.1f} "
            f"pi_loss={avg_pi_loss:.3f} "
            f"ent_loss={avg_ent_loss:.3f} "
            f"kl_ref={avg_kl:.4f} "
            f"group_std={avg_gstd:.3f} "
            f"total_loss={avg_total:.3f} "
            f"time={elapsed:.1f}s"
        )

    env.close()


# =========================
# CLI
# =========================
def parse_args() -> GRPOConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--env", type=str, default="CartPole-v1")
    p.add_argument("--total-steps", type=int, default=200000)
    p.add_argument("--rollout-steps", type=int, default=2048)
    p.add_argument("--pi-lr", type=float, default=3e-4)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--train-iters", type=int, default=10)
    p.add_argument("--minibatch-size", type=int, default=64)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    # [GRPO-Δ]
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--norm-in-group", action="store_true")
    p.add_argument("--kl-ref-coef", type=float, default=0.02)
    p.add_argument("--freeze-ref", action="store_true")

    args = p.parse_args().__dict__
    cfg = GRPOConfig(
        env_id=args["env"],
        total_steps=args["total_steps"],
        rollout_steps=args["rollout_steps"],
        pi_lr=args["pi_lr"],
        clip_ratio=args["clip"],
        train_iters=args["train_iters"],
        minibatch_size=args["minibatch_size"],
        ent_coef=args["ent_coef"],
        max_grad_norm=args["max_grad_norm"],
        seed=args["seed"],
        group_size=args["group_size"],
        norm_in_group=args["norm_in_group"],
        kl_ref_coef=args["kl_ref_coef"],
        freeze_ref_after_init=args["freeze_ref"],
    )
    return cfg


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
