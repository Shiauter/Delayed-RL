import torch
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torch.nn.functional as F
from torch.distributions import Categorical
import time
from torchviz import make_dot

from actor_control_group import Actor
from util import Memory, clamp
from config_control_group import Config

EPS = 1e-6

class Learner:
    # region config args
    # env
    s_size: int
    a_size: int
    delay: int
    hidden_size: int
    h0: list
    T_horizon: int

    # policy
    gamma: float
    lmbda: float
    critic_weight: float
    entropy_weight: float
    advtg_norm: bool
    gate_reg_weight: float
    gate_reg_weight_to_set: float
    set_gate_reg_weight_at_ep: int
    eps_clip: float
    eps_clip_value: float

    # pred_model
    p_iters: int
    z_size: int
    reconst_loss_method: str
    pause_update_ep: int
    joint_elbo_weight: float
    do_os: bool
    rollout_loss_weight: float

    # training params
    learning_mode: str
    num_memos: int
    epoch_joint: int
    epoch_pred_model: int
    epoch_policy: int
    lr_joint: float
    lr_pred_model: float
    lr_policy: float
    do_lr_sched: bool
    do_draw_graph: bool
    device: str
    # endregion

    def __init__(self, config: Config):
        for key, value in vars(config).items():
            if key in self.__annotations__:
                setattr(self, key, value)

        self.actor: Actor = Actor(config)
        self.optimizers: dict = self._init_optim(config)
        self.schedulers: dict = self._init_sched(config)

    def _make_batch(self, memory: Memory):
        s, a, prob_a, r, s_prime, done, a_lst, v = \
            map(lambda key: torch.tensor(memory.exps[key]).unsqueeze(0).to(self.device).detach(), memory.keys)

        # (batch=1, ep_len, data_size)
        return s, a, r, s_prime, done, prob_a, a_lst, v

    # region pred_model
    def make_pred_s_tis(self, s, a_lsts, h_in):
        s, a_lsts = s.split(1, dim=1), a_lsts.split(1, dim=1)
        s_ti = []
        for i in range(len(s)):
            _, h_out, pred_s = self.actor.pred_present(s[i], a_lsts[i], h_in)
            s_ti.append(pred_s)
            h_in = h_out
        return torch.cat(s_ti, dim=0)

    def make_offset_seq(self, s, offset: tuple, limit):
        # idx = torch.arange(limit).unsqueeze(1) + torch.arange(offset[0], offset[1]).unsqueeze(0)
        # idx = idx.clamp(max=len(s) - 1)
        # target = s[idx].view(offset[1] - offset[0], -1, self.s_size)
        target = s.unfold(dimension=1, size=self.delay+1, step=1).squeeze(0).permute(0, 2, 1)
        target = target[:, 1:, :]
        return target

    def _cal_pred_model_loss(self, s, a_lsts, first_hidden, s_prime):
        limit = s.size(1) - self.actor.delay
        target = self.make_offset_seq(s, (1, self.p_iters + 1), limit)
        pred = self.make_pred_s_tis(s[:, :limit, :], a_lsts[:, :limit, :], first_hidden)
        mse_loss = F.mse_loss(pred, target)
        log = {"mse_loss": mse_loss}
        return log
    # endregion

    # region policy
    def _cal_advantage(self, v_s, r, v_prime, done_mask):
        # v_s       -> (ep_len - delay, 1)
        # r         -> (ep_len, 1)
        # v_prime   -> (ep_len - delay, 1)
        # done_mask -> (ep_len, 1)

        td_target = r[self.delay:] + self.gamma * v_prime * done_mask[self.delay:]
        delta = td_target - v_s

        advtg_lst = []
        advtg_t = 0.0
        for delta_t in reversed(delta):
            advtg_t = self.gamma * self.lmbda * advtg_t + delta_t.item()
            advtg_lst.append([advtg_t])
        advtg_lst.reverse()
        advantage = torch.tensor(advtg_lst, dtype=torch.float).to(self.device)
        if self.advtg_norm:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        return_target = advantage + v_s
        return advantage, return_target # (ep_len, 1)

    def _make_pi_and_critic(self, s, a_lsts, first_hidden):
        # s      -> (batch=1, ep_len, s_size)
        # a_lsts -> (batch=1, ep_len, delay) # (batch=1, ep_len - 1, delay) when calculating v_prime
        # first_hidden -> (seq_len=1, batch=1, hidden_size)

        h_in = first_hidden
        s, a_lsts = torch.split(s, 1, dim=1), torch.split(a_lsts, 1, dim=1)
        second_hidden = None
        res = {"pi": [], "h_out": [], "v": []}
        for i in range(len(s)):
            if i >= len(s) - self.delay: break
            _, pi, h_out, v = self.actor.sample_action(s[i], a_lsts[i], h_in)
            h_in = h_out
            if second_hidden is None: second_hidden = h_out

            res["pi"].append(pi)
            res["h_out"].append(h_out)
            res["v"].append(v)

        for k in res.keys():
            res[k] = torch.cat(res[k], dim=1).squeeze(0)
        return res["pi"], res["v"], second_hidden

    def _cal_ppo_loss(self, s, s_prime, a, prob_a, r, done, a_lst, first_hidden, v_old):
        # s       -> (batch=1, ep_len, s_size)
        # s_prime -> (batch=1, ep_len, s_size)
        # a       -> (batch=1, ep_len, a_size=1)
        # prob_a  -> (batch=1, ep_len, 1)
        # r       -> (batch=1, ep_len, 1)
        # done    -> (batch=1, ep_len, 1)
        # a_lst   -> (batch=1, ep_len, delay)
        # first_hidden -> (seq_len=1, batch=1, hidden_size)

        # remove batch dimension
        a, prob_a, r, done, v_old = a.squeeze(0), prob_a.squeeze(0), r.squeeze(0), done.squeeze(0), v_old.squeeze(0)
        empty_a_lst = torch.zeros_like(a_lst[0]).unsqueeze(0)
        a_lst_prime = torch.cat([a_lst[:, 1:], empty_a_lst], dim=1)

        pi, v_s, second_hidden = self._make_pi_and_critic(s, a_lst, first_hidden)
        _ , v_prime, _ = self._make_pi_and_critic(s_prime, a_lst_prime, second_hidden)
        advantage, return_target = self._cal_advantage(v_s, r, v_prime, done)

        pi_a, prob_a = pi.gather(1, a[self.delay:]), prob_a[:len(prob_a) - self.delay]
        log_pi_a, log_prob_a = torch.log(pi_a.clamp_min(EPS)), torch.log(prob_a.clamp_min(EPS))
        ratio = torch.exp(log_pi_a - log_prob_a)  # a/b == exp(log(a)-log(b))

        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
        policy_loss = -torch.min(surr1, surr2).mean() # expected value

        clipped_mask = (ratio < 1 - self.eps_clip) | (ratio > 1 + self.eps_clip)
        num_clipped = clipped_mask.sum()
        clipped_percentage = num_clipped / ratio.shape[0]
        clipped_ratio = ratio[clipped_mask]
        clipped_distances = torch.maximum(
            clipped_ratio - (1 + self.eps_clip),
            (1 - self.eps_clip) - clipped_ratio
        ).abs()
        avg_clipped_distance = clipped_distances.mean() if num_clipped > 0 else torch.tensor(0.0)

        v_old = v_old[:len(v_old) - self.delay]
        v_clip = v_old +(v_s - v_old).clamp(-self.eps_clip_value, self.eps_clip_value)
        vf1 = F.smooth_l1_loss(v_s, return_target.detach())
        vf2 = F.smooth_l1_loss(v_clip, return_target.detach())
        critic_loss = self.critic_weight * torch.max(vf1, vf2).mean()
        delta_v = v_s - v_old
        v_clipped_mask = (delta_v.abs() > self.eps_clip_value)
        v_clipped_persentage = v_clipped_mask.float().mean()
        avg_clipped_value = torch.clamp(delta_v.abs() - self.eps_clip_value, min=0).mean()

        entropy = Categorical(pi).entropy().mean()
        entropy_bonus = -self.entropy_weight * entropy

        with torch.no_grad():
            kld_policy = (log_prob_a - log_pi_a).mean()
            advtg_mean = advantage.mean()
            advtg_std = advantage.std()
            advtg_min, advtg_max = advantage.min(), advantage.max()

            v_s_std = v_s.std()
            v_s_mean = v_s.mean()
            td_target_std = return_target.std()
            td_target_mean = return_target.mean()

        log = {
            "policy_loss": policy_loss,
            "critic_loss": critic_loss,
            "entropy_bonus": entropy_bonus,
            "kld_policy": kld_policy,
            "clipped_percentage": clipped_percentage,
            "avg_clipped_distance": avg_clipped_distance,
            "advtg_mean": advtg_mean,
            "advtg_std": advtg_std,
            "advtg_min": advtg_min,
            "advtg_max": advtg_max,
            "v_s_std": v_s_std,
            "v_s_mean": v_s_mean,
            "td_target_std": td_target_std,
            "td_target_mean": td_target_mean,
            "v_clipped_persentage": v_clipped_persentage,
            "avg_clipped_value": avg_clipped_value
        }

        return log
    # endregion

    # region training
    def learn(self, memory_list: list[Memory], current_episode: int):
        # pred_model_loss, ppo_loss, total_loss are added in training function
        keys = [
            # pred model
            "mse_loss",

            # policy
            "policy_loss", "critic_loss",
            "entropy_bonus", "kld_policy", "advtg_mean",
            "clipped_percentage", "avg_clipped_distance",
            "advtg_std", "advtg_min", "advtg_max", "v_s_std",
            "td_target_std", "v_s_mean", "td_target_mean",
            "v_clipped_persentage", "avg_clipped_value"
        ]

        log = {}
        for k in keys:
            log[k] = []

        # region joint
        if self.learning_mode == "joint":
            log["total_loss"] = []
            for epoch in range(self.epoch_joint):
                total_pred_model_loss, total_ppo_loss = [], []
                for i in range(self.num_memos):
                    s, a, r, s_prime, done, prob_a, a_lst, v = self._make_batch(memory_list[i])
                    first_hidden = memory_list[i].h0.detach().to(self.device)

                    log_pred_model = self._cal_pred_model_loss(s, a_lst, first_hidden, s_prime)
                    total_pred_model_loss.append(log_pred_model["mse_loss"])

                    for k in log_pred_model.keys():
                        log[k].append(log_pred_model[k])

                    log_policy = self._cal_ppo_loss(s, s_prime, a, prob_a, r, done, a_lst, first_hidden, v)
                    ppo_loss = log_policy["policy_loss"] + log_policy["critic_loss"] + log_policy["entropy_bonus"]
                    total_ppo_loss.append(ppo_loss)

                    for k in log_policy.keys():
                        log[k].append(log_policy[k])

                total_pred_model_loss = torch.stack(total_pred_model_loss).mean()
                total_ppo_loss = torch.stack(total_ppo_loss).mean()
                total_loss = total_ppo_loss + self.joint_elbo_weight * total_pred_model_loss
                log["total_loss"].append(total_loss)

                self.optimizers["joint"].zero_grad()
                total_loss.mean().backward()
                self.optimizers["joint"].step()

            avg_loss_str = f"total -> {total_loss:.6f}"
        # endregion

        # region separate
        elif self.learning_mode == "separate":
            log["pred_model_loss"] = []
            log["ppo_loss"] = []

            check_point = time.time()
            for epoch in range(self.epoch_policy):
                total_ppo_loss = []
                for i in range(self.num_memos):
                    s, a, r, s_prime, done, prob_a, a_lst, v = self._make_batch(memory_list[i])
                    first_hidden = memory_list[i].h0.detach().to(self.device)

                    log_policy = self._cal_ppo_loss(s, s_prime, a, prob_a, r, done, a_lst, first_hidden, v)
                    ppo_loss = log_policy["policy_loss"] + log_policy["critic_loss"] + log_policy["entropy_bonus"]
                    total_ppo_loss.append(ppo_loss)

                    for k in log_policy.keys():
                        log[k].append(log_policy[k])

                total_ppo_loss = torch.stack(total_ppo_loss).mean()
                log["ppo_loss"].append(total_ppo_loss)
                # print(f"total_policy_loss: {total_ppo_loss} / ep. {epoch + 1}")

                self._draw_graph(total_ppo_loss, self.actor, "ppo_loss")
                self.optimizers["policy"].zero_grad()
                if "rnn" in self.optimizers: self.optimizers["rnn"].zero_grad()
                total_ppo_loss.mean().backward()
                self.optimizers["policy"].step()
                if "rnn" in self.optimizers: self.optimizers["rnn"].step()
            # print(f"policy training: {time.time() - check_point} sec.")

            check_point = time.time()
            if self.actor.used_method == "P_Model" and self.delay > 0:
                for epoch in range(self.epoch_pred_model):
                    total_pred_model_loss = []
                    rollout_input_all = []
                    for i in range(self.num_memos):
                        s, a, r, s_prime, done, prob_a, a_lst, _ = self._make_batch(memory_list[i]) # (batch=1, seq_len, data_size)
                        first_hidden = memory_list[i].h0.detach().to(self.device) # (seq_len, batch, hidden_size)

                        log_pred_model = self._cal_pred_model_loss(s, a_lst, first_hidden, s_prime)
                        total_pred_model_loss.append(log_pred_model["mse_loss"])

                        for k in log.keys():
                            if k in log_pred_model:
                                log[k].append(log_pred_model[k])

                    total_pred_model_loss = torch.stack(total_pred_model_loss).mean()
                    log["pred_model_loss"].append(total_pred_model_loss)
                    # print(f"total_pred_model_loss: {total_pred_model_loss} / ep. {epoch + 1}")

                    if self.pause_update_ep is None or current_episode <= self.pause_update_ep:
                        self._draw_graph(total_pred_model_loss, self.actor, "pred_model_loss")
                        self.optimizers["pred_model"].zero_grad()
                        self.optimizers["rnn"].zero_grad()
                        total_pred_model_loss.mean().backward()
                        torch.nn.utils.clip_grad_norm_(self.actor.pred_model.parameters(), 1.0)
                        self.optimizers["pred_model"].step()
                        self.optimizers["rnn"].step()
            else:
                total_pred_model_loss = torch.tensor(0)

            # print(f"pred_model training: {time.time() - check_point} sec.")
            avg_loss_str = f"pred_model->{total_pred_model_loss:.6f}, policy->{total_ppo_loss:.6f}"
        # endregion

        for k in log.keys():
            try:
                if len(log[k]) > 0:
                    log[k] = torch.stack(log[k]).mean()
                else:
                    log[k] = None
            except Exception as err:
                print()
                print(err, k)
        return log, avg_loss_str
    # endregion

    # region learner utils
    def _init_optim(self, config: Config):
        optimizers = {}
        if self.learning_mode == "joint":
            params = []
            if hasattr(self.actor, "rnn"): params.append({"params": self.actor.rnn.parameters()})
            if hasattr(self.actor, "policy"): params.append({"params": self.actor.policy.parameters()})
            if hasattr(self.actor, "pred_model"): params.append({"params": self.actor.pred_model.parameters()})
            optimizers["joint"] = optim.Adam(params, lr=self.lr_joint)

        elif self.learning_mode == "separate":
            if hasattr(self.actor, "pred_model"):
                optimizers["pred_model"] = optim.Adam(
                    [
                        {"params": self.actor.pred_model.parameters()},
                    ],
                    lr=self.lr_pred_model
                )
            if hasattr(self.actor, "policy"):
                optimizers["policy"] = optim.Adam(
                    [
                        {"params": self.actor.policy.parameters()}
                    ],
                    lr=self.lr_policy
                )
            if hasattr(self.actor, "rnn"):
                optimizers["rnn"] = optim.Adam(
                    [
                        {"params": self.actor.rnn.parameters()}
                    ],
                    lr=self.lr_pred_model
                )
        else:
            raise ValueError(f"Unknown learning_mode: {config.learning_mode}")

        return optimizers

    def _init_sched(self, config: Config):
        schedulers = {}
        if self.learning_mode == "joint":
            schedulers["joint"] = sched.ReduceLROnPlateau(
                self.optimizers["joint"], mode='min', factor=0.5, patience=6,
                threshold=1e-3, threshold_mode='rel',
                cooldown=2, min_lr=5e-5
            )
        elif self.learning_mode == "separate":
            if "pred_model" in self.optimizers:
                schedulers["pred_model"] = sched.ReduceLROnPlateau(
                    self.optimizers["pred_model"], mode='min', factor=0.5, patience=6,
                    threshold=1e-3, threshold_mode='rel',
                    cooldown=2, min_lr=5e-5
                )
            schedulers["policy"] = sched.LambdaLR(
                self.optimizers["policy"],
                lr_lambda=lambda u: max(0.5, 1.0 - u / config.K_epoch_training)
            )
        else:
            raise ValueError(f"Unknown learning_mode: {config.learning_mode}")

        return schedulers

    def sched_step(self, loss_log: dict, ep: int):
        if self.learning_mode == "separate":
            if self.do_lr_sched:
                nll_loss, kld_policy, clipfrac = \
                    loss_log["nll_loss"], loss_log["kld_policy"], loss_log["avg_clipped_distance"]

                self.schedulers["pred_model"].step(nll_loss)
                # self.schedulers["policy"].step()
                self._lr_adapter_policy(kld_policy, clipfrac)
            out = {"lr_policy": self.optimizers["policy"].param_groups[0]['lr']}
            if "pred_model" in self.optimizers:
                out["lr_pred_model"] = self.optimizers["pred_model"].param_groups[0]['lr']
            return out
    def _lr_adapter_policy(self, kld, clipfrac,
            target_kl=0.01, low=5e-5, high=1e-3, down=0.8, up=1.2):
        for g in self.optimizers["policy"].param_groups:
            lr = g["lr"]
            if kld < 0.5 * target_kl and clipfrac < 0.02:
                lr *= up
            elif kld > 1.5 * target_kl or clipfrac > 0.04:
                lr *= down
            g["lr"] = min(max(lr, low), high)

    def _collect_param_dict(self, actor: Actor):
        param_dict = {}

        submods = []
        if hasattr(actor, "pred_model"): submods.append(("pred", actor.pred_model))
        if hasattr(actor, "rnn"):        submods.append(("rnn", actor.rnn))
        if hasattr(actor, "policy"):     submods.append(("policy", actor.policy))

        for prefix, mod in submods:
            if hasattr(mod, "named_parameters"):
                for n, p in mod.named_parameters(recurse=True):
                    param_dict[f"{prefix}.{n}"] = p
        return param_dict

    def _draw_graph(self, loss, actor, fname):
        if self.do_draw_graph:
            params = self._collect_param_dict(actor)
            dot = make_dot(loss, params=params)
            dot.save(f"{fname}.dot")
    # endregion