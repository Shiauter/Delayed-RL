import torch
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torch.nn.functional as F
from torch.distributions import Categorical
import time

from actor_vrnn_v2 import Actor
from util import Memory, clamp
from config import Config

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

    # pred_model
    p_iters: int
    z_size: int
    reconst_loss_method: str
    pause_update_ep: int
    joint_elbo_weight: float

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
        s, a, prob_a, r, s_prime, done, t, a_lst = \
            map(lambda key: torch.tensor(memory.exps[key]).unsqueeze(0).to(self.device).detach(), memory.keys)

        # (batch=1, ep_len, data_size)
        return s, a, r, s_prime, done, prob_a, a_lst

    # region pred_model
    def _get_start_h(self, s, h_in):
        # posterior
        # s       -> (batch=1, ep_len - delay, s_size)
        # a_lsts  -> (batch=1, ep_len - delay, delay)
        # h       -> (seq_len=1, batch=1, hidden_size)
        # s_truth -> (delay, ep_len - delay, s_size)
        all_h = []
        pred_h = h_in
        s = torch.split(s, 1, dim=1)
        for i in range(len(s)):
            phi_x, phi_z = self.actor.pred_model(s[i], pred_h)
            cond_in = torch.cat([phi_x, phi_z], dim=-1)
            _, pred_h = self.actor.rnn(cond_in, pred_h)
            all_h.append(pred_h)
        return torch.cat(all_h[:-1], dim=1)

    def _cal_pred_model_loss(self, s, a_lsts, first_hidden, s_prime):
        # s       -> (batch=1, ep_len, s_size) # current s
        # a_lsts  -> (batch=1, ep_len, delay)
        # h       -> (seq_len=1, batch=1, hidden_size)

        # predicting one step future state using true state
        s = torch.cat([s[:, 1:, :], s_prime[:, -1, :].unsqueeze(1)], dim=1).split(1, dim=1)
        a = torch.split(a_lsts, 1, dim=-1)[0].split(1, dim=1) # split into actions
        h = first_hidden

        kld_loss, nll_loss, mse_loss = [], [], []
        dec_std_log = []
        z_vanishing_log = {
            "delta_mse_prior": [],
            "delta_mse_zero": [],
            "delta_mse_shuf": [],
            "delta_nll_prior": [],
            "delta_nll_zero": [],
            "delta_nll_shuf": [],
        }
        for i in range(len(s)):
            kld, nll, mse, phi_x, phi_z, dec_std, z_vanishing_out = self.actor.pred_model.cal_loss(s[i], a[i], h)
            cond_in = torch.cat([phi_x, phi_z], dim=-1)
            _, h = self.actor.rnn(cond_in.detach(), h.detach())

            kld_loss.append(kld.squeeze())
            nll_loss.append(nll.squeeze())
            mse_loss.append(mse.squeeze())
            dec_std_log.append(dec_std)

            for k in z_vanishing_log.keys():
                z_vanishing_log[k].append(z_vanishing_out[k])

        kld_loss, nll_loss, mse_loss = torch.stack(kld_loss).mean(), torch.stack(nll_loss).mean(), torch.stack(mse_loss).mean()
        for k in z_vanishing_log.keys():
            z_vanishing_log[k] = torch.stack(z_vanishing_log[k]).mean()

        dec_std_log = torch.stack(dec_std_log).mean(dim=-1)
        dec_std_mean = dec_std_log.mean()
        dec_std_max = dec_std_log.max()
        dec_std_min = dec_std_log.min()

        return kld_loss, nll_loss, mse_loss, dec_std_mean, dec_std_max, dec_std_min, z_vanishing_log
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

    def _cal_ppo_loss(self, s, s_prime, a, prob_a, r, done, a_lst, first_hidden):
        # s       -> (batch=1, ep_len, s_size)
        # s_prime -> (batch=1, ep_len, s_size)
        # a       -> (batch=1, ep_len, a_size=1)
        # prob_a  -> (batch=1, ep_len, 1)
        # r       -> (batch=1, ep_len, 1)
        # done    -> (batch=1, ep_len, 1)
        # a_lst   -> (batch=1, ep_len, delay)
        # first_hidden -> (seq_len=1, batch=1, hidden_size)

        # remove batch dimension
        a, prob_a, r, done = a.squeeze(0), prob_a.squeeze(0), r.squeeze(0), done.squeeze(0)
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
        cliped_percentage = num_clipped / ratio.shape[0]
        clipped_ratio = ratio[clipped_mask]
        clipped_distances = torch.maximum(
            clipped_ratio - (1 + self.eps_clip),
            (1 - self.eps_clip) - clipped_ratio
        ).abs()
        avg_clipped_distance = clipped_distances.mean() if num_clipped > 0 else torch.tensor(0.0)

        critic_loss = self.critic_weight * F.smooth_l1_loss(v_s, return_target.detach())

        entropy = Categorical(pi).entropy().mean()
        entropy_bonus = -self.entropy_weight * entropy

        with torch.no_grad():
            kl_div = (log_prob_a - log_pi_a).mean()
            advtg_mean = advantage.mean()
            advtg_std = advantage.std()
            advtg_min, advtg_max = advantage.min(), advantage.max()

            v_s_std = v_s.std()
            v_s_mean = v_s.mean()
            td_target_std = return_target.std()
            td_target_mean = return_target.mean()

        return policy_loss, critic_loss, entropy_bonus, kl_div, \
            advtg_mean, cliped_percentage, avg_clipped_distance, \
            advtg_std, advtg_min, advtg_max, v_s_std, td_target_std, \
            v_s_mean, td_target_mean
    # endregion

    # region training
    def learn(self, memory_list: list[Memory], current_episode: int):
        # pred_model_loss, ppo_loss, total_loss are added in training function
        keys = [
            # pred model
            "kld_loss", "nll_loss", "mse_loss",
            "dec_std_mean", "dec_std_max", "dec_std_min",
            "delta_mse_prior", "delta_mse_zero", "delta_mse_shuf",
            "delta_nll_prior", "delta_nll_zero", "delta_nll_shuf",

            # policy
            "policy_loss", "critic_loss",
            "entropy_bonus", "kld_policy", "advtg_mean",
            "clipped_percentage", "avg_clipped_distance",
            "advtg_std", "advtg_min", "advtg_max", "v_s_std",
            "td_target_std", "v_s_mean", "td_target_mean"
        ]

        loss_log = {}
        for k in keys:
            loss_log[k] = []

        # region joint
        if self.learning_mode == "joint":
            loss_log["total_loss"] = []
            for epoch in range(self.epoch_joint):
                total_pred_model_loss, total_ppo_loss = [], []
                for i in range(self.num_memos):
                    s, a, r, s_prime, done, prob_a, a_lst = self._make_batch(memory_list[i])
                    first_hidden = memory_list[i].h0.detach().to(self.device)

                    kld_loss, nll_loss, mse_loss, dec_std_mean, dec_std_max, dec_std_min, z_vanishing_out = self._cal_pred_model_loss(s, a, first_hidden, s_prime)
                    if self.reconst_loss_method == "NLL":
                        total_pred_model_loss.append(kld_loss + nll_loss)
                    elif self.reconst_loss_method == "MSE":
                        total_pred_model_loss.append(kld_loss + mse_loss)

                    loss_log["kld_loss"].append(kld_loss)
                    loss_log["nll_loss"].append(nll_loss)
                    loss_log["mse_loss"].append(mse_loss)
                    loss_log["dec_std_mean"].append(dec_std_mean)
                    loss_log["dec_std_max"].append(dec_std_max)
                    loss_log["dec_std_min"].append(dec_std_min)
                    loss_log["delta_mse_prior"].append(z_vanishing_out["delta_mse_prior"])
                    loss_log["delta_mse_zero"].append(z_vanishing_out["delta_mse_zero"])
                    loss_log["delta_mse_shuf"].append(z_vanishing_out["delta_mse_shuf"])
                    loss_log["delta_nll_prior"].append(z_vanishing_out["delta_nll_prior"])
                    loss_log["delta_nll_zero"].append(z_vanishing_out["delta_nll_zero"])
                    loss_log["delta_nll_shuf"].append(z_vanishing_out["delta_nll_shuf"])


                    policy_loss, critic_loss, entropy_bonus, kld_policy, advtg_mean, \
                    clipped_percentage, avg_clipped_distance, advtg_std, advtg_min, advtg_max, \
                    v_s_std, td_target_std, v_s_mean, td_target_mean = \
                        self._cal_ppo_loss(s, s_prime, a, prob_a, r, done, a_lst, first_hidden)
                    ppo_loss = policy_loss + critic_loss + entropy_bonus
                    total_ppo_loss.append(ppo_loss)

                    loss_log["policy_loss"].append(policy_loss)
                    loss_log["critic_loss"].append(critic_loss)
                    loss_log["entropy_bonus"].append(entropy_bonus)
                    loss_log["kld_policy"].append(kld_policy)
                    loss_log["advtg_mean"].append(advtg_mean)
                    loss_log["clipped_percentage"].append(clipped_percentage)
                    loss_log["avg_clipped_distance"].append(avg_clipped_distance)
                    loss_log["advtg_std"].append(advtg_std)
                    loss_log["advtg_min"].append(advtg_min)
                    loss_log["advtg_max"].append(advtg_max)
                    loss_log["v_s_std"].append(v_s_std)
                    loss_log["td_target_std"].append(td_target_std)
                    loss_log["v_s_mean"].append(v_s_mean)
                    loss_log["td_target_mean"].append(td_target_mean)

                total_pred_model_loss = torch.stack(total_pred_model_loss).mean()
                total_ppo_loss = torch.stack(total_ppo_loss).mean()
                total_loss = total_ppo_loss + self.joint_elbo_weight * total_pred_model_loss
                loss_log["total_loss"].append(total_loss)

                self.optimizers["joint"].zero_grad()
                total_loss.mean().backward()
                self.optimizers["joint"].step()

            avg_loss_str = f"total -> {total_loss:.6f}"
        # endregion

        # region separate
        elif self.learning_mode == "separate":
            loss_log["pred_model_loss"] = []
            loss_log["ppo_loss"] = []

            check_point = time.time()
            for epoch in range(self.epoch_policy):
                total_ppo_loss = []
                for i in range(self.num_memos):
                    s, a, r, s_prime, done, prob_a, a_lst = self._make_batch(memory_list[i])
                    first_hidden = memory_list[i].h0.detach().to(self.device)

                    policy_loss, critic_loss, entropy_bonus, kld_policy, advtg_mean, \
                    clipped_percentage, avg_clipped_distance, advtg_std, advtg_min, advtg_max, \
                    v_s_std, td_target_std, v_s_mean, td_target_mean = \
                        self._cal_ppo_loss(s, s_prime, a, prob_a, r, done, a_lst, first_hidden)
                    ppo_loss = policy_loss + critic_loss + entropy_bonus
                    total_ppo_loss.append(ppo_loss)

                    loss_log["policy_loss"].append(policy_loss)
                    loss_log["critic_loss"].append(critic_loss)
                    loss_log["entropy_bonus"].append(entropy_bonus)
                    loss_log["kld_policy"].append(kld_policy)
                    loss_log["advtg_mean"].append(advtg_mean)
                    loss_log["clipped_percentage"].append(clipped_percentage)
                    loss_log["avg_clipped_distance"].append(avg_clipped_distance)
                    loss_log["advtg_std"].append(advtg_std)
                    loss_log["advtg_min"].append(advtg_min)
                    loss_log["advtg_max"].append(advtg_max)
                    loss_log["v_s_std"].append(v_s_std)
                    loss_log["td_target_std"].append(td_target_std)
                    loss_log["v_s_mean"].append(v_s_mean)
                    loss_log["td_target_mean"].append(td_target_mean)
                total_ppo_loss = torch.stack(total_ppo_loss).mean()
                loss_log["ppo_loss"].append(total_ppo_loss)
                # print(f"total_policy_loss: {total_ppo_loss} / ep. {epoch + 1}")

                self.optimizers["policy"].zero_grad()
                total_ppo_loss.mean().backward()
                self.optimizers["policy"].step()
            # print(f"policy training: {time.time() - check_point} sec.")

            check_point = time.time()
            for epoch in range(self.epoch_pred_model):
                total_pred_model_loss = []
                for i in range(self.num_memos):
                    s, a, r, s_prime, done, prob_a, a_lst = self._make_batch(memory_list[i]) # (batch=1, seq_len, data_size)
                    first_hidden = memory_list[i].h0.detach().to(self.device) # (seq_len, batch, hidden_size)

                    kld_loss, nll_loss, mse_loss, dec_std_mean, dec_std_max, dec_std_min, z_vanishing_out = self._cal_pred_model_loss(s, a, first_hidden, s_prime)
                    if self.reconst_loss_method == "NLL":
                        total_pred_model_loss.append(kld_loss + nll_loss)
                    elif self.reconst_loss_method == "MSE":
                        total_pred_model_loss.append(kld_loss + mse_loss)

                    loss_log["kld_loss"].append(kld_loss)
                    loss_log["nll_loss"].append(nll_loss)
                    loss_log["mse_loss"].append(mse_loss)
                    loss_log["dec_std_mean"].append(dec_std_mean)
                    loss_log["dec_std_max"].append(dec_std_max)
                    loss_log["dec_std_min"].append(dec_std_min)
                    loss_log["delta_mse_prior"].append(z_vanishing_out["delta_mse_prior"])
                    loss_log["delta_mse_zero"].append(z_vanishing_out["delta_mse_zero"])
                    loss_log["delta_mse_shuf"].append(z_vanishing_out["delta_mse_shuf"])
                    loss_log["delta_nll_prior"].append(z_vanishing_out["delta_nll_prior"])
                    loss_log["delta_nll_zero"].append(z_vanishing_out["delta_nll_zero"])
                    loss_log["delta_nll_shuf"].append(z_vanishing_out["delta_nll_shuf"])

                total_pred_model_loss = torch.stack(total_pred_model_loss).mean()
                loss_log["pred_model_loss"].append(total_pred_model_loss)
                # print(f"total_pred_model_loss: {total_pred_model_loss} / ep. {epoch + 1}")

                if self.pause_update_ep is None or current_episode <= self.pause_update_ep:
                    self.optimizers["pred_model"].zero_grad()
                    total_pred_model_loss.mean().backward()
                    self.optimizers["pred_model"].step()
            # print(f"pred_model training: {time.time() - check_point} sec.")
            avg_loss_str = f"pred_model->{total_pred_model_loss:.6f}, policy->{total_ppo_loss:.6f}"
        # endregion

        for k in loss_log.keys():
            try:
                if len(loss_log[k]) > 0:
                    loss_log[k] = torch.mean(torch.stack(loss_log[k]))
                else:
                    loss_log[k] = None
            except Exception as err:
                print()
                print(err, k)
        return loss_log, avg_loss_str
    # endregion

    # region learner utils
    def _init_optim(self, config: Config):
        optimizers = {}
        if self.learning_mode == "joint":
            optimizers["joint"] = optim.Adam(
                [
                    {"params": self.actor.rnn.parameters()},
                    {"params": self.actor.policy.parameters()},
                    {"params": self.actor.pred_model.parameters()}
                ],
                lr=self.lr_joint
            )
        elif self.learning_mode == "separate":
            optimizers["pred_model"] = optim.Adam(
                [
                    {"params": self.actor.pred_model.parameters()},
                    # {"params": self.actor.rnn.parameters()}
                ],
                lr=self.lr_pred_model
            )
            optimizers["policy"] = optim.Adam(
                [
                    {"params": self.actor.pred_model.enc_net.parameters()},
                    {"params": self.actor.pred_model.enc_mean.parameters()},
                    # {"params": self.actor.pred_model.enc_std.parameters()},
                    {"params": self.actor.pred_model.phi_x.parameters()},
                    {"params": self.actor.pred_model.phi_z.parameters()},
                    {"params": self.actor.rnn.parameters()},
                    {"params": self.actor.policy.parameters()}
                ],
                lr=self.lr_policy
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
                cooldown=2, min_lr=5e-5, verbose=True
            )
        elif self.learning_mode == "separate":
            schedulers["pred_model"] = sched.ReduceLROnPlateau(
                self.optimizers["pred_model"], mode='min', factor=0.5, patience=6,
                threshold=1e-3, threshold_mode='rel',
                cooldown=2, min_lr=5e-5, verbose=True
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
            return {
                "lr_pred_model": self.optimizers["pred_model"].param_groups[0]['lr'],
                "lr_policy": self.optimizers["policy"].param_groups[0]['lr']
            }

    def _lr_adapter_policy(self, kld, clipfrac,
            target_kl=0.01, low=5e-5, high=1e-3, down=0.8, up=1.2):
        for g in self.optimizers["policy"].param_groups:
            lr = g["lr"]
            if kld < 0.5 * target_kl and clipfrac < 0.02:
                lr *= up
            elif kld > 1.5 * target_kl or clipfrac > 0.04:
                lr *= down
            g["lr"] = min(max(lr, low), high)
    # endregion