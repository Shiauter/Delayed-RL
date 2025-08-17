import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import time

from actor_vrnn_v2 import Actor
from util import Memory, clamp
from config import Config

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
    kld_policy_range: list

    # pred_model
    p_iters: int
    z_size: int
    reconst_loss_method: str
    pause_update_ep: int
    kld_range: list

    # training params
    learning_mode: str
    num_memos: int
    epoch_tier_joint: int
    epoch_tier_policy: int
    epoch_tier_pred_model: int
    lr_tier_joint: int
    lr_tier_policy: int
    lr_tier_pred_model: int
    epoch_tier: list
    lr_tier: list
    device: str
    # endregion

    def __init__(self, config: Config):
        for key, value in vars(config).items():
            if key in self.__annotations__:
                setattr(self, key, value)

        self.actor: Actor = Actor(config)
        self.optimizers: dict = self._init_optim(config)

    def _make_batch(self, memory: Memory):
        s, a, prob_a, r, s_prime, done, t, a_lst = \
            map(lambda key: torch.tensor(memory.exps[key]).unsqueeze(0).to(self.device), memory.keys)

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

    def _cal_pred_model_loss(self, s, a_lsts, first_hidden):
        # s       -> (batch=1, ep_len, s_size) # current s
        # a_lsts  -> (batch=1, ep_len, delay)
        # h       -> (seq_len=1, batch=1, hidden_size)

        all_h = self._get_start_h(s, first_hidden) # ep_len - 1

        # predicting one step future state using true state
        a = torch.split(a_lsts[:, :-1], 1, dim=-1)[0] # split into actions
        kld_loss, nll_loss, mse_loss, dec_std = self.actor.pred_model.cal_loss(s[:, 1:], a, all_h)
        kld_loss, nll_loss, mse_loss = kld_loss.sum(dim=0).mean(), nll_loss.sum(dim=0).mean(), mse_loss.sum(dim=0).mean()

        dec_std_mean = dec_std.sum(dim=0).mean()
        dec_std_max = dec_std.sum(dim=0).max()
        dec_std_min = dec_std.sum(dim=0).min()

        return kld_loss, nll_loss, mse_loss, dec_std_mean, dec_std_max, dec_std_min
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
        ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

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

        kl_div = (torch.log(prob_a) - torch.log(pi_a)).mean()
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
            for epoch in range(self.epoch_tier[self.epoch_tier_joint]):
                total_pred_model_loss, total_ppo_loss = [], []
                for i in range(self.num_memos):
                    s, a, r, s_prime, done, prob_a, a_lst = self._make_batch(memory_list[i])
                    first_hidden = memory_list[i].h0.detach().to(self.device)

                    if self.p_iters > 0:
                        kld_loss, nll_loss, mse_loss = self._cal_pred_model_loss(s, a, first_hidden)
                        if self.reconst_loss_method == "NLL":
                            total_pred_model_loss.append(kld_loss + nll_loss)
                        elif self.reconst_loss_method == "MSE":
                            total_pred_model_loss.append(kld_loss + mse_loss)

                        loss_log["kld_loss"].append(kld_loss)
                        loss_log["nll_loss"].append(nll_loss)
                        loss_log["mse_loss"].append(mse_loss)

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
                total_loss = total_ppo_loss + total_pred_model_loss
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
            for epoch in range(self.epoch_tier[self.epoch_tier_policy]):
                total_ppo_loss = []
                timer = time.time()
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
                print(f"memo loop: {time.time() - timer} sec.")
                total_ppo_loss = torch.stack(total_ppo_loss).mean()
                loss_log["ppo_loss"].append(total_ppo_loss)
                # print(f"total_policy_loss: {total_ppo_loss} / ep. {epoch + 1}")

                timer = time.time()
                self.optimizers["policy"].zero_grad()
                total_ppo_loss.mean().backward()
                self.optimizers["policy"].step()
                print(f"backprop: {time.time() - timer} sec.")
            print(f"policy training: {time.time() - check_point} sec.")

            check_point = time.time()
            for epoch in range(self.epoch_tier[self.epoch_tier_pred_model]):
                total_pred_model_loss = []
                for i in range(self.num_memos):
                    s, a, r, s_prime, done, prob_a, a_lst = self._make_batch(memory_list[i]) # (batch=1, seq_len, data_size)
                    first_hidden = memory_list[i].h0.detach().to(self.device) # (seq_len, batch, hidden_size)

                    kld_loss, nll_loss, mse_loss, dec_std_mean, dec_std_max, dec_std_min = self._cal_pred_model_loss(s, a, first_hidden)
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

                total_pred_model_loss = torch.stack(total_pred_model_loss).mean()
                loss_log["pred_model_loss"].append(total_pred_model_loss)
                # print(f"total_pred_model_loss: {total_pred_model_loss} / ep. {epoch + 1}")

                if self.pause_update_ep is None or current_episode <= self.pause_update_ep:
                    self.optimizers["pred_model"].zero_grad()
                    total_pred_model_loss.mean().backward()
                    self.optimizers["pred_model"].step()
            print(f"pred_model training: {time.time() - check_point} sec.")
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
                lr=self.lr_tier[self.lr_tier_joint]
            )
        elif self.learning_mode == "separate":
            optimizers["pred_model"] = optim.Adam(
                [
                    {"params": self.actor.rnn.parameters()},
                    {"params": self.actor.pred_model.parameters()}
                ],
                lr=self.lr_tier[self.lr_tier_pred_model]
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
                lr=self.lr_tier[self.lr_tier_policy]
            )
        else:
            raise ValueError(f"Unknown learning_mode: {config.learning_mode}")

        return optimizers

    def adjust_learning_params(self, loss_log: dict, prev_loss_log: dict, ep: int):
        # if self.p_iters == 0: return -1, -1
        kld, nll, advtg_mean, kld_policy, entropy, avg_clipped_distance = \
            loss_log["kld_loss"], loss_log["nll_loss"], \
            loss_log["advtg_mean"], loss_log["kld_policy"], \
            loss_log["entropy_bonus"], loss_log["avg_clipped_distance"]

        # prev_kld, prev_nll, prev_advtg_mean, prev_kld_policy, prev_entropy = \
        #     prev_loss_log["kld_loss"], prev_loss_log["nll_loss"], \
        #     prev_loss_log["advtg_mean"], prev_loss_log["kld_policy"], prev_loss_log["entropy_bonus"]

        self._cal_pred_model_param_tier(kld)
        self._cal_policy_param_tier(kld_policy, avg_clipped_distance)

        if self.learning_mode == "separate":
            for param_group in self.optimizers["pred_model"].param_groups:
                param_group['lr'] = self.lr_tier[self.lr_tier_pred_model]
            for param_group in self.optimizers["policy"].param_groups:
                param_group['lr'] = self.lr_tier[self.lr_tier_policy]
        # print(f"Param Tier -> pred_model: {pred_model_tier} / policy: {policy_tier}")

        if ep == self.set_gate_reg_weight_at_ep:
            self.gate_reg_weight = self.gate_reg_weight_to_set

        return self.epoch_tier_pred_model, self.lr_tier_pred_model,\
               self.epoch_tier_policy, self.lr_tier_policy

    def _cal_pred_model_param_tier(self, kld):
        if kld < self.kld_range[0]: tier_adj = -1
        elif kld > self.kld_range[1]: tier_adj = 1
        else: tier_adj = 0

        # self.epoch_tier_pred_model = clamp(
        #     self.epoch_tier_pred_model + tier_adj, 0, len(self.epoch_tier) - 1)
        self.lr_tier_pred_model = clamp(
            self.lr_tier_pred_model + tier_adj, 0, len(self.lr_tier) - 1)

    def _cal_policy_param_tier(self, kld, avg_clipped_distance):
        if kld < self.kld_policy_range[0] and avg_clipped_distance < 0.02: tier_adj = -1
        elif kld > self.kld_policy_range[1] or avg_clipped_distance > 0.04: tier_adj = 1
        else: tier_adj = 0

        # self.epoch_tier_policy = clamp(
        #     self.epoch_tier_policy + tier_adj, 0, len(self.epoch_tier) - 1)
        self.lr_tier_policy = clamp(
            self.lr_tier_policy + tier_adj, 0, 2)

    # endregion