import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import time

from actor_vrnn import Actor
from util import Memory
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

    # pred_model
    p_iters: int
    z_size: int
    reconst_loss_method: str
    pause_update_ep: int

    # training params
    learning_mode: str
    num_memos: int
    K_epoch_policy: int
    K_epoch_pred_model: int
    K_epoch_learn: int
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
    def _make_offset_seq(self, target, offset: tuple, limit: int):
        # 給定一組episode中真實的states，以每個state的index為基準，找出相對它offset[0] ~ offset[1] - 1的位置的states所組成的序列
        # limit是最後一個被選為基準的state的index
        # e.g. 假設目前以s[1]為基準，且offset = (1, 3)，對應的序列為s[1] ~ s[2]，limit = 2則代表最後為基準準的state為s[2]

        # target -> (ep_len, s_size)

        idx = torch.arange(limit).unsqueeze(1) + torch.arange(offset[0], offset[1]).unsqueeze(0)
        idx = idx.clamp(max=len(target) - 1)
        res = target[idx].view(offset[1] - offset[0], -1, self.s_size)

        return res # (offset_len, target_seq_len, target_size)

    def _get_start_h_post(self, s, a_lsts, h, s_truth):
        # posterior
        # s       -> (batch=1, ep_len - delay, s_size)
        # a_lsts  -> (batch=1, ep_len - delay, delay)
        # h       -> (seq_len=1, batch=1, hidden_size)
        # s_truth -> (delay, ep_len - delay, s_size)

        # remove batch dimension
        s, a_lsts = s.squeeze(0), a_lsts.squeeze(0)
        h_post = [h]
        for x_post, x_cond, a_lst in zip(s_truth[0], s, a_lsts):
            x_post, x_cond = x_post.view(1, 1, -1), x_cond.view(1, 1, -1)
            a_first = torch.split(a_lst, 1, dim=-1)[0].view(1, 1, -1)

            # for training pred_model
            _, _, phi_x_post, phi_z_post, _, _ = self.actor.pred_model.reconstruct(x_post, x_cond, a_first, h_post[-1])
            rnn_in_post = torch.cat([phi_x_post, phi_z_post], dim=-1)
            _, h_t_post = self.actor.rnn(rnn_in_post, h_post[-1])
            h_post.append(h_t_post)

        # (seq_len=1, batch, hidden_size)
        h_post = torch.cat(h_post, dim=1)[:, :-1, :]
        return h_post

    def _make_pred_s_tis(self, s, a_lsts, h, s_truth):
        # s       -> (batch=1, ep_len - delay, s_size) # current s
        # a_lsts  -> (batch=1, ep_len - delay, delay)
        # h       -> (seq_len=1, batch=1, hidden_size)
        # s_truth -> (delay, ep_len - delay, s_size) # for delay prediction

        all_h_in = self._get_start_h_post(s, a_lsts, h, s_truth)
        kld_loss, nll_loss = [], []
        mse_loss = []

        # predicting future state using all states and corresponding starting hidden
        s_truth = s_truth.unsqueeze(1) # add batch dimension
        pred_s = s
        a_lst = torch.split(a_lsts, 1, dim=-1) # split into actions
        for i in range(self.p_iters):
            kld, nll, phi_x_truth, phi_z_truth, mse, pred_s = self.actor.pred_model.reconstruct(s_truth[i], pred_s, a_lst[i], all_h_in)
            rnn_in_truth = torch.cat([phi_x_truth, phi_z_truth], dim=-1)
            _, all_h_in = self.actor.rnn(rnn_in_truth, all_h_in)
            kld_loss.append(kld)
            nll_loss.append(nll)
            mse_loss.append(mse)

        # sum(dim=0) for delay prediction
        kld_loss = torch.cat(kld_loss, dim=0).sum(dim=0).mean()
        nll_loss = torch.cat(nll_loss, dim=0).sum(dim=0).mean()
        mse_loss = torch.stack(mse_loss, dim=0).sum(dim=0).mean()
        return kld_loss, nll_loss, mse_loss

    def _cal_pred_model_loss(self, s, a_lsts, first_hidden):
        # s            -> (batch=1, ep_len, s_size)
        # a_lsts       -> (batch=1, ep_len, delay)
        # first_hidden -> (seq_len=1, batch=1, hidden_size)

        if self.p_iters > 0:
            limit = s.shape[1] - self.delay
            offset = (1, self.delay + 1)
            s_truth = self._make_offset_seq(s.squeeze(0), offset, limit)
            kld_loss, nll_loss, mse_loss = self._make_pred_s_tis(s[:, :limit], a_lsts[:, :limit], first_hidden, s_truth)
        return kld_loss, nll_loss, mse_loss
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

        # remove batch dimension
        s, a_lsts = s.squeeze(0), a_lsts.squeeze(0)
        h_in = first_hidden.squeeze(0)
        res = {"pi": [], "h_out": [], "v": [], "gated_val": []}
        for i, (state, a_lst) in enumerate(zip(s, a_lsts)):
            if i >= s.shape[0] - self.delay: break
            state, a_lst = state.unsqueeze(0), a_lst.unsqueeze(0)
            _, pi, h_out, _, v, gated_val = self.actor.sample_action(state, a_lst, h_in)
            h_in = h_out

            res["pi"].append(pi)
            res["h_out"].append(h_out)
            res["v"].append(v)
            res["gated_val"].append(gated_val.view(-1))

        second_hidden = res["h_out"][0].unsqueeze(0)
        for k in res.keys():
            res[k] = torch.stack(res[k]).squeeze(1)
        return res["pi"], res["v"], second_hidden, res["gated_val"]

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

        # The length of two a_lst doesn't have to be the same here
        # because delay > 0 and it ignored the tail of a_lst
        pi, v_s, second_hidden, gated_val = self._make_pi_and_critic(s, a_lst, first_hidden)
        _ , v_prime, _, _ = self._make_pi_and_critic(s_prime, a_lst[:, 1:], second_hidden)

        advantage, return_target = self._cal_advantage(v_s, r, v_prime, done)

        pi_a, prob_a = pi.gather(0, a[self.delay:]), prob_a[:-self.delay]
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

        gate_reg = self.gate_reg_weight * ((gated_val - 0.5) ** 2).mean()

        kl_div = (pi * (torch.log(pi) - torch.log(prob_a))).sum(dim=-1).mean()
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
            v_s_mean, td_target_mean, gate_reg
    # endregion

    # region training
    def learn(self, memory_list: list[Memory], current_episode: int):
        # pred_model_loss, ppo_loss, total_loss are added in training function
        keys = [
            # pred model
            "kld_loss", "nll_loss",
            "mse_loss",

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
            for epoch in range(self.K_epoch_learn):
                total_pred_model_loss, total_ppo_loss = [], []
                for i in range(self.num_memos):
                    s, a, r, s_prime, done, prob_a, a_lst = self._make_batch(memory_list[i])
                    first_hidden = memory_list[i].h0.detach().to(self.device)

                    kld_loss, nll_loss, mse_loss = self._cal_pred_model_loss(s, a_lst, first_hidden)
                    if self.reconst_loss_method == "NLL":
                        total_pred_model_loss.append(kld_loss + nll_loss)
                    elif self.reconst_loss_method == "MSE":
                        total_pred_model_loss.append(kld_loss + mse_loss)

                    loss_log["kld_loss"].append(kld_loss)
                    loss_log["nll_loss"].append(nll_loss)
                    loss_log["mse_loss"].append(mse_loss)

                    policy_loss, critic_loss, entropy_bonus, kld_policy, advtg_mean, \
                    clipped_percentage, avg_clipped_distance, advtg_std, advtg_min, advtg_max, \
                    v_s_std, td_target_std, v_s_mean, td_target_mean, gate_reg = \
                        self._cal_ppo_loss(s, s_prime, a, prob_a, r, done, a_lst, first_hidden)
                    ppo_loss = policy_loss + critic_loss + entropy_bonus + gate_reg
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
                total_loss = total_pred_model_loss + total_pred_model_loss
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

            for epoch in range(self.K_epoch_pred_model):
                total_pred_model_loss = []
                for i in range(self.num_memos):
                    s, a, r, s_prime, done, prob_a, a_lst = self._make_batch(memory_list[i]) # (batch=1, seq_len, data_size)
                    first_hidden = memory_list[i].h0.detach().to(self.device) # (seq_len, batch, hidden_size)

                    kld_loss, nll_loss, mse_loss = self._cal_pred_model_loss(s, a_lst, first_hidden)
                    if self.reconst_loss_method == "NLL":
                        total_pred_model_loss.append(kld_loss + nll_loss)
                    elif self.reconst_loss_method == "MSE":
                        total_pred_model_loss.append(kld_loss + mse_loss)

                    loss_log["kld_loss"].append(kld_loss)
                    loss_log["nll_loss"].append(nll_loss)
                    loss_log["mse_loss"].append(mse_loss)

                total_pred_model_loss = torch.stack(total_pred_model_loss).mean()
                loss_log["pred_model_loss"].append(total_pred_model_loss)
                # print(f"total_pred_model_loss: {total_pred_model_loss} / ep. {epoch + 1}")

                if self.pause_update_ep is None or current_episode <= self.pause_update_ep:
                    self.optimizers["pred_model"].zero_grad()
                    total_pred_model_loss.mean().backward()
                    self.optimizers["pred_model"].step()


            for epoch in range(self.K_epoch_policy):
                total_ppo_loss = []
                for i in range(self.num_memos):
                    s, a, r, s_prime, done, prob_a, a_lst = self._make_batch(memory_list[i])
                    first_hidden = memory_list[i].h0.detach().to(self.device)

                    policy_loss, critic_loss, entropy_bonus, kld_policy, advtg_mean, \
                    clipped_percentage, avg_clipped_distance, advtg_std, advtg_min, advtg_max, \
                    v_s_std, td_target_std, v_s_mean, td_target_mean, gate_reg = \
                        self._cal_ppo_loss(s, s_prime, a, prob_a, r, done, a_lst, first_hidden)
                    ppo_loss = policy_loss + critic_loss + entropy_bonus + gate_reg
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

            avg_loss_str = f"pred_model->{total_pred_model_loss:.6f}, policy->{total_ppo_loss:.6f}"
        # endregion

        for k in loss_log.keys():
            try:
                loss_log[k] = torch.mean(torch.stack(loss_log[k]))
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
                lr=config.lr
            )
        elif self.learning_mode == "separate":
            optimizers["pred_model"] = optim.Adam(
                [
                    {"params": self.actor.rnn.parameters()},
                    {"params": self.actor.pred_model.parameters()}
                ],
                lr=config.lr_pred_model
            )
            optimizers["policy"] = optim.Adam(
                [
                    {"params": self.actor.policy.parameters()}
                ],
                lr=config.lr_policy
            )
        else:
            raise ValueError(f"Unknown learning_mode: {config.learning_mode}")

        return optimizers

    def adjust_learning_params(self, loss_log: dict, prev_loss_log: dict, ep: int):
        kld, nll, advtg_mean, kld_policy, entropy = \
            loss_log["kld_loss"], loss_log["nll_loss"], \
            loss_log["advtg_mean"], loss_log["kld_policy"], loss_log["entropy_bonus"]

        # prev_kld, prev_nll, prev_advtg_mean, prev_kld_policy, prev_entropy = \
        #     prev_loss_log["kld_loss"], prev_loss_log["nll_loss"], \
        #     prev_loss_log["advtg_mean"], prev_loss_log["kld_policy"], prev_loss_log["entropy_bonus"]

        pred_model_tier = self._cal_pred_model_param_tier(kld, nll, ep)
        policy_tier = self._cal_policy_param_tier(pred_model_tier, kld_policy, entropy, advtg_mean, ep)
        self.K_epoch_pred_model = self.epoch_tier[pred_model_tier]
        self.K_epoch_policy = self.epoch_tier[policy_tier]
        if self.learning_mode == "separate":
            for param_group in self.optimizers["pred_model"].param_groups:
                param_group['lr'] = self.lr_tier[pred_model_tier]
            for param_group in self.optimizers["policy"].param_groups:
                param_group['lr'] = self.lr_tier[pred_model_tier]
        # print(f"Param Tier -> pred_model: {pred_model_tier} / policy: {policy_tier}")

        if ep == self.set_gate_reg_weight_at_ep:
            self.gate_reg_weight = self.gate_reg_weight_to_set

        return pred_model_tier, policy_tier

    def _cal_pred_model_param_tier(self, kld, recon, ep):
        epoch_tier = 4
        if kld > 0.5:
            epoch_tier = 0
        if kld > 0.3:
            epoch_tier = 1
        if kld > 0.1:
            epoch_tier = 2
        if kld > 0.05:
            epoch_tier = 3
        return epoch_tier

    def _cal_policy_param_tier(self, pred_model_tier, kld, entropy, adv, ep):
        epoch_tier = 4
        if pred_model_tier <= 2:
            epoch_tier = 0
        return epoch_tier
    # endregion