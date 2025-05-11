
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import time

from actor_vrnn import Actor
from util import Memory
from config import Config

class Learner:
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
    eps_clip: float

    # pred_model
    p_iters: int
    z_size: int
    reconst_loss_method: str
    pause_update_ep: int

    # training params
    num_memos: int
    num_actors: int
    K_epoch_policy: int
    K_epoch_pred_model: int
    K_epoch_learn: int
    epoch_tier: list
    lr_tier: list
    device: str

    def __init__(self, config: Config):
        for key, value in vars(config).items():
            if key in self.__annotations__:
                setattr(self, key, value)

        self.actor = Actor(config)
        self.actor.set_device(config.device)

        self.optim_pred_model = optim.Adam(
            [
                {"params": self.actor.rnn.parameters()},
                {"params": self.actor.pred_model.parameters()}
            ],
            lr=config.lr_pred_model
        )
        self.optim_policy = optim.Adam(
            [
                {"params": self.actor.policy.parameters()}
            ],
            lr=config.lr_policy
        )
        self.optim = optim.Adam(
            [
                {"params": self.actor.rnn.parameters()},
                {"params": self.actor.policy.parameters()},
                {"params": self.actor.pred_model.parameters()}
            ],
            lr=config.lr
        )

    def make_batch(self, memory: Memory):
        s, a, prob_a, r, s_prime, done, t, a_lst = \
            map(lambda key: torch.tensor(memory.exps[key]).to(self.device), memory.keys)
        return s, a, r, s_prime, done, prob_a, a_lst

    def make_offset_seq(self, s, offset: tuple, limit):
        # 給定一組episode中真實的states，以每個state的index為基準，找出相對它offset[0] ~ offset[1] - 1的位置的states所組成的序列
        # limit是最後一個被選為基準的state的index
        # e.g. 假設目前以s[1]為基準，且offset = (1, 3)，對應的序列為s[1] ~ s[2]，limit = 2則代表最後為基準準的state為s[2]
        idx = torch.arange(limit).unsqueeze(1) + torch.arange(offset[0], offset[1]).unsqueeze(0)
        idx = idx.clamp(max=len(s) - 1)
        # print(idx)
        res = s[idx].view(offset[1] - offset[0], -1, self.s_size)
        return res

    def make_pred_s_tis(self, s_truth, s, a, h):
        total_loss = []

        # get all starting hidden
        start_time = time.time()
        h_truth, h_cond = [h], [h]
        for x_truth, x_cond, a_lst in zip(s_truth[0], s, a):
            x_truth, x_cond = x_truth.view(1, 1, -1), x_cond.view(1, 1, -1)
            a_first = torch.split(a_lst, 1, dim=-1)[0].view(1, 1, -1)

            _, _, phi_x_truth, phi_z_truth, _, _ = self.actor.pred_model.reconstruct(x_truth, x_cond, a_first, h_truth[-1])
            rnn_in_truth = torch.cat([phi_x_truth, phi_z_truth], dim=-1)
            _, h_t_truth = self.actor.rnn(rnn_in_truth, h_truth[-1])
            h_truth.append(h_t_truth)

            _, phi_x_cond, phi_z_cond = self.actor.pred_model(x_cond, a_first, h_cond[-1])
            rnn_in_cond = torch.cat([phi_x_cond, phi_z_cond], dim=-1)
            _, h_t_cond = self.actor.rnn(rnn_in_cond, h_cond[-1])
            h_cond.append(h_t_cond)
            # print(f"--- {time.time() - start_time} seconds for starting hidden ---")
        h_truth = torch.cat(h_truth, dim=1)[:, :-1, :]
        h_cond = torch.cat(h_cond, dim=1)[:, :-1, :]

        # 用所有states和對應的starting hidden產生對未來狀態的預測
        start_time = time.time()
        s_truth, pred_s, a = s_truth.unsqueeze(1), s.unsqueeze(0), a.unsqueeze(0) # makes them have same size as hidden
        kld_loss, nll_loss = [], []
        mse_loss = []
        a_lst = torch.split(a, 1, dim=-1)
        for i in range(self.p_iters):
            kld, nll, phi_x_truth, phi_z_truth, mse, pred_s = self.actor.pred_model.reconstruct(s_truth[i], pred_s, a_lst[i], h_truth)
            rnn_in_truth = torch.cat([phi_x_truth, phi_z_truth], dim=-1)
            _, h_truth = self.actor.rnn(rnn_in_truth, h_truth)
            kld_loss.append(kld)
            nll_loss.append(nll)
            mse_loss.append(mse)

            mu_out, phi_x_cond, phi_z_cond = self.actor.pred_model(pred_s, a_lst[i], h_cond)
            rnn_in_cond = torch.cat([phi_x_cond, phi_z_cond], dim=-1)
            o_cond, h_cond  = self.actor.rnn(rnn_in_cond, h_cond) # for policy traing
        # print(f"--- {time.time() - start_time} seconds for following sequences ---")
        kld_loss = torch.cat(kld_loss, dim=0).sum(dim=0).mean()
        nll_loss = torch.cat(nll_loss, dim=0).sum(dim=0).mean()
        mse_loss = torch.stack(mse_loss, dim=0).sum(dim=0).mean()

        # o_cond = torch.cat([o_cond, phi_z_cond], dim=-1)
        empty_data = torch.zeros(1, 1, self.hidden_size).to(self.device) # for v_prime, placeholder for s_prime after terminal state
        o_cond = torch.cat([o_cond, empty_data], dim=1)
        # mu_out = torch.cat([mu_out, empty_data], dim=1)
        return kld_loss, nll_loss, o_cond, mse_loss

    def cal_advantage(self, v_s, r, v_prime, done_mask):
        td_target = r + self.gamma * v_prime * done_mask
        delta = td_target - v_s

        advtg_lst = []
        advtg_t = 0.0
        for delta_t in reversed(delta):
            advtg_t = self.gamma * self.lmbda * advtg_t + delta_t.item()
            advtg_lst.append([advtg_t])
        advtg_lst.reverse()
        advantage = torch.tensor(advtg_lst, dtype=torch.float).to(self.device)
        return_target = advantage + v_s
        return advantage, return_target

    def make_pi_and_critic(self, o):
        second_hidden = o[0].unsqueeze(0)
        pi = self.actor.policy.pi(o)
        v = self.actor.policy.v(o)
        pi, v = pi.squeeze(0), v.squeeze(0)
        return pi, v, second_hidden.detach()

    def cal_pred_model_loss(self, s, a_lst, first_hidden):
        if self.p_iters > 0:
            limit = len(s) - self.actor.delay
            s_truth = self.make_offset_seq(s, (1, self.actor.delay + 1), limit)
            kld_loss, nll_loss, o_ti, mse_loss = self.make_pred_s_tis(s_truth, s[:limit], a_lst[:limit], first_hidden)
            # print(f"nll: {nll_loss}, kld: {kld_loss}")
        return kld_loss, nll_loss, o_ti, mse_loss

    def cal_ppo_loss(self, o_ti, a, prob_a, r, done):
        pi, v_s, _ = self.make_pi_and_critic(o_ti[:, :-1, :])
        _ , v_prime, _ = self.make_pi_and_critic(o_ti[:, 1:, :])
        advantage, return_target = self.cal_advantage(v_s, r[self.delay:], v_prime, done[self.delay:])

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

        kl_div = (pi * (torch.log(pi) - torch.log(prob_a))).sum(dim=-1).mean()
        advtg_mean = advantage.mean(dim=-1)

        return policy_loss, critic_loss, entropy_bonus, kl_div, advtg_mean, cliped_percentage, avg_clipped_distance

    def learn(self, memory_list: list[Memory]):
        keys = ["total_loss", "pred_model_loss", "ppo_loss",
                "policy_loss", "critic_loss", "entropy_bonus",
                "kld_loss", "nll_loss"]
        loss_log = {}
        for k in keys:
            loss_log[k] = []

        for epoch in range(self.K_epoch_learn):
            total_ppo_loss, total_pred_model_loss = 0, 0
            for i in range(self.num_memos):
                s, a, r, s_prime, done, prob_a, a_lst = self.make_batch(memory_list[i])
                first_hidden = memory_list[i].h0.detach()

                kld_loss, nll_loss, o_ti = self.cal_pred_model_loss(s, a_lst, first_hidden)
                total_pred_model_loss += kld_loss + nll_loss

                loss_log["kld_loss"].append(kld_loss.mean())
                loss_log["nll_loss"].append(nll_loss.mean())

                policy_loss, critic_loss, entropy_bonus, _, _ = self.cal_ppo_loss(o_ti, a, prob_a, r, done)
                ppo_loss = policy_loss + critic_loss + entropy_bonus

                total_ppo_loss += ppo_loss

                loss_log["policy_loss"].append(policy_loss.mean())
                loss_log["critic_loss"].append(critic_loss.mean())
                loss_log["entropy_bonus"].append(entropy_bonus.mean())

            total_ppo_loss /= self.num_memos
            total_pred_model_loss /= self.num_memos
            total_loss = total_ppo_loss + total_pred_model_loss

            self.optim.zero_grad()
            total_loss.mean().backward()
            self.optim.step()

            loss_log["total_loss"].append(total_loss)
            loss_log["ppo_loss"].append(total_ppo_loss)
            loss_log["pred_model_loss"].append(total_pred_model_loss)

        for k in loss_log.keys():
            try:
                loss_log[k] = torch.mean(torch.stack(loss_log[k]))
            except Exception as err:
                print(err, k)
        return loss_log, f'{loss_log["total_loss"]:.9f}'

    def separated_learning(self, memory_list: list[Memory], current_episode: int):
        # output的指標需要先加在keys中
        keys = [
            "pred_model_loss", "kld_loss", "nll_loss",
            "mse_loss",

            "ppo_loss", "policy_loss", "critic_loss",
            "entropy_bonus", "kld_policy", "advtg_mean",
            "clipped_percentage", "avg_clipped_distance"
        ]

        loss_log = {}
        for k in keys:
            loss_log[k] = []

        # pred_model
        start_time = time.time()
        for epoch in range(self.K_epoch_pred_model):
            total_pred_model_loss = []
            for i in range(self.num_memos):
                start_loss_time = time.time()
                s, a, r, s_prime, done, prob_a, a_lst = self.make_batch(memory_list[i])
                first_hidden = memory_list[i].h0.detach().to(self.device)

                kld_loss, nll_loss, o_ti, mse_loss = self.cal_pred_model_loss(s, a_lst, first_hidden)
                if self.reconst_loss_method == "NLL":
                    total_pred_model_loss.append(kld_loss + nll_loss)
                elif self.reconst_loss_method == "MSE":
                    total_pred_model_loss.append(kld_loss + mse_loss)

                loss_log["kld_loss"].append(kld_loss)
                loss_log["nll_loss"].append(nll_loss)
                loss_log["mse_loss"].append(mse_loss)
                # print(f"--- {time.time() - start_loss_time} seconds for pred_model loss ---")

            total_pred_model_loss = torch.stack(total_pred_model_loss).mean()
            loss_log["pred_model_loss"].append(total_pred_model_loss)
            # print(f"total_pred_model_loss: {total_pred_model_loss} / ep. {epoch + 1}")

            if self.pause_update_ep is None or current_episode <= self.pause_update_ep:
                self.optim_pred_model.zero_grad()
                total_pred_model_loss.mean().backward()
                self.optim_pred_model.step()
        # print(f"--- {time.time() - start_time} seconds for pred_model training ---")

        # policy
        start_time = time.time()
        for epoch in range(self.K_epoch_policy):
            total_ppo_loss = 0
            for i in range(self.num_memos):
                start_loss_time = time.time()
                s, a, r, s_prime, done, prob_a, a_lst = self.make_batch(memory_list[i])
                first_hidden = memory_list[i].h0.detach().to(self.device)

                _, _, o_ti, _ = self.cal_pred_model_loss(s, a_lst, first_hidden)

                # print(f"--- {time.time() - start_loss_time} seconds for pred_model loss in policy training ---")
                start_loss_time = time.time()
                policy_loss, critic_loss, entropy_bonus, kld_policy, advtg_mean, clipped_percentage, avg_clipped_distance = self.cal_ppo_loss(o_ti, a, prob_a, r, done)
                # ppo_loss = policy_loss + critic_loss + entropy_bonus
                ppo_loss = policy_loss
                total_ppo_loss += ppo_loss

                loss_log["policy_loss"].append(policy_loss.mean())
                loss_log["critic_loss"].append(critic_loss.mean())
                loss_log["entropy_bonus"].append(entropy_bonus.mean())
                loss_log["kld_policy"].append(kld_policy.mean())
                loss_log["advtg_mean"].append(advtg_mean.mean())
                loss_log["clipped_percentage"].append(clipped_percentage.mean())
                loss_log["avg_clipped_distance"].append(avg_clipped_distance.mean())
                # print(f"--- {time.time() - start_loss_time} seconds for policy loss ---")

            total_ppo_loss /= self.num_memos
            loss_log["ppo_loss"].append(total_ppo_loss)
            # print(f"total_policy_loss: {total_ppo_loss} / ep. {epoch + 1}")

            self.optim_policy.zero_grad()
            total_ppo_loss.mean().backward()
            self.optim_policy.step()
        # print(f"--- {time.time() - start_time} seconds for policy training ---")

        for k in loss_log.keys():
            try:
                loss_log[k] = torch.mean(torch.stack(loss_log[k]))
            except Exception as err:
                print()
                print(err, k)
        return loss_log, f"pred_model->{total_pred_model_loss:.6f}, policy->{total_ppo_loss:.6f}"

    def adjust_learning_params(self, loss_log: dict, prev_loss_log: dict):
        kld, nll, advtg_mean, kld_policy, entropy = \
            loss_log["kld_loss"], loss_log["nll_loss"], \
            loss_log["advtg_mean"], loss_log["kld_policy"], loss_log["entropy_bonus"]

        # prev_kld, prev_nll, prev_advtg_mean, prev_kld_policy, prev_entropy = \
        #     prev_loss_log["kld_loss"], prev_loss_log["nll_loss"], \
        #     prev_loss_log["advtg_mean"], prev_loss_log["kld_policy"], prev_loss_log["entropy_bonus"]

        pred_model_tier = self._cal_pred_model_param_tier(kld, nll)
        policy_tier = self._cal_policy_param_tier(pred_model_tier, kld_policy, entropy, advtg_mean)
        self.K_epoch_pred_model, self.K_epoch_policy = self.epoch_tier[pred_model_tier], self.epoch_tier[policy_tier]
        for param_group in self.optim_pred_model.param_groups:
            param_group['lr'] = self.lr_tier[pred_model_tier]
        for param_group in self.optim_policy.param_groups:
            param_group['lr'] = self.lr_tier[pred_model_tier]
        # print(f"Param Tier -> pred_model: {pred_model_tier} / policy: {policy_tier}")
        return pred_model_tier, policy_tier

    def _cal_pred_model_param_tier(self, kld, recon):
        if kld > 0.1:
            return 4
        if kld > 0.05:
            return 3
        if kld > 0.02:
            return 2
        if kld > 0.01:
            return 1
        return 0

    def _cal_policy_param_tier(self, pred_model_tier, kld, entropy, adv):
        if pred_model_tier >= 2: return 0
        else: return 2
