
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import time

from actor_vrnn import Actor
from util import Memory
from config import Config

class Learner:
    s_size: int
    a_size: int
    gamma: float
    lmbda: float
    critic_weight: float
    entropy_weight: float
    eps_clip: float
    K_epoch_policy: int
    K_epoch_pred_model: int
    K_epoch_learn: int
    delay: int
    p_iters: int
    num_memos: int
    num_actors: int
    T_horizon: int
    hidden_size: int
    z_size: int
    h0: list
    epoch_tier: list
    lr_tier: list
    device: str

    def __init__(self, config: Config, optim_pred_model=None, optim_policy=None, optimizer=None):
        for key, value in vars(config).items():
            if key in self.__annotations__:
                setattr(self, key, value)

        self.actor = Actor(config)
        self.actor.set_device(config.device)

        self.optim_pred_model = optim_pred_model
        self.optim_policy = optim_policy
        self.optim = optimizer

    def make_batch(self, memory: Memory):
        s, a, prob_a, r, s_prime, done, t, a_lst = \
            map(lambda key: torch.tensor(memory.exps[key]).to(self.device), memory.keys)
        return s, a, r, s_prime, done, prob_a, a_lst

    def make_offset_seq(self, s, offset: tuple, limit):
        idx = torch.arange(limit).unsqueeze(1) + torch.arange(offset[0], offset[1]).unsqueeze(0)
        idx = idx.clamp(max=len(s) - 1)
        # print(idx)
        res = s[idx].view(offset[1] - offset[0], -1, self.s_size)
        return res

    def make_pred_s_tis(self, s_truth, s, a, h):
        total_loss = []

        with torch.no_grad():
            start_time = time.time()
            # get all starting hidden
            h_truth, h_cond = [h], [h]
            for x_truth, x_cond, a_lst in zip(s_truth, s, a):
                x_truth, x_cond = x_truth.view(1, 1, -1), x_cond.view(1, 1, -1)
                a_first = torch.split(a_lst, 1, dim=-1)[0].view(1, 1, -1)

                _, _, phi_x_truth, phi_z_truth, _ = self.actor.pred_model.reconstruct(x_truth, x_cond, a_first, h_truth[-1])
                rnn_in_truth = torch.cat([phi_x_truth, phi_z_truth], dim=-1)
                _, h_t_truth = self.actor.rnn(rnn_in_truth, h_truth[-1])
                h_truth.append(h_t_truth)

                _, phi_x_cond, phi_z_cond = self.actor.pred_model(x_cond, a_first, h_cond[-1])
                rnn_in_cond = torch.cat([phi_x_cond, phi_z_cond], dim=-1)
                _, h_t_cond = self.actor.rnn(rnn_in_cond, h_cond[-1])
                h_cond.append(h_t_cond)
            print(f"--- {time.time() - start_time} seconds for starting hidden ---")
        h_truth = torch.cat(h_truth, dim=1)[:, :-1, :]
        h_cond = torch.cat(h_cond, dim=1)[:, :-1, :]

        start_time = time.time()
        s_truth, s, a = s_truth.unsqueeze(0), s.unsqueeze(0), a.unsqueeze(0)
        kld_loss, nll_loss = 0, 0
        mse_loss = 0
        a_lst = torch.split(a, 1, dim=-1)
        for i in range(self.p_iters):
            kld, nll, phi_x_truth, phi_z_truth, mse = self.actor.pred_model.reconstruct(s_truth, s, a_lst[i], h_truth)
            rnn_in_truth = torch.cat([phi_x_truth, phi_z_truth], dim=-1)
            _, h_truth = self.actor.rnn(rnn_in_truth, h_truth)
            kld_loss += kld
            nll_loss += nll
            mse_loss += mse

            mu_out, phi_x_cond, phi_z_cond = self.actor.pred_model(s, a_lst[i], h_cond)
            rnn_in_cond = torch.cat([phi_x_cond, phi_z_cond], dim=-1)
            o_cond, h_cond  = self.actor.rnn(rnn_in_cond, h_cond)
        print(f"--- {time.time() - start_time} seconds for following sequences ---")
        kld_loss = torch.mean(kld_loss, dim=1)
        nll_loss = torch.mean(nll_loss, dim=1)

        # o_cond = torch.cat([o_cond, phi_z_cond], dim=-1)
        empty_data = torch.zeros(1, 1, self.hidden_size).to(self.device) # for v_prime
        o_cond = torch.cat([o_cond, empty_data], dim=1)
        # mu_out = torch.cat([mu_out, empty_data], dim=1)
        return kld_loss, nll_loss, o_cond, mse_loss

    def make_pred_s_tis_old(self, s_truth, s, a, h):
        # total_kld_loss, total_nll_loss = [], []
        # # print(s_truth.shape, s.shape, h.shape, a.shape)
        # h_truth, h_cond = h, h
        # o_ti = []
        # for x_truth, x_cond, a_lst in zip(s_truth, s, a):
        #     # print(x_truth.shape, x_cond.shape, a_lst.shape)
        #     iter_kld_loss, iter_nll_loss = [], []
        #     a_lst = torch.split(a_lst, 1, dim=-1)
        #     x_truth, x_cond = x_truth.view(1, 1, -1), x_cond.view(1, 1, -1)
        #     for i in range(self.p_iters):
        #         # print(x_truth.shape, x_cond.shape, h_truth.shape, h_cond.shape, a_lst[i].shape)
        #         kld_loss, nll_loss, phi_x_truth, phi_z_truth = self.actor.pred_model.reconstruct(x_truth, x_cond, a_lst[i].view(1, 1, -1), h_truth)
        #         rnn_in_truth = torch.cat([phi_x_truth, phi_z_truth], dim=-1).view(1, 1, -1)
        #         o_truth, h_truth = self.actor.rnn(rnn_in_truth, h_truth)
        #         # h要維持第一個

        #         # pred_x, phi_x_cond, phi_z_cond = self.actor.pred_model()
        #         # rnn_in_cond = torch.cat([phi_x_cond, phi_z_cond], dim=-1).view(1, 1, -1)
        #         # o_cond, h_cond  = self.actor.rnn(rnn_in_cond, h_cond)

        #         iter_kld_loss.append(kld_loss)
        #         iter_nll_loss.append(nll_loss)

        #     total_kld_loss.append(torch.stack(iter_kld_loss))
        #     total_nll_loss.append(torch.stack(iter_nll_loss))
        #     o_ti.append(o_truth)
        # total_kld_loss = torch.stack(total_kld_loss).mean(dim=1).mean(dim=0)
        # total_nll_loss = torch.stack(total_nll_loss).mean(dim=1).mean(dim=0)

        # o_ti.append(torch.zeros_like(o_truth)) # for v_prime
        # o_ti = torch.cat(o_ti, dim=1)[-1].unsqueeze(0)
        # return total_kld_loss, total_nll_loss, o_ti
        pass

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
            s_truth = self.make_offset_seq(s, (1, 2), limit).squeeze(0)
            kld_loss, nll_loss, o_ti, mse_loss = self.make_pred_s_tis(s_truth, s[:limit], a_lst[:limit], first_hidden)
            # kld_loss, nll_loss, o_ti = self.make_pred_s_tis_old(s_truth, s[:limit], a_lst[:limit], first_hidden)
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

        critic_loss = self.critic_weight * F.smooth_l1_loss(v_s, return_target.detach())

        entropy = Categorical(pi).entropy().mean()
        entropy_bonus = -self.entropy_weight * entropy

        kl_div = (pi * (torch.log(pi) - torch.log(prob_a))).sum(dim=-1).mean()
        advtg_mean = advantage.mean(dim=-1)

        return policy_loss, critic_loss, entropy_bonus, kl_div, advtg_mean

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

    def separated_learning(self, memory_list: list[Memory]):
        keys = [
            "pred_model_loss", "kld_loss", "nll_loss",
            "mse_loss",

            "ppo_loss", "policy_loss", "critic_loss",
            "entropy_bonus", "kld_policy", "advtg_mean"
        ]

        loss_log = {}
        for k in keys:
            loss_log[k] = []

        start_time = time.time()
        for epoch in range(self.K_epoch_pred_model):
            total_pred_model_loss = 0
            for i in range(self.num_memos):
                start_loss_time = time.time()
                s, a, r, s_prime, done, prob_a, a_lst = self.make_batch(memory_list[i])
                first_hidden = memory_list[i].h0.detach().to(self.device)

                kld_loss, nll_loss, o_ti, mse_loss = self.cal_pred_model_loss(s, a_lst, first_hidden)
                total_pred_model_loss += kld_loss + nll_loss

                loss_log["kld_loss"].append(kld_loss.mean())
                loss_log["nll_loss"].append(nll_loss.mean())
                loss_log["mse_loss"].append(mse_loss.mean())
                print(f"--- {time.time() - start_loss_time} seconds for pred_model loss ---")

            total_pred_model_loss /= self.num_memos
            loss_log["pred_model_loss"].append(total_pred_model_loss)

            self.optim_pred_model.zero_grad()
            total_pred_model_loss.mean().backward()
            self.optim_pred_model.step()
        print(f"--- {time.time() - start_time} seconds for pred_model training ---")

        start_time = time.time()
        pred_o_ti = []
        for epoch in range(self.K_epoch_policy):
            total_ppo_loss = 0
            for i in range(self.num_memos):
                start_loss_time = time.time()
                s, a, r, s_prime, done, prob_a, a_lst = self.make_batch(memory_list[i])
                first_hidden = memory_list[i].h0.detach().to(self.device)

                with torch.no_grad():
                    _, _, o_ti, _ = self.cal_pred_model_loss(s, a_lst, first_hidden)
                    if len(pred_o_ti) < i + 1:
                        pred_o_ti.append(o_ti)

                print(f"--- {time.time() - start_loss_time} seconds for pred_model loss in policy training ---")
                start_loss_time = time.time()
                policy_loss, critic_loss, entropy_bonus, kld_policy, advtg_mean = self.cal_ppo_loss(pred_o_ti[i], a, prob_a, r, done)
                ppo_loss = policy_loss + critic_loss + entropy_bonus
                total_ppo_loss += ppo_loss

                loss_log["policy_loss"].append(policy_loss.mean())
                loss_log["critic_loss"].append(critic_loss.mean())
                loss_log["entropy_bonus"].append(entropy_bonus.mean())
                loss_log["kld_policy"].append(kld_policy.mean())
                loss_log["advtg_mean"].append(advtg_mean.mean())
                print(f"--- {time.time() - start_loss_time} seconds for policy loss ---")

            total_ppo_loss /= self.num_memos
            loss_log["ppo_loss"].append(total_ppo_loss)

            self.optim_policy.zero_grad()
            total_ppo_loss.mean().backward()
            self.optim_policy.step()
        print(f"--- {time.time() - start_time} seconds for policy training ---")

        for k in loss_log.keys():
            try:
                loss_log[k] = torch.mean(torch.stack(loss_log[k]))
            except Exception as err:
                print()
                print(err, k)
        return loss_log, "No total loss"

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

        kld = abs(kld)
        if kld > 0.09:
            return 4
        if kld > 0.06:
            return 3
        if kld > 0.04:
            return 2
        if kld > 0.01:
            return 1
        return 0
