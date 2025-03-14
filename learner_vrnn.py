
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from actor_vrnn import Actor
from util import Memory
from config import Config

EPS = torch.finfo(torch.float).eps # numerical logs

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
    h0: list

    def __init__(self, actor: Actor, optim_pred_model, optim_policy, optimizer, config: Config):
        for key, value in vars(config).items():
            if key in self.__annotations__:
                setattr(self, key, value)

        self.actor = actor
        self.optim = optimizer

    def make_batch(self, memory: Memory):
        s, a, prob_a, r, s_prime, done, t, a_lst = \
            map(lambda key: torch.tensor(memory.exps[key]), memory.keys)
        return s, a, r, s_prime, done, prob_a, a_lst

    def make_offset_seq(self, s, offset: tuple, limit):
        idx = torch.arange(limit).unsqueeze(1) + torch.arange(offset[0], offset[1]).unsqueeze(0)
        idx = idx.clamp(max=len(s) - 1)
        # print(idx)
        res = s[idx].view(offset[1] - offset[0], -1, self.s_size)
        return res

    def make_pred_s_tis(self, s_truth, s, a, h):
        total_loss = []
        # print(s_truth.shape, s.shape, h.shape, a.shape)

        """
        1. get all start h_in with s_truth
        2. start p iters for prior and posterior
        """
        h_truth, h_cond = h, h
        o_ti = []
        for x_truth, x_cond, a_lst in zip(s_truth, s, a):
            iter_loss = []
            a_lst = torch.split(a_lst, 1, dim=-1)
            for i in range(self.p_iters):
                # print(x_truth.shape, x_cond.shape, h_truth.shape, h_cond.shape, a_lst[i].shape)
                loss, out  = self.actor.pred_model.reconstruct(x_truth, x_cond, h_truth.view(-1), h_cond.view(-1), a_lst[i])
                phi_x_truth, phi_z_truth, phi_x_cond, phi_z_cond = (data.view(1, 1, -1) for data in out)
                o_truth, h_truth = self.actor.rnn(torch.cat([phi_x_truth, phi_z_truth], dim=-1), h_truth)
                o_cond, h_cond  = self.actor.rnn(torch.cat([phi_x_cond, phi_z_cond], dim=-1), h_cond)
                kld_loss, nll_loss = loss
                iter_loss.append(kld_loss + nll_loss)
            total_loss.append(torch.stack(iter_loss))
            o_ti.append(o_truth)
        total_loss = torch.stack(total_loss).mean(dim=1).mean(dim=0)

        o_ti.append(torch.zeros_like(o_truth)) # for v_prime
        o_ti = torch.cat(o_ti, dim=1)[-1].unsqueeze(0)
        return total_loss, o_ti

    def cal_advantage(self, v_s, r, v_prime, done_mask):
        td_target = r + self.gamma * v_prime * done_mask
        delta = td_target - v_s

        advtg_lst = []
        advtg_t = 0.0
        for delta_t in reversed(delta):
            advtg_t = self.gamma * self.lmbda * advtg_t + delta_t.item()
            advtg_lst.append([advtg_t])
        advtg_lst.reverse()
        advantage = torch.tensor(advtg_lst, dtype=torch.float)
        return_target = advantage + v_s
        return advantage, return_target

    def make_pi_and_critic(self, o):
        second_hidden = o[0].unsqueeze(0)
        pi = self.actor.policy.pi(o)
        v = self.actor.policy.v(o)
        pi, v = pi.squeeze(0), v.squeeze(0)
        return pi, v, second_hidden.detach()

    def learn(self, memory_list: list[Memory]):
        keys = ["total_loss", "pred_model_loss", "ppo_loss",
                "policy_loss", "critic_loss", "entropy_bonus"]
        loss_log = {}
        for k in keys:
            loss_log[k] = []

        for epoch in range(self.K_epoch_learn):
            total_ppo_loss, total_pred_model_loss = 0, 0
            for i in range(self.num_memos):
                s, a, r, s_prime, done, prob_a, a_lst = self.make_batch(memory_list[i])
                first_hidden = memory_list[i].h0.detach()

                # pred model
                if self.p_iters > 0:
                    limit = len(s) - self.actor.delay
                    s_truth = self.make_offset_seq(s, (1, 2), limit).squeeze(0)
                    pred_model_loss, o_ti = self.make_pred_s_tis(s_truth, s[:limit], a_lst[:limit], first_hidden)

                total_pred_model_loss += pred_model_loss

                # policy
                pi, v_s, second_hidden = self.make_pi_and_critic(o_ti[:, :-1, :])
                _ , v_prime, _ = self.make_pi_and_critic(o_ti[:, 1:, :])
                advantage, return_target = self.cal_advantage(v_s, r[self.delay:], v_prime, done[self.delay:])

                pi_a, prob_a = pi.gather(1, a[self.delay:]), prob_a[:len(prob_a) - self.delay]
                ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
                policy_loss = torch.min(surr1, surr2).mean() # expected value

                critic_loss = self.critic_weight * F.smooth_l1_loss(v_s, return_target.detach())

                entropy = Categorical(pi).entropy().mean()
                entropy_bonus = self.entropy_weight * entropy

                ppo_loss = - policy_loss + critic_loss - entropy_bonus

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
            loss_log[k] = torch.mean(torch.stack(loss_log[k]))
        return loss_log, f'{loss_log["total_loss"]:.9f}'

    def _kld_gauss(self, mean_1, std_1, mean_2, std_2):
        """Using std to compute KLD"""

        kld_element =  (2 * torch.log(std_2 + EPS) - 2 * torch.log(std_1 + EPS) +
            (std_1.pow(2) + (mean_1 - mean_2).pow(2)) /
            std_2.pow(2) - 1)
        return	0.5 * torch.sum(kld_element)