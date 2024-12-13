
import torch
import torch.nn.functional as F
from actor import Actor
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
    delay: int
    p_iters: int
    num_memos: int
    num_actors: int
    T_horizon: int
    hidden_size: int
    h0: list

    def __init__(self, actor: Actor, optim_pred_model, optim_policy, config: Config):
        for key, value in vars(config).items():
            if key in self.__annotations__:
                setattr(self, key, value)

        self.actor = actor
        self.optim_pred_model = optim_pred_model
        self.optim_policy = optim_policy

    def make_batch(self, memory: Memory):
        s, a, prob_a, r, s_prime, done, t, a_lst = \
            map(lambda key: torch.stack(memory.exps[key]), memory.keys)
        return s, a, r, s_prime, done, prob_a, a_lst

    def make_pred_s_tis(self, s, a_lsts, h_in, limit):
        # s_ti = []
        # for i in range(limit):
        #     _, h_out, pred_s = self.actor.pred_present(s[i], a_lsts[i], h_in)
        #     s_ti.append(pred_s)
        #     h_in = h_out
        _, _, pred_s = self.actor.pred_present(s[:limit], a_lsts[:, :limit], h_in, self.p_iters)
        pred_s = pred_s.transpose(0, 1)
        return pred_s

    def make_offset_seq(self, s, offset: tuple, limit):
        idx = torch.arange(limit).unsqueeze(1) + torch.arange(offset[0], offset[1]).unsqueeze(0)
        idx = idx.clamp(max=len(s) - 1)
        target = s[idx].view(-1, offset[1] - offset[0], self.s_size)
        return target

    def learn_pred_model(self, memory_list: list[Memory]):

        keys = ["pred_model_loss"]
        loss_log = {}
        for k in keys:
            loss_log[k] = []

        if self.p_iters == 0:
            loss_log["pred_model_loss"] = torch.tensor(0, dtype=torch.float)
            return loss_log, "None"

        for epoch in range(self.K_epoch_pred_model):
            pred_model_loss = 0
            for i in range(self.num_memos):
                s, a, r, s_prime, done, prob_a, a_lst = self.make_batch(memory_list[i])
                first_hidden = memory_list[i].h0.detach()

                limit = len(s) - self.actor.delay
                target = self.make_offset_seq(s, (1, self.p_iters + 1), limit)
                pred = self.make_pred_s_tis(s.unsqueeze(1), a_lst.unsqueeze(0), first_hidden, limit)
                loss = self.actor.pred_model.criterion(pred, target)
                pred_model_loss += loss
            pred_model_loss /= self.num_memos
            # print(f"> Epoch {epoch + 1}. Loss : {total_loss.item()}")

            self.optim_pred_model.zero_grad()
            pred_model_loss.mean().backward()
            self.optim_pred_model.step()
            loss_log["pred_model_loss"].append(pred_model_loss)

        for k in loss_log.keys():
            loss_log[k] = torch.mean(torch.stack(loss_log[k]))
        return loss_log, f'{loss_log["pred_model_loss"]:.9f}'

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

    # def make_pi_a(self, s, a, h_in, a_lst, limit):
    #     probs = []
    #     for i in range(limit):
    #         _, pi, h_out, _ = self.actor.sample_action(s[i], a_lst[i], h_in)
    #         probs.append(pi.view(-1))
    #         h_in = h_out
    #     return torch.stack(probs).gather(1,a)

    # def make_pi_a_by_true_s(self, s, a, h_in):
    #     pi, _ = self.actor.pred_pi(s, h_in)
    #     return pi.squeeze(1).gather(1,a)

    def make_pi_and_critic(self, s, h_in):
        pi, v, second_hidden = self.actor.pred_prob_and_critic(
            s.unsqueeze(1), h_in
        )
        pi, v = pi[self.delay:], v[self.delay:]
        return pi.squeeze(1), v.squeeze(1), second_hidden

    # def make_pi_and_critic_by_sample(self, s, a_lst, h_in):
    #     o, h_out, pred_s = self.actor.pred_present(s.unsqueeze(1), a_lst.unsqueeze(0), h_in, self.p_iters)
    #     pi = self.actor.policy.pi(o)
    #     v = self.actor.policy.v(o)
    #     return pi.squeeze(0), v.squeeze(0), h_out[:, 0].unsqueeze(0)

    # def make_pi_and_critic(self, s, a_lst, h_in):
    #     v = self.actor.pred_critic(s.unsqueeze(1), h_in)
    #     pi, h_out = self.actor.pred_prob(s.unsqueeze(1), a_lst.unsqueeze(0), h_in)
    #     return pi.squeeze(0), v.squeeze(1), h_out

    def learn_policy(self, memory_list: list[Memory]):
        keys = ["ppo_loss", "policy_loss", "critic_loss", "entropy_bonus"]
        loss_log = {}
        for k in keys:
            loss_log[k] = []

        for epoch in range(self.K_epoch_policy):
            ppo_loss = 0
            for i in range(self.num_memos):
                s, a, r, s_prime, done, prob_a, a_lst = self.make_batch(memory_list[i])
                first_hidden  = memory_list[i].h0.detach()
                # second_hidden = memory_list[i].h1.detach()

                # pi, v_s, second_hidden = self.make_pi_and_critic_by_sample(s[:-self.delay], a_lst[:-self.delay], first_hidden)
                # s_offset = self.make_offset_seq(s, (0, self.delay + 1), len(s) - self.delay).transpose(0, 1)
                pi, v_s, second_hidden = self.make_pi_and_critic(s, first_hidden)
                # pi, v_s, second_hidden= self.make_pi_and_critic(s[:-self.delay], a_lst[:-self.delay], first_hidden)

                # _, v_prime, _ = self.make_pi_and_critic_by_sample(s_prime[:-self.delay], a_lst[1:-self.delay + 1], second_hidden)
                # s_prime_offset = self.make_offset_seq(s_prime, (0, self.delay + 1), len(s_prime) - self.delay).transpose(0, 1)
                _ , v_prime, _ = self.make_pi_and_critic(s_prime, second_hidden)
                # _ , v_prime, _ = self.make_pi_and_critic(s_prime[:-self.delay], a_lst[:-self.delay]+1, second_hidden)

                # _, _, second_hidden = self.actor.rnn(s[0].unsqueeze(1), first_hidden)
                # print(second_hidden.shape)

                advantage, return_target = self.cal_advantage(v_s, r[:len(r) - self.delay], v_prime, done[self.delay:])
                # advantage, td_target = advantage[self.actor.delay:], td_target[self.actor.delay:]

                # using pred_s or true s
                # v_s, v_prime = v_s[self.actor.delay:], v_prime[self.actor.delay:]
                # a, prob_a = a[self.actor.delay:], prob_a[:-self.actor.delay]
                # limit = len(s) - self.actor.delay
                # pi_a = self.make_pi_a(s, a, first_hidden, a_lst, limit)
                pi_a, prob_a = pi.gather(1, a[self.delay:]), prob_a[:len(prob_a) - self.delay]
                ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
                policy_loss = torch.min(surr1, surr2).mean() # expected value

                critic_loss = self.critic_weight * F.smooth_l1_loss(v_s, return_target.detach())

                # prob or prob_a?
                # entropy = -(pi_a * torch.log(pi_a)).sum(dim=0)
                self.actor.dist.set_probs(pi)
                entropy = self.actor.dist.entropy().mean()
                entropy_bonus = self.entropy_weight * entropy

                loss = - policy_loss + critic_loss - entropy_bonus
                ppo_loss += loss
                loss_log["policy_loss"].append(policy_loss.mean())
                loss_log["critic_loss"].append(critic_loss.mean())
                loss_log["entropy_bonus"].append(entropy_bonus.mean())

            ppo_loss /= self.num_memos
            # print(f"> Epoch {epoch + 1}. Loss : {total_loss.item()}")

            self.optim_policy.zero_grad()
            ppo_loss.mean().backward()
            self.optim_policy.step()

            loss_log["ppo_loss"].append(ppo_loss)

        for k in loss_log.keys():
            loss_log[k] = torch.mean(torch.stack(loss_log[k]))
        return loss_log, f'{loss_log["ppo_loss"]:.9f}'
