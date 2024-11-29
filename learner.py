
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

    def learn_pred_model(self, memory_list: list[Memory], h0):
        if self.p_iters == 0: return

        loss_log = []
        for epoch in range(self.K_epoch_pred_model):
            total_loss = 0
            for i in range(self.num_memos):
                s, a, r, s_prime, done, prob_a, a_lst = self.make_batch(memory_list[i])
                first_hidden = h0

                limit = len(s) - self.actor.delay
                target = self.make_offset_seq(s, (1, self.p_iters + 1), limit)
                pred = self.make_pred_s_tis(s.unsqueeze(1), a_lst.unsqueeze(0), first_hidden, limit)
                loss = self.actor.pred_model.criterion(pred, target)
                total_loss += loss
            total_loss /= self.num_memos
            # print(f"> Epoch {epoch + 1}. Loss : {total_loss.item()}")

            self.optim_pred_model.zero_grad()
            total_loss.mean().backward()
            self.optim_pred_model.step()
            loss_log.append(total_loss)
        loss_mean = torch.mean(torch.stack(loss_log))
        print(f"|| Avg Loss  : {loss_mean}")
        return loss_mean

    def cal_advantage(self, v_s, r, v_prime, done_mask):
        td_target = r + self.gamma * v_prime * done_mask
        delta = td_target - v_s
        delta = delta.detach().numpy()

        advantage_lst = []
        advantage = 0.0
        for delta_t in delta[::-1]:
            advantage = self.gamma * self.lmbda * advantage + delta_t[0]
            advantage_lst.append([advantage])
        advantage_lst.reverse()
        advantage = torch.tensor(advantage_lst, dtype=torch.float)
        return advantage, td_target

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

    def make_pi_and_critic(self, s, s_offset, h_in):
        pi, v, second_hidden = self.actor.pred_prob_and_critic(
            s.unsqueeze(1), s_offset, h_in
        )
        return pi.squeeze(1), v.squeeze(1), second_hidden

    # def make_pi_and_critic(self, s, a_lst, h_in):
    #     v = self.actor.pred_critic(s.unsqueeze(1), h_in)
    #     pi, h_out = self.actor.pred_prob(s.unsqueeze(1), a_lst.unsqueeze(0), h_in)
    #     return pi.squeeze(0), v.squeeze(1), h_out

    def learn_policy(self, memory_list: list[Memory], h0):
        loss_log = []
        for epoch in range(self.K_epoch_policy):
            total_loss = 0
            for i in range(self.num_memos):
                s, a, r, s_prime, done, prob_a, a_lst = self.make_batch(memory_list[i])

                s_offset = self.make_offset_seq(s, (0, self.delay + 1), len(s) - self.delay).transpose(0, 1)
                first_hidden  = h0
                pi, v_s, second_hidden= self.make_pi_and_critic(s, s_offset, first_hidden)
                # pi, v_s, second_hidden= self.make_pi_and_critic(s[:-self.delay], a_lst[:-self.delay], first_hidden)

                s_prime_offset = self.make_offset_seq(s_prime, (0, self.delay + 1), len(s_prime) - self.delay).transpose(0, 1)
                _ , v_prime, _ = self.make_pi_and_critic(s_prime, s_prime_offset, second_hidden)
                # _ , v_prime, _ = self.make_pi_and_critic(s_prime[:-self.delay], a_lst[:-self.delay]+1, second_hidden)

                # _, _, second_hidden = self.actor.rnn(s[0].unsqueeze(1), first_hidden)
                # print(second_hidden.shape)

                advantage, td_target = self.cal_advantage(v_s, r[self.delay:], v_prime, done[self.p_iters:])
                # advantage, td_target = advantage[self.actor.delay:], td_target[self.actor.delay:]

                # using pred_s or true s
                # v_s, v_prime = v_s[self.actor.delay:], v_prime[self.actor.delay:]
                # a, prob_a = a[self.actor.delay:], prob_a[:-self.actor.delay]
                # limit = len(s) - self.actor.delay
                # pi_a = self.make_pi_a(s, a, first_hidden, a_lst, limit)
                pi_a, prob_a = pi.gather(1, a[self.delay:]), prob_a[:-self.delay]
                ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
                policy_loss = -torch.min(surr1, surr2)

                critic_loss = self.critic_weight * F.smooth_l1_loss(v_s, td_target)

                entropy = -(pi_a * torch.log(pi_a)).sum(dim=-1).mean()
                entropy_bonus = self.entropy_weight * entropy

                loss = policy_loss + critic_loss - entropy_bonus
                total_loss += loss.mean()
            total_loss /= self.num_memos
            # print(f"> Epoch {epoch + 1}. Loss : {total_loss.item()}")

            self.optim_policy.zero_grad()
            total_loss.mean().backward()
            self.optim_policy.step()
            loss_log.append(total_loss)
        loss_mean = torch.mean(torch.stack(loss_log))
        print(f"|| Avg Loss  : {loss_mean}")
        return loss_mean
