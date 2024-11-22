
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
    eps_clip: float
    K_epoch_policy: int
    K_epoch_pred_model: int
    p_iters: int
    num_memos: int
    num_actors: int
    T_horizon: int

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

    def make_pred_s_tis(self, s, a_lst, h_in, limit):
        s_ti = []
        for i in range(limit):
            _, h_out, pred_s = self.actor.pred_present(s[i], a_lst[i], h_in)
            s_ti.append(pred_s)
            h_in = h_out
        return torch.stack(s_ti)

    def learn_pred_model(self, memory_list: list[Memory]):
        if self.p_iters == 0: return

        loss_log = []
        for epoch in range(self.K_epoch_pred_model):
            total_loss = 0
            for i in range(self.num_memos):
                s, a, r, s_prime, done, prob_a, a_lst = self.make_batch(memory_list[i])
                first_hidden  = self.actor.h0

                target = []
                limit = len(s) -self.p_iters
                for i in range(limit):
                    start, end = i + 1, min(i + self.p_iters, len(s) - 1) + 1
                    before_done = s[start : end].tolist()
                    after_done = [s[-1] for _ in range(self.p_iters - (end - start))]
                    target.append(before_done + after_done)
                target = torch.tensor(target, dtype=torch.float).view(-1, self.p_iters, self.s_size)

                pred = self.make_pred_s_tis(s, a_lst, first_hidden, limit)
                loss = self.actor.pred_model.criterion(pred, target)
                total_loss += loss
            total_loss /= self.num_memos
            print(f"> Epoch {epoch + 1}. Loss : {total_loss.item()}")

            self.optim_pred_model.zero_grad()
            total_loss.mean().backward()
            self.optim_pred_model.step()
            loss_log.append(total_loss)
        print(f"Avg Loss : {torch.mean(torch.stack(loss_log))}")

    def test(self, memory):
        s, a, r, s_prime, done, prob_a, a_lst = self.make_batch(memory)
        res = self.actor.pred_critic(s, self.actor.h0)
        print(res.shape)
        res = self.actor.pred_critic(s[0], self.actor.h0)
        print(res.shape)

    def cal_advantage(self, s, r, s_prime, done_mask, first_hidden, second_hidden):
        v_prime = self._v(s_prime, second_hidden)
        td_target = r + self.gamma * v_prime * done_mask
        v_s = self._v(s, first_hidden)
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

    def learn_policy(self, memory_list: list[Memory]):
        for epoch in range(self.K_epoch_policy):
            total_loss = 0
            for i in range(self.num_memos):
                s, a, r, s_prime, done, prob_a, a_lst = self.make_batch(memory_list[i])
                first_hidden  = self.actor.h0
                second_hidden = self.actor.pred_critic(s[0], self.actor.h0)[1]

                advantage, td_target = self.cal_advantage(s, r, s_prime, done, first_hidden, second_hidden)

                # a, prob, h_out, pred_s_ti = model.sample_action(s, a_lst, h_in)
                pi, h = self._pi(s, first_hidden)
                pi_a = pi.squeeze(1).gather(1,a)
                ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
                loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self._v(s, first_hidden), td_target.detach())
                total_loss += loss
            total_loss /= self.num_memos
            print(f"> Epoch {epoch + 1}. Loss : {total_loss.item()}")

            self.optim_policy.zero_grad()
            total_loss.mean().backward()
            self.optim_policy.step()