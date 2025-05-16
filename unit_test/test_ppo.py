import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import gymnasium as gym
import time

from actor import Actor
from learner import Learner
from config import Config
from util import Memory


def eval(config: Config, model: Actor):
    with torch.no_grad():
        env = gym.make(config.env_name, render_mode="rgb_array")
        memory = Memory(config.T_horizon)

        s, _ = env.reset()
        h_out = torch.zeros(config.h0).float()
        a_lst = [i % 2 for i in range(config.delay)]
        done = False
        frames = []
        total_reward = 0

        while not done:
            h_in = h_out
            a, prob, h_out, _ = model.sample_action(
                torch.from_numpy(s).view(1, 1, -1),
                torch.tensor(a_lst).view(1, 1, -1),
                h_in
            )

            prob = prob.view(-1)
            a = a.item()
            a_lst.append(a)

            delay_a = a_lst[0]
            s_prime, r, terminated, truncated, _ = env.step(delay_a)
            total_reward += r
            done = terminated or truncated
            exp = {
                "states": s.tolist(),
                "actions": [delay_a],
                "probs": [prob[a].item()],
                "rewards": [r / 100.0],
                "states_prime": s_prime.tolist(),
                "dones": [0 if done else 1],
                "a_lsts": a_lst[:-1]
            }
            memory.store(**exp)
            memory.set_hidden(h_in.detach())
            memory.score += r

            frame = env.render()
            frames.append(frame)

            s = s_prime
            a_lst.pop(0)
            if done:
                break

        return memory


def train_policy(memory_list: list[Memory], config: Config, learner: Learner):
    # output的指標需要先加在keys中
    keys = [
        "ppo_loss", "policy_loss", "critic_loss",
        "entropy_bonus", "kld_policy", "advtg_mean",
        "clipped_percentage", "avg_clipped_distance"
    ]

    loss_log = {}
    for k in keys:
        loss_log[k] = []

    avg_loss = []

    # policy
    start_time = time.time()
    for epoch in range(1, config.K_epoch_learn + 1):
        total_ppo_loss = 0
        for i in range(config.num_memos):
            start_loss_time = time.time()
            s, a, r, s_prime, done, prob_a, a_lst = learner.make_batch(memory_list[i])
            first_hidden = memory_list[i].h0.detach()

            pi, v_s, second_hidden = learner.make_pi_and_critic(s, a_lst, first_hidden)
            _ , v_prime, _ = learner.make_pi_and_critic(s_prime, a_lst, second_hidden)
            advantage, return_target = learner.cal_advantage(v_s, r[learner.delay:], v_prime, done[learner.delay:])
            pi_a, prob_a = pi.gather(1, a[learner.delay:]), prob_a[:len(prob_a) - learner.delay]
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - learner.eps_clip, 1 + learner.eps_clip) * advantage
            policy_loss = torch.min(surr1, surr2).mean() # expected value

            critic_loss = learner.critic_weight * F.smooth_l1_loss(v_s, return_target.detach())

            entropy = Categorical(pi).entropy().mean()
            entropy_bonus = learner.entropy_weight * entropy

            loss = - policy_loss + critic_loss - entropy_bonus
            total_ppo_loss += loss
            loss_log["policy_loss"].append(policy_loss.mean())
            loss_log["critic_loss"].append(critic_loss.mean())
            loss_log["entropy_bonus"].append(entropy_bonus.mean())

        total_ppo_loss /= config.num_memos
        loss_log["ppo_loss"].append(total_ppo_loss)
        # print(f"total_policy_loss: {total_ppo_loss} / ep. {epoch + 1}")

        avg_loss.append(total_ppo_loss.mean().detach())
        learner.optim.zero_grad()
        print(f"Loss: {total_ppo_loss.mean():.6f}, Epoch: {epoch} / {config.K_epoch_policy}")
        total_ppo_loss.mean().backward()
        learner.optim.step()
    print(f"Avg Loss: {torch.stack(avg_loss).mean():.6f}")


if __name__ == "__main__":
    config = Config(env_name="CartPole-v1")
    config.delay = 0
    config.p_iters = 0
    learner = Learner(config)
    actor = Actor(config)

    num_batch_training = 500
    for batch in range(1, num_batch_training + 1):
        memory_list = []
        print("Scores - ")
        for ep in range(config.num_memos):
            memory = eval(config, actor)
            memory_list.append(memory)
            print(f"{ep + 1}. {memory.score}, ")
        print(f"Start training Batch {batch}")
        learner.actor.load_params(actor.output_params())
        train_policy(memory_list, config, learner)
        actor.load_params(learner.actor.output_params())
