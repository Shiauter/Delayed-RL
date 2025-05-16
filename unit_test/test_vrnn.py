import torch
import gymnasium as gym
import time

from vae import VAE
from actor_vrnn import RNN
from actor import Actor
from learner_vrnn import Learner
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


def train_vrnn(memory_list: list[Memory], config: Config, learner: Learner):
    # output的指標需要先加在keys中
    keys = [
        "pred_model_loss", "kld_loss", "nll_loss",
        "mse_loss"
    ]

    loss_log = {}
    for k in keys:
        loss_log[k] = []

    avg_loss = []
    for epoch in range(1, config.K_epoch_learn + 1):
        total_pred_model_loss = []
        for i in range(config.num_memos):
            start_loss_time = time.time()
            s, a, r, s_prime, done, prob_a, a_lst = learner.make_batch(memory_list[i])
            first_hidden = memory_list[i].h0.detach().to(config.device)

            kld_loss, nll_loss, o_ti, mse_loss = learner.cal_pred_model_loss(s, a_lst, first_hidden)
            if config.reconst_loss_method == "NLL":
                total_pred_model_loss.append(kld_loss + nll_loss)
            elif config.reconst_loss_method == "MSE":
                total_pred_model_loss.append(kld_loss + mse_loss)

            loss_log["kld_loss"].append(kld_loss)
            loss_log["nll_loss"].append(nll_loss)
            loss_log["mse_loss"].append(mse_loss)
            # print(f"--- {time.time() - start_loss_time} seconds for pred_model loss ---")

        total_pred_model_loss = torch.stack(total_pred_model_loss).mean()
        loss_log["pred_model_loss"].append(total_pred_model_loss)
        # print(f"total_pred_model_loss: {total_pred_model_loss} / ep. {epoch + 1}")

        avg_loss.append(total_pred_model_loss.mean().detach())
        learner.optim_pred_model.zero_grad()
        print(f"Loss: {total_pred_model_loss.mean():.6f}, Epoch: {epoch} / {config.K_epoch_pred_model}")
        total_pred_model_loss.mean().backward()
        learner.optim_pred_model.step()
    print(f"Avg Loss: {torch.stack(avg_loss).mean():.6f}")


if __name__ == "__main__":
    config = Config(env_name="CartPole-v1")
    config.delay = 1
    config.p_iters = 1
    learner = Learner(config)
    actor = Actor(config)
    model_state = torch.load("lstm_ppo_high_score_agent.tar", weights_only=True)
    actor.load_params(model_state)

    num_batch_training = 100
    num_episode_per_batch = 5
    for batch in range(1, num_batch_training + 1):
        memory_list = []
        print("Scores - ")
        for ep in range(num_episode_per_batch):
            memory = eval(config, actor)
            memory_list.append(memory)
            print(f"{ep + 1}. {memory.score}, ")
        print(f"Start training Batch {batch}")
        train_vrnn(memory_list, config, learner)

