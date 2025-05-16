import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import torch

from actor import Actor
from config import Config

def eval(config: Config, model: Actor):
    with torch.no_grad():
        env = gym.make(config.env_name, render_mode="rgb_array")

        s, _ = env.reset()
        h_out = torch.zeros(config.h0).float()
        a_lst = [i % 2 for i in range(config.delay)]
        done = False
        frames = []
        total_reward = 0

        while not done:
            h_in = h_out
            a, _, h_out, _ = model.sample_action(
                torch.from_numpy(s).view(1, 1, -1),
                torch.tensor(a_lst).view(1, 1, -1),
                h_in
            )

            a = a.item()
            a_lst.append(a)

            delay_a = a_lst[0]
            s_prime, r, terminated, truncated, _ = env.step(delay_a)
            total_reward += r
            done = terminated or truncated
            frame = env.render()
            frames.append(frame)

            s = s_prime
            a_lst.pop(0)
            if done:
                print(f"Total Reward: {total_reward}")
                break

if __name__ == "__main__":
    config = Config(env_name="CartPole-v1")
    actor = Actor(config)
    model_state = torch.load("lstm_ppo_high_score_agent.tar", weights_only=True)
    actor.load_params(model_state)
    for _ in range(5):
        eval(config, actor)