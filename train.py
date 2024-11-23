import gymnasium as gym
import torch
import torch.optim as optim
import torch.multiprocessing as mp

from actor import Actor
from learner import Learner
from util import Memory
from config import Config

def collect_data(env_name, model_state, config: Config, conn):
    env = gym.make(env_name)
    model = Actor(config)
    model.load_params(model_state)
    memory = Memory(model.T_horizon)

    s, _ = env.reset()
    h_out = config.h0
    a_lst = [i % 2 for i in range(model.delay)]
    done = False

    while not done:
        for t in range(model.T_horizon):
            h_in = h_out
            a, prob, h_out, _ = model.sample_action(
                torch.from_numpy(s), torch.tensor(a_lst), h_in
                )
            prob = prob.view(-1)
            a_lst.append(a)

            delay_a = a_lst.pop(0)
            s_prime, r, terminated, truncated, _ = env.step(delay_a)
            done = terminated or truncated

            exp = {
                "states": torch.from_numpy(s).detach(),
                "actions": torch.tensor(delay_a).detach().view(-1),
                "probs": prob[delay_a].detach().view(-1),
                "rewards": torch.tensor(r / 100.0).detach().view(-1),
                "states_prime": torch.from_numpy(s_prime).detach(),
                "dones": torch.tensor(0.0 if done else 1.0).detach().view(-1),
                "timesteps": torch.tensor(t).detach().view(-1),
                "a_lsts": torch.tensor(a_lst).detach().view(-1)
            }
            memory.store(**exp)
            memory.score += r

            s = s_prime
            if done:
                break
    env.close()

    conn.send(memory)
    conn.close()

def parallel_process(config: Config, model_state) -> list[Memory]:
    env_name, num_memos, num_actors = config.env_name, config.num_memos, config.num_actors

    processes = []
    parent_conns, child_conns = zip(*[mp.Pipe() for _ in range(num_memos)])
    aggregated_data = []
    num_batches = (num_memos + num_actors - 1) // num_actors
    for batch_idx in range(num_batches):
        start = batch_idx * num_actors
        end = min(start + num_actors, num_memos)

        for i in range(start, end):
            p = mp.Process(target=collect_data, args=(env_name, model_state, config, child_conns[i]))
            processes.append(p)
            p.start()

        for i in range(start, end):
            data = parent_conns[i].recv()
            aggregated_data.append(data)

        for p in processes:
            p.join()

        processes.clear()

    return aggregated_data

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    config = Config(env_name="CartPole-v1")
    actor = Actor(config)
    optim_pred_model = optim.Adam(
        [
            {"params": actor.rnn.parameters()},
            {"params": actor.pred_model.parameters()}
        ],
        lr=config.lr_pred_model
    )
    optim_policy = optim.Adam(
        [
            {"params": actor.rnn.parameters()},
            {"params": actor.policy.parameters()}
        ],
        lr=config.lr_policy
    )
    learner = Learner(actor, optim_pred_model, optim_policy, config)

    print("=== Start ===")
    for ep in range(1, config.K_epoch_training + 1):
        print(f"Batch. {ep}")
        print(" - Collecting data...")
        memory_list = parallel_process(config, actor.output_params())
        total_score = sum([memo.score for memo in memory_list])
        print(f"Avg score : {total_score / config.num_memos:.1f}")

        print(" - Training for predictive model...")
        learner.learn_pred_model(memory_list, config.h0)

        print(" - Training for policy...")
        learner.learn_policy(memory_list, config.h0)

        model_state = actor.output_params()
        learner.actor.load_params(model_state)
    print("=== Finished ===")