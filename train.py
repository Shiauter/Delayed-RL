import gymnasium as gym
import torch
import torch.optim as optim
import torch.multiprocessing as mp

from actor import Actor
from learner import Learner
from util import Memory
from config import Config


def collect_data(env_name, model: Actor, conn):
    env = gym.make(env_name)
    memory = Memory(model.T_horizon)

    s, _ = env.reset()
    h_out = model.h0
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
                # "h_ins": tuple(map(lambda t: t.detach(), h_in)),
                # "h_outs": tuple(map(lambda t: t.detach(), h_out)),
                "dones": torch.tensor(0 if done else 1).detach().view(-1),
                "timesteps": torch.tensor(t).detach().view(-1),
                "a_lsts": torch.tensor(a_lst).detach().view(-1)
            }
            memory.store(**exp)
            s = s_prime
            memory.score += r
            if done:
                break
    env.close()

    conn.send(memory)
    conn.close()

def parallel_process(env_name, model: Actor, num_memos, num_actors) -> list[Memory]:
    processes = []
    parent_conns, child_conns = zip(*[mp.Pipe() for _ in range(num_memos)])

    aggregated_data = []
    num_batches = (num_memos + num_actors - 1) // num_actors
    for batch_idx in range(num_batches):
        start = batch_idx * num_actors
        end = min(start + num_actors, num_memos)

        for i in range(start, end):
            p = mp.Process(target=collect_data, args=(env_name, model, child_conns[i]))
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
        lr=1e-3
    )
    optim_policy = optim.Adam(
        [
            {"params": actor.rnn.parameters()},
            {"params": actor.policy.parameters()}
        ],
        lr=3e-4
    )
    learner = Learner(actor, optim_pred_model, optim_policy, config)
    K_epoch_training = 5

    print("Start.")
    for ep in range(1, K_epoch_training + 1):
        print(f"Batch. {ep} - Collecting data...")
        memory_list = parallel_process(config.env_name, actor, config.num_memos, config.num_actors)
        total_score = sum([memo.score for memo in memory_list])
        print(f"Avg score : {total_score / config.num_memos:.1f}")

        # learner.learn_pred_model(memory_list)
        learner.test(memory_list[0])

    print("Finished.")