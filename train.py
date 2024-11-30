import gymnasium as gym
import torch
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

from actor import Actor
from learner import Learner
from util import Memory, merge_dict
from config import Config

def collect_data(env_name, model_state, config: Config, conn):
    env = gym.make(env_name)
    model = Actor(config)
    model.load_params(model_state)
    memory = Memory(model.T_horizon)

    s, _ = env.reset()
    h_out = torch.zeros(config.h0).float()
    a_lst = [i % 2 for i in range(model.delay)]
    done = False

    while not done:
        for t in range(model.T_horizon):
            h_in = h_out
            a, prob, h_out, _ = model.sample_action(
                torch.from_numpy(s).view(1, 1, -1), # (seq_len, batch, s_size)
                torch.tensor(a_lst).view(1, 1, -1),
                h_in
            )
            # a, prob, h_out, _ = model.sample_action(
            #     torch.from_numpy(s), torch.tensor(a_lst), h_in
            #     )
            prob = prob.view(-1)
            a_lst.append(a)

            delay_a = a_lst.pop(0)
            s_prime, r, terminated, truncated, _ = env.step(delay_a)
            done = terminated or truncated

            exp = {
                "states": torch.from_numpy(s).detach(),
                "actions": torch.tensor(delay_a).detach().view(-1),
                "probs": prob[a].detach().view(-1),
                "rewards": torch.tensor(r / 100.0).detach().view(-1),
                "states_prime": torch.from_numpy(s_prime).detach(),
                "dones": torch.tensor(0 if done else 1).detach().view(-1),
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
            {"params": actor.policy.parameters()},
            # {"params": actor.pred_model.parameters()}
        ],
        lr=config.lr_policy
    )
    learner = Learner(actor, optim_pred_model, optim_policy, config)
    writer = SummaryWriter(f"{config.log_root}/{config.experiment_name}")
    config_text = config.get_json()
    writer.add_text('Configuration', config_text)

    print("=== Start ===\n")
    for ep in range(1, config.K_epoch_training + 1):
        print(f"* Epoch {ep}.")
        print(f"> {'Collecting data...':<35}", end=" ")
        memory_list = parallel_process(config, actor.output_params())
        total_score = sum([memo.score for memo in memory_list])
        score_mean = total_score / config.num_memos
        print(f"|| Avg score : {score_mean:.1f}")

        print(f"> {'Training for predictive model...':<35}", end=" ")
        pred_model_log, avg_loss = learner.learn_pred_model(memory_list)
        print(f"|| Avg Loss  : {avg_loss}")

        print(f"> {'Training for policy...':<35}", end=" ")
        ppo_log, avg_loss = learner.learn_policy(memory_list)
        print(f"|| Avg Loss  : {avg_loss}")

        model_state = actor.output_params()
        learner.actor.load_params(model_state)

        saved_path = f"{config.model_root}/{config.experiment_name}/epoch_{ep}_{config.model_name}"
        torch.save(actor, saved_path)
        print(f"> Model is saved in {saved_path}")

        try:
            log = merge_dict(pred_model_log, ppo_log)
            log["score"] = score_mean
            for k, v in log.items():
                writer.add_scalar(k, v, ep)
        except KeyError as e:
            print(f"Error: {e}")
        print()
    print("=== Finished ===\n")
    writer.close()