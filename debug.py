import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import torch
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

from actor import Actor
from learner import Learner
from util import Memory, merge_dict, check_dir_exist, action_data_sample
from config import Config


def worker(env_name, config: Config, conn):
    env = gym.make(env_name)
    memory = Memory(config.T_horizon)
    done = False
    try:
        while True:
            cmd, data = conn.recv()
            if cmd == "reset":
                s, _ = env.reset(seed=123)
                h_out = torch.zeros(config.h0).float()
                a_lst = [action[0] for action in action_data_sample] ###
                conn.send((s, a_lst, h_out))

            elif cmd == "step":
                h_in = h_out
                a, prob, h_out, _ = data
                prob = prob.view(-1)
                a = a.item()
                if len(a_lst) == 0:
                    a_lst.append(a)

                delay_a = a_lst[0]
                s_prime, r, terminated, truncated, _ = env.step(delay_a)
                done = terminated or truncated
                exp = {
                    "states": s.tolist(),
                    "actions": [delay_a],
                    "probs": [prob[a].item()],
                    "rewards": [r / 100.0],
                    "states_prime": s_prime.tolist(),
                    "dones": [0 if done else 1],
                    "a_lsts":[] ###
                }
                memory.store(**exp)
                memory.set_hidden(h_in.detach())
                memory.score += r

                s = s_prime
                a_lst.pop(0)
                conn.send((s, a_lst, h_in, done))
            elif cmd == "get_memo":
                conn.send(memory)
            elif cmd == "close":
                break
    finally:
        env.close()
        conn.close()


def event_loop(config: Config, actor: Actor):
    env_name, num_memos, num_actors = config.env_name, config.num_memos, config.num_actors

    aggregated_data = []
    while num_memos > 0:
        current_num_workers = min(num_memos, num_actors)
        parent_conns, child_conns = zip(*[mp.Pipe() for _ in range(current_num_workers)])
        processes = [mp.Process(target=worker, args=(env_name, config, conn)) for conn in child_conns]

        for p in processes:
            p.start()

        for conn in parent_conns:
            conn.send(("reset", None))
        observations = [conn.recv() for conn in parent_conns]

        active_envs = [True] * current_num_workers

        while any(active_envs):
            actions = []
            for obs in observations:
                if obs is not None:
                    s, a_lst, h_in = obs
                    a, prob, h_out, _ = actor.sample_action(
                        torch.from_numpy(s).view(1, 1, -1), # (seq_len, batch, s_size)
                        torch.tensor(a_lst).view(1, 1, -1),
                        h_in
                    )
                    actions.append((a, prob, h_out, _))
                else:
                    actions.append(None)

            for conn, action, active in zip(parent_conns, actions, active_envs):
                if active and action is not None:
                    conn.send(("step", action))

            for i, conn in enumerate(parent_conns):
                if active_envs[i]:
                    s, a_lst, h_in, done = conn.recv()
                    observations[i] = (s, a_lst, h_in)
                    if done:
                        active_envs[i] = False
                else:
                    observations[i] = None

        for conn in parent_conns:
            conn.send(("get_memo", None))
            memo = conn.recv()
            aggregated_data.append(memo)
            conn.send(("close", None))

        for p in processes:
            p.join()

        num_memos -= num_actors

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
            {"params": actor.policy.parameters()}
        ],
        lr=config.lr_policy
    )
    optimizer = optim.Adam(
        [
            {"params": actor.rnn.parameters()},
            {"params": actor.policy.parameters()},
            {"params": actor.pred_model.parameters()}
        ],
        lr=config.lr
    )
    learner = Learner(actor, config, optim_pred_model, optim_policy, optimizer)

    print()
    print("=== Start ===\n")
    for ep in range(1, config.K_epoch_training + 1):
        print(f"* Epoch {ep}.")
        print(f"> {'Collecting data...':<35}", end=" ")

        model_state = actor.output_params()
        learner.actor.load_params(model_state)
        memory_list = event_loop(config, actor)
        total_score = sum([memo.score for memo in memory_list])
        avg_score = total_score / config.num_memos
        print(f"|| Avg score : {avg_score:.1f}")

        # print(f"> {'Training...':<35}", end=" ")
        # loss_log, avg_loss_str = learner.learn(memory_list)
        # print(f"|| Avg Loss  : {avg_loss_str}")

        # end_flag = False
        # for memo in memory_list:
        #     if memo.score >= 400:
        #         print(memo.exps['actions'])
        #         end_flag = True
        #         break
        # if end_flag:
        #     break
        print()
    print("=== Finished ===\n")