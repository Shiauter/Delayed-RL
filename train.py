import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import torch
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

from actor_vrnn import Actor
from learner_vrnn import Learner
from util import Memory, merge_dict, check_dir_exist
from config import Config


def recording_eval(env_name, model_state, record_dir:str, epoch:int):
    with torch.no_grad():
        env = gym.make(env_name, render_mode="rgb_array")
        model = Actor(config)
        model.load_params(model_state)

        s, _ = env.reset()
        h_out = torch.zeros(config.h0).float()
        a_lst = [i % 2 for i in range(config.delay)]
        done = False
        frames = []

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
            done = terminated or truncated
            frame = env.render()
            frames.append(frame)

            s = s_prime
            a_lst.pop(0)
            if done:
                break
        record_path = f"{record_dir}/epoch_{epoch}.mp4"
        clip = ImageSequenceClip(frames, fps=30)
        clip.write_videofile(
            record_path,
            codec="libx264",
            logger=None
        )
        env.close()
        return record_path

# def collect_data(env_name, model_state, config: Config, conn):
#     try:
#         with torch.no_grad():
#             env = gym.make(env_name)
#             model = Actor(config)
#             model.load_params(model_state)
#             memory = Memory(config.T_horizon)

#             s, _ = env.reset()
#             h_out = torch.zeros(config.h0).float()
#             a_lst = [i % 2 for i in range(config.delay)]
#             done = False

#             while not done:
#                 for t in range(model.T_horizon):
#                     h_in = h_out
#                     a, prob, h_out, _ = model.sample_action(
#                         torch.from_numpy(s).view(1, 1, -1), # (seq_len, batch, s_size)
#                         torch.tensor(a_lst).view(1, 1, -1),
#                         h_in
#                     )
#                     # a, prob, h_out, _ = model.sample_action(
#                     #     torch.from_numpy(s), torch.tensor(a_lst), h_in
#                     #     )
#                     prob = prob.view(-1)
#                     a = a.item()
#                     a_lst.append(a)

#                     delay_a = a_lst.pop(0)
#                     s_prime, r, terminated, truncated, _ = env.step(delay_a)
#                     done = terminated or truncated

#                     # exp = {
#                     #     "states": torch.from_numpy(s).detach(),
#                     #     "actions": torch.tensor(delay_a).detach().view(-1),
#                     #     "probs": prob[a].detach().view(-1),
#                     #     "rewards": torch.tensor(r / 100.0).detach().view(-1),
#                     #     "states_prime": torch.from_numpy(s_prime).detach(),
#                     #     "dones": torch.tensor(0 if done else 1).detach().view(-1),
#                     #     "timesteps": torch.tensor(t).detach().view(-1),
#                     #     "a_lsts": torch.tensor(a_lst).detach().view(-1)
#                     # }
#                     exp = {
#                         "states": s.tolist(),
#                         "actions": [delay_a],
#                         "probs": [prob[a].item()],
#                         "rewards": [r / 100.0],
#                         "states_prime": s_prime.tolist(),
#                         "dones": [0 if done else 1],
#                         "timesteps": [t],
#                         "a_lsts": a_lst
#                     }
#                     memory.store(**exp)
#                     memory.set_hidden(h_in.detach())
#                     memory.score += r

#                     s = s_prime
#                     if done:
#                         break
#             conn.send(memory)
#     finally:
#         env.close()
#         conn.close()

# def parallel_process(config: Config, model_state) -> list[Memory]:
#     env_name, num_memos, num_actors = config.env_name, config.num_memos, config.num_actors

#     processes = []
#     parent_conns, child_conns = zip(*[mp.Pipe() for _ in range(num_memos)])
#     aggregated_data = []
#     num_batches = (num_memos + num_actors - 1) // num_actors
#     for batch_idx in range(num_batches):
#         start = batch_idx * num_actors
#         end = min(start + num_actors, num_memos)

#         processes.clear()

#         for i in range(start, end):
#             p = mp.Process(target=collect_data, args=(env_name, model_state, config, child_conns[i]))
#             processes.append(p)
#             p.start()

#         for i in range(start, end):
#             data = parent_conns[i].recv()
#             aggregated_data.append(data)

#         for p in processes:
#             p.join()
#             p.close()

#     return aggregated_data

def worker(env_name, config: Config, conn):
    env = gym.make(env_name)
    memory = Memory(config.T_horizon)
    done = False
    try:
        while True:
            cmd, data = conn.recv()
            if cmd == "reset":
                s, _ = env.reset()
                h_out = torch.zeros(config.h0).float()
                a_lst = [i % 2 for i in range(config.delay)]
                conn.send((s, a_lst, h_out))

            elif cmd == "step":
                h_in = h_out
                a, prob, h_out, _ = data
                prob = prob.view(-1)
                a = a.item()
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
                    "a_lsts": a_lst[:-1]
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
            {"params": actor.rnn.parameters()},
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
    learner = Learner(actor, optim_pred_model, optim_policy, optimizer, config)


    print()
    print(f"* Experiment: {config.experiment_name}\n")

    do_save = config.do_save
    do_train = config.do_train
    if do_save:
        log_dir, saved_folder, record_dir = config.log_dir, config.saved_folder, config.record_dir
        check_dir_exist(log_dir, saved_folder, record_dir)

        writer = SummaryWriter(log_dir)
        config_text = config.get_json()
        writer.add_text('Configuration', config_text)

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

        # print(f"> {'Training for policy...':<35}", end=" ")
        # ppo_log, avg_loss_str = learner.learn_policy(memory_list)
        # print(f"|| Avg Loss  : {avg_loss_str}")

        # print(f"> {'Training for predictive model...':<35}", end=" ")
        # pred_model_log, avg_loss_str = learner.learn_pred_model(memory_list)
        # print(f"|| Avg Loss  : {avg_loss_str}")

        if do_train:
            print(f"> {'Training...':<35}", end=" ")
            loss_log, avg_loss_str = learner.learn(memory_list)
            print(f"|| Avg Loss  : {avg_loss_str}")

        if do_save:
            print("-" * 65)
            do_record = ep % config.record_interval == 0 and do_save
            if do_record:
                record_path = recording_eval(config.env_name, model_state, record_dir, ep)
                print(f"> Recording is saved in \"{record_path}\"")

            saved_path = f"{saved_folder}/epoch_{ep}_{config.model_name}"
            torch.save(actor, saved_path)
            print(f"> Model is saved in \"{saved_path}\"")

            try:
                # log = merge_dict(pred_model_log, ppo_log)
                log = loss_log
                log["score"] = avg_score
                for k, v in log.items():
                    writer.add_scalar(k, v, ep)
            except KeyError as e:
                print(f"Error: {e}")
        print()
    print("=== Finished ===\n")

    if do_save:
        writer.close()