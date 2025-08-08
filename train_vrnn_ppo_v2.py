import gymnasium as gym
import torch
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

from actor_vrnn_v2 import Actor
from learner_vrnn_v2 import Learner
from util import Memory, merge_dict, check_dir_exist, action_data_sample
from config import Config


# region eval
def recording_eval(config: Config, model_state, record_dir:str, epoch:int):
    with torch.no_grad():
        env = gym.make(config.env_name, render_mode="rgb_array")
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


# region training loop
def worker(env_name, config: Config, conn):
    env = gym.make(env_name)
    memory = Memory(config.T_horizon)
    done = False
    try:
        while True:
            cmd, data = conn.recv()
            if cmd == "reset":
                s, _ = env.reset(seed=config.env_seed)
                h_out = torch.zeros(config.h0).float()
                copied_actions = action_data_sample[config.delay:]
                a_lst = [i % 2 for i in range(config.delay)]
                done = False
                conn.send((s, a_lst, h_out, done))

            elif cmd == "step":
                h_in = h_out
                a, prob, h_out, _ = data
                prob = prob.view(-1)
                a = a.item()
                # a = 0 if len(copied_actions) == 0 else copied_actions[0][0]
                a_lst.append(a)

                delay_a = a_lst[0]
                s_prime, r, terminated, truncated, _ = env.step(delay_a)
                done = terminated or truncated
                exp = {
                    "states": s.tolist(),
                    "actions": [delay_a],
                    "probs": [prob[a].item()],
                    "rewards": [r / config.reward_scale],
                    "states_prime": s_prime.tolist(),
                    "dones": [0 if done else 1],
                    "a_lsts": a_lst[:-1]
                }
                memory.store(**exp)
                memory.set_hidden(h_in.detach())
                memory.score += r

                s = s_prime
                a_lst.pop(0)
                if len(copied_actions) > 0:
                    copied_actions.pop(0)
                conn.send((s, a_lst, h_out, done)) # check h_in or h_out
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
                    s, a_lst, h_in, done = obs
                    a, prob, h_out, _ = actor.sample_action(
                        torch.from_numpy(s).view(1, 1, -1), # (seq_len, batch, s_size)
                        torch.tensor(a_lst).view(1, 1, -1),
                        h_in
                    )
                    actions.append((a.detach(), prob.detach(), h_out.detach(), done))
                else:
                    actions.append(None)

            for conn, action, active in zip(parent_conns, actions, active_envs):
                if active and action is not None:
                    conn.send(("step", action))

            for i, conn in enumerate(parent_conns):
                if active_envs[i]:
                    s, a_lst, h_in, done = conn.recv()
                    active_envs[i] = False if done else True
                    observations[i] = (s, a_lst, h_in, done)
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
# endregion


# region main
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    config = Config(env_name="CartPole-v1")
    actor = Actor(config)
    learner = Learner(config)
    prev_loss_log = None


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
        print(f"> {'Collecting data...':<30}", end=" ")

        memory_list = event_loop(config, actor)
        total_score = sum([memo.score for memo in memory_list])
        avg_score = total_score / config.num_memos
        print(f"|| Avg score : {avg_score:.1f}")

        model_state = actor.output_params()
        if do_train:
            print(f"> {'Training...':<30}", end=" ")
            learner.actor.load_params(model_state)
            loss_log, avg_loss_str = learner.learn(memory_list, ep)
            print(f"|| Avg Loss  : {avg_loss_str}")
            pred_model_param_tier, policy_param_tier = learner.adjust_learning_params(loss_log, prev_loss_log, ep)
            prev_loss_log = loss_log
            actor.load_params(learner.actor.output_params())

        if do_save:
            print("-" * 65)
            do_record = ep % config.record_interval == 0 and do_save
            if do_record:
                record_path = recording_eval(config, model_state, record_dir, ep)
                print(f"> Recording is saved in \"{record_path}\"")

            saved_path = f"{saved_folder}/epoch_{ep}_{config.model_name}"
            torch.save(model_state, saved_path)
            print(f"> Model is saved in \"{saved_path}\"")

            try:
                # log = merge_dict(pred_model_log, ppo_log)
                log = loss_log
                log["score"] = avg_score
                log["pred_model_param_tier"] = pred_model_param_tier
                log["policy_param_tier"] = policy_param_tier
                for k, v in log.items():
                    if v is not None:
                        writer.add_scalar(k, v, ep)
            except KeyError as e:
                print(f"Error: {e}")
        print()
    print("=== Finished ===\n")

    if do_save:
        writer.close()
# endregion