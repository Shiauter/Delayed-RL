import torch
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

from actor_vrnn import Actor
from learner_vrnn import Learner
from util import Memory, merge_dict, check_dir_exist, action_data_sample
from config import Config
import pickle

def train(learner: Learner, memory_list: list[Memory]):
    for _ in range(learner.K_epoch_pred_model):
        total_pred_model_loss = []
        for i in range(learner.num_memos):
            s, a, r, s_prime, done, prob_a, a_lst = learner.make_batch(memory_list[i])
            first_hidden = memory_list[i].h0.detach().to(learner.device)

            kld_loss, nll_loss, o_ti, mse_loss = learner.cal_pred_model_loss(s, a_lst, first_hidden)
            total_pred_model_loss.append(kld_loss + nll_loss)

        total_pred_model_loss = torch.stack(total_pred_model_loss).mean()
        learner.optim_pred_model.zero_grad()
        total_pred_model_loss.backward()
        learner.optim_pred_model.step()

        print(f"total pred model loss: {total_pred_model_loss}")

with open("fixed_data_1_episode_407_score.pkl", "rb") as f:
    memory_list = pickle.load(f)
    print(len(memory_list), memory_list[0].score)

    config = Config(env_name="CartPole-v1")
    learner = Learner(config)

    print()
    print("=== Start ===\n")
    for ep in range(1, config.K_epoch_training + 1):
        print(f"> {'Training...':<35}")
        # loss_log, avg_loss_str = learner.separated_learning(memory_list)
        train(learner, memory_list)