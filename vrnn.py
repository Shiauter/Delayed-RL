import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 定義 VRNN 模型
class VRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VRNN, self).__init__()
        # 編碼器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2 * latent_dim)  # 輸出 μ 和 log(σ^2)
        )
        # 解碼器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )
        # RNN 模組
        self.rnn = nn.GRU(input_dim + latent_dim, hidden_dim, batch_first=True)
        # 儲存隱藏狀態
        self.hidden_dim = hidden_dim

        self.phi_x = nn.Linear()
        self.phi_z = nn.Linear()

    def forward(self, x, h_prev):
        batch_size, seq_len, input_dim = x.size()
        h = h_prev

        recon_loss = 0
        kl_loss = 0
        outputs = []

        for t in range(seq_len):
            # 獲取當前步的輸入
            x_t = x[:, t, :]
            x_h = torch.cat([x_t, h.squeeze(0)], dim=-1)

            # 編碼器輸出 μ 和 log(σ^2)
            encoder_out = self.encoder(x_h)
            mu, log_var = torch.chunk(encoder_out, 2, dim=-1)
            std = torch.exp(0.5 * log_var)

            # 從潛在空間取樣 z_t
            z_t = mu + std * torch.randn_like(std)

            # 解碼器生成重建輸出
            x_recon = self.decoder(torch.cat([z_t, h.squeeze(0)], dim=-1))
            outputs.append(x_recon)

            # 計算損失
            recon_loss += torch.mean((x_t - x_recon) ** 2)
            kl_loss += -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp())

            # 更新 RNN 隱藏狀態
            rnn_input = torch.cat([x_t, z_t], dim=-1).unsqueeze(1)
            _, h = self.rnn(rnn_input, h)

        return torch.stack(outputs, dim=1), h, recon_loss / seq_len, kl_loss / seq_len

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_dim)

# PPO 策略網絡
class VRNNPPO(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim, latent_dim):
        super(VRNNPPO, self).__init__()
        self.vrnn = VRNN(input_dim, hidden_dim, latent_dim)

        # 策略頭（生成動作分佈）
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
        # 價值頭（生成價值函數估計）
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x, h):
        vrnn_output, h, recon_loss, kl_loss = self.vrnn(x, h)
        policy = self.policy_head(h.squeeze(0))
        value = self.value_head(h.squeeze(0))
        return policy, value, h, recon_loss, kl_loss

# 訓練主函數
def train_vrnn_ppo(env, model, optimizer, num_epochs=1000, gamma=0.99):
    for epoch in range(num_epochs):
        state = env.reset()
        h = model.vrnn.init_hidden(batch_size=1)

        rewards = []
        log_probs = []
        values = []
        recon_losses = []
        kl_losses = []

        done = False
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            policy, value, h, recon_loss, kl_loss = model(state_tensor, h)

            action = torch.multinomial(policy, num_samples=1).item()
            log_prob = torch.log(policy.squeeze(0)[action])
            next_state, reward, done, _ = env.step(action)

            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(value)
            recon_losses.append(recon_loss)
            kl_losses.append(kl_loss)

            state = next_state

        # 計算回報
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)

        # 計算 PPO 損失
        log_probs = torch.stack(log_probs)
        values = torch.cat(values).squeeze(-1)
        advantages = returns - values

        policy_loss = -(log_probs * advantages.detach()).mean()
        value_loss = advantages.pow(2).mean()
        recon_loss = torch.stack(recon_losses).mean()
        kl_loss = torch.stack(kl_losses).mean()

        loss = policy_loss + 0.5 * value_loss + recon_loss + kl_loss

        # 更新模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}, Loss: {loss.item()}, Policy Loss: {policy_loss.item()}, Value Loss: {value_loss.item()}")

# 環境初始化與訓練
input_dim = 4  # 假設環境狀態維度為 4
action_dim = 2  # 假設動作空間維度為 2
hidden_dim = 64
latent_dim = 16

env = ...  # 使用 OpenAI Gym 環境
model = VRNNPPO(input_dim, action_dim, hidden_dim, latent_dim)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

train_vrnn_ppo(env, model, optimizer)

###################
import torch
import torch.nn as nn

EPS = torch.finfo(torch.float).eps # numerical logs

class VRNN(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim):
        super().__init__()

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim

        self.phi_x = nn.Sequential()
        self.phi_z = nn.Sequential()

        self.enc = nn.Sequential()
        self.enc_mean = nn.Linear()
        self.enc_std = nn.Linear()

        self.dec = nn.Sequential()
        self.dec_mean = nn.Linear()
        self.dec_std = nn.Linear()

        self.prior = nn.Sequential()
        self.prior_mean = nn.Linear()
        self.prior_std = nn.Linear()

        self.rnn = nn.GRU()

    # Used for training
    def forward(self, x, h):
        seq_len = x.size(0)

        keys = ["mean", "std"]
        enc_outs = {k: [] for k in keys}
        dec_outs = {k: [] for k in keys}

        kld_loss, nll_loss = 0.0, 0.0
        h_t = h.clone()

        for t in range(seq_len):
            phi_x_t = self.phi_x(x[t])

            # encoder
            enc_t = self.enc(torch.cat([phi_x_t, h_t[-1]], dim=1))
            enc_mean_t, enc_std_t = self.enc_mean(enc_t), self.enc_std(enc_t)

            z_t = self._reparameterized_sample(enc_mean_t, enc_std_t)
            phi_z_t = self.phi_z(z_t)

            # prior
            prior_t = self.prior(h_t[-1])
            prior_mean_t, prior_std_t = self.prior_mean(prior_t), self.prior_std(prior_t)

            # decoder
            dec_t = self.dec(torch.cat([phi_z_t, h_t[-1]], dim=1))
            dec_mean_t, dec_std_t = self.dec_mean(dec_t), self.dec_std(dec_t)

            _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], dim=1).unsqueeze(0))

            kld_loss += self._kld_gauss(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t)
            nll_loss += self._nll_gauss(dec_mean_t, x[t])

            enc_out = {"mean": enc_mean_t, "std": enc_std_t}
            dec_out = {"mean": dec_mean_t, "std": dec_std_t}
            for k in keys:
                enc_outs[k].append(enc_out[k])
                dec_outs[k].append(dec_out[k])

        return kld_loss, nll_loss, enc_outs, dec_outs

    def sample(self, x, h, seq_len):
        pred_s = x.clone()
        h_t = h.clone()

        for t in range(seq_len):
            # prior
            prior_t = self.prior(h_t[-1])
            prior_mean_t, prior_std_t = self.prior_mean(prior_t), self.prior_std(prior_t)

            z_t = self._reparameterized_sample(prior_mean_t, prior_std_t)
            phi_z_t = self.phi_z(z_t)

            # decoder
            dec_t = self.dec(torch.cat([phi_z_t, h_t[-1]], dim=1))
            dec_mean_t, dec_std_t = self.dec_mean(dec_t), self.dec_std(dec_t)

            phi_x_t = self.phi_x(dec_mean_t)
            _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], dim=1).unsqueeze(0))

            pred_s = dec_mean_t.data
        return pred_s

    def _reparameterized_sample(self, mean, std):
        eps = torch.empty_like(std).normal_()
        return mean + eps * std

    def _kld_gauss(self, mean_1, std_1, mean_2, std_2):
        """Using std to compute KLD"""
        kld_element =  (2 * torch.log(std_2 + EPS) - 2 * torch.log(std_1 + EPS) +
            (std_1.pow(2) + (mean_1 - mean_2).pow(2)) /
            std_2.pow(2) - 1)
        return	0.5 * torch.sum(kld_element)

    def _nll_bernoulli(self, theta, x):
        return - torch.sum(x*torch.log(theta + EPS) + (1-x)*torch.log(1-theta-EPS))


    def _nll_gauss(self, mean, std, x):
        return torch.sum(torch.log(std + EPS) + torch.log(2*torch.pi)/2 + (x - mean).pow(2)/(2*std.pow(2)))