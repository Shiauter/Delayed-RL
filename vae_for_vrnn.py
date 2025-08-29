import torch
import torch.nn as nn
import torch.nn.functional as F
import math

EPS = 1e-6
LOG_2PI = math.log(2 * math.pi)

class VAE(nn.Module):
    def __init__(self, x_dim, z_dim, a_dim, h_dim, pred_s_source: str, nll_include_const: bool, set_std_to_1: bool, z_source: str):
        super().__init__()

        self.x_dim = x_dim
        self.z_dim = z_dim
        self.a_dim = a_dim
        self.h_dim = h_dim
        self.pred_s_source = pred_s_source
        self.nll_include_const = nll_include_const
        self.set_std_to_1 = set_std_to_1
        self.z_source = z_source

        self.phi_x = nn.Sequential(
            nn.Linear(x_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, x_dim),
            nn.ReLU()
        )
        self.phi_z = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, z_dim),
            nn.ReLU()
        )

        self.enc_net = nn.Sequential(
            nn.Linear(x_dim + h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU()
        )
        self.enc_mean = nn.Linear(h_dim, z_dim)
        self.enc_std = nn.Linear(h_dim, z_dim)

        self.prior_net = nn.Sequential(
            nn.Linear(a_dim + h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU()
        )
        self.prior_mean = nn.Linear(h_dim, z_dim)
        self.prior_std = nn.Linear(h_dim, z_dim)

        self.dec_net = nn.Sequential(
            nn.Linear(z_dim + a_dim + h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU()
        )
        self.dec_std = nn.Linear(h_dim, x_dim)
        self.dec_mean = nn.Linear(h_dim, x_dim)

    def enc(self, phi_x, h):
        return self._guassian_head(
            self.enc_net, self.enc_mean, self.enc_std,
            phi_x, h
        )

    def dec(self, phi_z, a, h):
        a = F.one_hot(a.long(), self.a_dim).view(*a.shape[:-1], self.a_dim).float()
        return self._guassian_head(
            self.dec_net, self.dec_mean, self.dec_std,
            phi_z, a, h
        )

    def prior(self, a, h):
        a = F.one_hot(a.long(), self.a_dim).view(*a.shape[:-1], self.a_dim).float()
        return self._guassian_head(
            self.prior_net, self.prior_mean, self.prior_std,
            a, h
        )

    def cal_loss(self, x, a, h):
        phi_x = self.phi_x(x) # truth state
        enc_mean, enc_std = self.enc(phi_x, h)
        z_t = self._reparameterized_sample(enc_mean, enc_std, "sampled")
        phi_z = self.phi_z(z_t)
        dec_mean, dec_std = self.dec(phi_z, a, h)

        prior_mean_t, prior_std_t = self.prior(a, h) # pred state

        kld_loss = self._kld_gauss(enc_mean, enc_std, prior_mean_t, prior_std_t)
        nll_loss = self._nll_gauss(dec_mean, dec_std, x)
        mse_loss = torch.pow(x - dec_mean, 2).sum(dim=-1)

        log = {
            "kld_loss": kld_loss,
            "nll_loss": nll_loss,
            "mse_loss": mse_loss,
            "dec_std": dec_std.mean(dim=-1)
        }
        log.update(self._eval_z_usage(x, a, h))

        return phi_x, phi_z, log

    def _eval_z_usage(self, x, a, h):
        with torch.no_grad():
            phi_x = self.phi_x(x) # truth state
            enc_mean, enc_std = self.enc(phi_x, h)
            z_post = self._reparameterized_sample(enc_mean, enc_std, self.z_source)
            phi_z_post = self.phi_z(z_post)
            dec_mean_post, dec_std_post = self.dec(phi_z_post, a, h)
            mse_post = ((x - dec_mean_post)**2).sum(-1).mean()
            nll_post = self._nll_gauss(dec_mean_post, dec_std_post, x).mean()

            prior_mean, prior_std = self.prior(a, h)
            z_prior = self._reparameterized_sample(prior_mean, prior_std, self.z_source)
            phi_z_prior = self.phi_z(z_prior)
            dec_mean_prior, dec_std_prior = self.dec(phi_z_prior, a, h)
            mse_prior = ((x - dec_mean_prior)**2).sum(-1).mean()
            nll_prior = self._nll_gauss(dec_mean_prior, dec_std_prior, x).mean()

            z_zero = torch.zeros_like(z_prior)
            phi_z_zero = self.phi_z(z_zero)
            dec_mean_zero, dec_std_zero = self.dec(phi_z_zero, a, h)
            mse_zero = ((x - dec_mean_zero)**2).sum(-1).mean()
            nll_zero = self._nll_gauss(dec_mean_zero, dec_std_zero, x).mean()

            # timestep shuffle
            idx = torch.randperm(z_post.size(1), device=z_post.device)
            z_shuf = z_post[:, idx, :]
            phi_z_shuf = self.phi_z(z_shuf)
            dec_mean_shuf, dec_std_shuf = self.dec(phi_z_shuf, a, h)
            mse_shuf = ((x - dec_mean_shuf)**2).sum(-1).mean()
            nll_shuf = self._nll_gauss(dec_mean_shuf, dec_std_shuf, x).mean()

            return {
                "delta_mse_prior": (mse_prior - mse_post),
                "delta_mse_zero": (mse_zero - mse_post),
                "delta_mse_shuf": (mse_shuf - mse_post),
                "delta_nll_prior": (nll_prior - nll_post),
                "delta_nll_zero": (nll_zero - nll_post),
                "delta_nll_shuf": (nll_shuf - nll_post),
            }

    def forward(self, s_t, h):
        # s = (batch, s_size)
        # h = (batch, hidden_size)

        phi_x = self.phi_x(s_t)
        enc_mean, enc_std = self.enc(phi_x, h)
        z_t = self._reparameterized_sample(enc_mean, enc_std, self.z_source)
        phi_z = self.phi_z(z_t)

        return phi_x, phi_z

    def rollout(self, a, h):
        # a = (batch, 1)
        # h = (batch, hidden_size)

        prior_mean, prior_std = self.prior(a, h)
        z_t = self._reparameterized_sample(prior_mean, prior_std, self.z_source)
        phi_z = self.phi_z(z_t)

        dec_mean, dec_std = self.dec(phi_z, a, h)
        if self.pred_s_source == "dec_mean_t": pred_s = dec_mean
        elif self.pred_s_source == "sampled_s": pred_s = self._reparameterized_sample(dec_mean, dec_std)
        phi_x = self.phi_x(pred_s)

        return phi_x, phi_z

    def _guassian_head(self, net, mean_layer, std_layer, *inputs):
        x = net(torch.cat(inputs, dim=-1))
        x_mean, x_std = mean_layer(x), std_layer(x)
        x_std = EPS + (1.0 - EPS) * torch.sigmoid(x_std) # limited within [EPS, 1.0]
        # x_std = F.softplus(x_std) + EPS
        if self.set_std_to_1:
            x_std = torch.ones_like(x_std)
        return x_mean, x_std

    def _reparameterized_sample(self, mean, std, source: str = "mean"):
        if source == "mean":
            return mean
        elif source == "sampled":
            eps = torch.randn_like(std)
            return mean + eps * std
        else:
            raise ValueError(f"Unknown z_source: {source}")

    def _kld_gauss(self, mean_1, std_1, mean_2, std_2):
        log_var_1 = 2 * torch.log(std_1 + EPS)
        log_var_2 = 2 * torch.log(std_2 + EPS)
        nume = std_1.pow(2) + (mean_1 - mean_2).pow(2)
        deno = std_2.pow(2) + EPS
        kld_element = (log_var_2 - log_var_1 + (nume / deno) - 1) * 0.5
        return	torch.sum(kld_element, dim=-1)

    def _nll_gauss(self, mean, std, x):
        var = std.pow(2) + EPS
        log_term = torch.log(var)
        if self.nll_include_const:
            log_term = log_term + LOG_2PI
        sqr_term = (x - mean).pow(2) / var
        nll_element = (log_term + sqr_term) * 0.5
        return torch.sum(nll_element, dim=-1)
