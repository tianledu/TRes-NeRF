import torch
import torch.nn.functional as F
from torch import nn


class CVAE(nn.Module):
    """Implementation of CVAE(Conditional Variational Auto-Encoder)"""

    def __init__(self, feature_size, y_input_size, latent_size):
        super(CVAE, self).__init__()

        self.fc_e1 = nn.Linear(feature_size + y_input_size, 64)
        self.fc_e2 = nn.Linear(64, 64)

        self.fc_mu = nn.Linear(64, latent_size)
        self.fc_log_std = nn.Linear(64, latent_size)

        self.fc_d1 = nn.Linear(latent_size + y_input_size, 64)
        self.fc_d2 = nn.Linear(64, 64)

        self.fc_embed = nn.Linear(64, 3)
        self.fc_res = nn.Linear(64, y_input_size)

    def encode(self, x, y):
        h1 = F.relu(self.fc_e1(torch.cat([x, y], dim=1)))  # concat features and labels
        h2 = F.relu(self.fc_e2(h1))
        mu = self.fc_mu(h2)
        log_std = self.fc_log_std(h2)
        return mu, log_std

    def decode(self, z, y):
        h3 = F.relu(self.fc_d1(torch.cat([z, y], dim=1)))  # concat latents and labels
        h4 = F.relu(self.fc_d2(h3))
        # embed = torch.sigmoid(self.fc_embed(h4))  # use sigmoid because the input image's pixel is between 0 and 1
        res = torch.tanh(self.fc_res(h4))  # use tanh because the input image's pixel is between -1 and 1
        embed = self.fc_embed(h4)
        res = self.fc_res(h4)
        return embed, res

    def reparametrize(self, mu, log_std):
        std = torch.exp(log_std)
        eps = torch.randn_like(std)  # simple from standard normal distribution
        z = mu + eps * std
        return z

    def forward(self, x, y):
        mu, log_std = self.encode(x, y)  # 获取编码器输出
        z = self.reparametrize(mu, log_std)  # 从正态分布中采样潜在编码z
        embed, res = self.decode(z, y)  # 获取解码器输出
        # print(mu.shape, log_std.shape, z.shape, embed.shape, res.shape)
        return embed, res, mu, log_std

    def loss_function(self, mu, log_std) -> torch.Tensor:
        kl_loss = -0.5 * (1 + 2 * log_std - mu.pow(2) - torch.exp(2 * log_std))
        kl_loss = torch.sum(kl_loss)  # 计算kl散度损失
        return kl_loss
