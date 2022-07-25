import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim


class EncodeSym(nn.Module):
    def __init__(self,
                 input_dim, out_dim: int = 1,
                 a_cof: float = 0.1,
                 b_cof: float = 0.1,
                 c_cof: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        weight_a = torch.tensor(a_cof) * torch.randn(input_dim, out_dim)
        self.weight_a = nn.Parameter(weight_a)
        self.weight_a_sym = nn.Parameter(-weight_a)
        self.weight_b = nn.Parameter(torch.tensor(b_cof) * (torch.rand(1, input_dim) - torch.tensor(0.5)))
        self.weight_c = nn.Parameter(torch.tensor(c_cof) * torch.randn(1, input_dim))

    def forward(self, x):
        x = x.reshape(-1, self.input_dim)
        y1 = torch.mm(torch.exp(-torch.pow((x - self.weight_b)/self.weight_c, 2)), self.weight_a)
        y2 = torch.mm(torch.exp(-torch.pow((x - self.weight_b) / self.weight_c, 2)), self.weight_a_sym)
        # y = (self.weight_a.T * torch.exp(-torch.pow((x - self.weight_b) / self.weight_c, 2))).sum(dim=-1)
        return y1+y2


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, device, hidden_dim=[1024, 8096], gamma=0.99, lr=1e-4):
        super(Critic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lr = lr
        self.device = device

        self.dim_list = hidden_dim

        self.first_layer = nn.Linear(in_features=self.state_dim + self.action_dim, out_features=hidden_dim[0])
        self.sec_layer = nn.Linear(in_features=hidden_dim[0], out_features=hidden_dim[1])

        self.out_layer_en = EncodeSym(self.dim_list[-1], 1, 0.1, 1, 0.1)
        self.out_layer = nn.Linear(self.dim_list[-1], 1)

        self.activation = nn.ReLU()

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

        self.apply(self.weights_init_)

    def weights_init_(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=1)
            torch.nn.init.constant_(m.bias, 0)

    def forward(self, state, action):

        x = torch.cat([state, action], dim=-1)
        x = self.activation(self.first_layer(x))

        # for layer in self.layer_module:  # not include out layer
        #     x = (layer(x))
        x = self.activation(self.sec_layer(x))   # #
        x1 = self.out_layer(x)
        x2 = self.out_layer_en(x)

        return (x1+x2).reshape(-1, 1)
