from torch import Tensor, no_grad
from torch.nn.functional import relu
from random import random, randrange
from torch.nn import Linear, Dropout, BatchNorm1d

from interfaces.model.linear.LinearModel import LinearModel


class LinearDuelingDQN(LinearModel):

    def __init__(self, number_of_observations: int, number_of_actions: int):
        super(LinearDuelingDQN, self).__init__()

        self.fc1 = Linear(number_of_observations, 128)
        self.fc2 = Linear(128, 64)
        self.fc3 = Linear(64, 64)

        self.dropout = Dropout(p=0.2)
        self.bn1d_x = BatchNorm1d(num_features=number_of_observations)
        self.bn1d_fc1 = BatchNorm1d(num_features=128)
        self.bn1d_fc2 = BatchNorm1d(num_features=64)
        self.bn1d_fc3 = BatchNorm1d(num_features=64)
        self.bn1d_adv_fc1 = BatchNorm1d(num_features=32)
        self.bn1d_val_fc1 = BatchNorm1d(num_features=32)

        self.fc1_adv = Linear(64, 32)
        self.fc1_val = Linear(64, 32)

        self.fc2_adv = Linear(32, number_of_actions)
        self.fc2_val = Linear(32, 1)

    def forward(self, x: Tensor) -> Tensor:
        latent: Tensor = self.bn1d_fc1(relu(self.fc1(self.bn1d_x(x))))
        latent = self.bn1d_fc2(relu(self.fc2(latent)))
        latent = self.bn1d_fc3(relu(self.fc3(latent)))

        adv: Tensor = self.bn1d_adv_fc1(relu(self.fc1_adv(latent)))
        adv = self.fc2_adv(self.dropout(adv))

        val: Tensor = self.bn1d_val_fc1(relu(self.fc1_val(latent)))
        val = self.fc2_val(self.dropout(val))

        q: Tensor = val + adv + adv.mean(1, keepdim=True)
        return q

    def get_action(self, x: Tensor, epsilon: float) -> int:
        if random() < epsilon:
            return randrange(self.fc2_adv.out_features)
        with no_grad():
            self.eval()
            q_val = self.forward(x=x)
            return q_val.max(1)[1].item()
