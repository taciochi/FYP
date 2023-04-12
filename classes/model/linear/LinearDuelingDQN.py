from torch import Tensor, no_grad
from random import random, randrange
from torch.nn import Linear, Dropout
from torch.nn.functional import relu

from interfaces.model.linear.LinearModel import LinearModel


class LinearDuelingDQN(LinearModel):

    def __init__(self, number_of_observations: int, number_of_actions: int):
        super(LinearDuelingDQN, self).__init__()

        self.fc1 = Linear(number_of_observations, 128)
        self.fc2 = Linear(128, 64)
        self.fc3 = Linear(64, 32)

        self.dropout = Dropout(p=0.15)

        self.fc1_adv = Linear(32, 16)
        self.fc1_val = Linear(32, 16)

        self.fc2_adv = Linear(16, number_of_actions)
        self.fc2_val = Linear(16, 1)

    def forward(self, x: Tensor) -> Tensor:
        latent: Tensor = relu(self.fc1(x))
        latent = relu(self.fc2(latent))
        latent = relu(self.fc3(latent))
        adv: Tensor = relu(self.fc1_adv(latent))
        adv = self.fc2_adv(self.dropout(adv))
        val: Tensor = relu(self.fc1_val(latent))
        val = self.fc2_val(self.dropout(val))
        q: Tensor = val + adv + adv.mean(1, keepdim=True)
        return q

    def get_action(self, x: Tensor, epsilon: float) -> int:
        if random() < epsilon:
            return randrange(self.fc2_adv.out_features)
        with no_grad():
            q_val = self.forward(x=x)
            return q_val.max(1)[1].item()
