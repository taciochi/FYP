from torch import Tensor, no_grad
from random import random, randrange
from torch.nn import Linear, Dropout
from torch.nn.functional import relu

from interfaces.model.linear.LinearModel import LinearModel


class LinearDQN(LinearModel):

    def __init__(self, number_of_observations: int, number_of_actions: int):
        super(LinearDQN, self).__init__()

        self.fc1 = Linear(number_of_observations, 128)
        self.fc2 = Linear(128, 64)
        self.fc3 = Linear(64, 32)

        self.dropout = Dropout(p=0.15)

        self.fc4 = Linear(32, 16)
        self.fc5 = Linear(16, number_of_actions)

    def forward(self, x: Tensor) -> Tensor:
        latent: Tensor = relu(self.fc1(x))
        latent = relu(self.fc2(latent))
        latent = relu(self.fc3(latent))
        latent = relu(self.fc4(latent))
        q: Tensor = self.fc5(self.dropout(latent))
        return q

    def get_action(self, x: Tensor, epsilon: float) -> int:
        if random() < epsilon:
            return randrange(self.fc5.out_features)
        with no_grad():
            q_val = self.forward(x=x)
            return q_val.max(1)[1].item()
