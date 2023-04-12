from random import random, randrange

from torch import no_grad
from torch import zeros, Tensor
from torch.nn import Conv2d, Linear
from torch.nn.functional import relu

from interfaces.model.convolutional.ConvolutionalModel import ConvolutionalModel


class ConvolutionalDuelingDQN(ConvolutionalModel):

    def __init__(self, in_channels: int, number_of_actions: int, game_width: int, game_height: int):
        super(ConvolutionalDuelingDQN, self).__init__()
        self.conv1 = Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = Conv2d(64, 64, kernel_size=3, stride=1)

        fc_input = self.get_conv_output_size(width=game_width, height=game_height)

        self.fc1_adv = Linear(fc_input, 512)
        self.fc1_val = Linear(fc_input, 512)
        self.fc2_adv = Linear(512, number_of_actions)
        self.fc2_val = Linear(512, 1)

    def forward(self, x) -> Tensor:
        latent = relu(self.conv1(x))
        latent = relu(self.conv2(latent))
        latent = relu(self.conv3(latent))
        flat = latent.view(x.size(0), -1)
        adv = relu(self.fc1_adv(flat))
        val = relu(self.fc1_val(flat))
        adv = self.fc2_adv(adv)
        val = self.fc2_val(val).expand(flat.size(0), self.fc2_adv.out_features)
        q = val + adv - adv.mean(1, keepdim=True)
        return q

    def get_action(self, x: Tensor, epsilon: float) -> int:
        if random() < epsilon:
            return randrange(self.fc2_adv.out_features)
        with no_grad():
            q_val = self.forward(x=x)
            return q_val.max(1)[1].item()

    def get_conv_output_size(self, width: int, height: int) -> int:
        img = zeros((1, height, width), requires_grad=False)
        out = self.conv1(img)
        out = self.conv2(out)
        out = self.conv3(out)
        return out.view(1, -1).size(1)
