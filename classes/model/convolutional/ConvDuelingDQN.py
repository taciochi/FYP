from random import random, randrange

from torch import no_grad
from torch import zeros, Tensor
from torch.nn.functional import relu
from torch.nn import Conv2d, Linear, BatchNorm2d, BatchNorm1d

from interfaces.model.convolutional.ConvolutionalModel import ConvolutionalModel


class ConvolutionalDuelingDQN(ConvolutionalModel):

    def __init__(self, in_channels: int, number_of_actions: int, game_width: int, game_height: int):
        super(ConvolutionalDuelingDQN, self).__init__()
        self.conv1 = Conv2d(in_channels, 64, kernel_size=3, stride=1)
        self.conv2 = Conv2d(64, 32, kernel_size=3, stride=1)
        self.conv3 = Conv2d(32, 16, kernel_size=3, stride=1)

        self.bn2d_conv1 = BatchNorm2d(num_features=64)
        self.bn2d_conv2 = BatchNorm2d(num_features=32)
        self.bn2d_conv3 = BatchNorm2d(num_features=16)
        self.bn1d_fc1_adv = BatchNorm1d(num_features=512)
        self.bn1d_fc1_val = BatchNorm1d(num_features=512)

        fc_input = self.get_conv_output_size(width=game_width, height=game_height)

        self.fc1_adv = Linear(fc_input, 512)
        self.fc1_val = Linear(fc_input, 512)
        self.fc2_adv = Linear(512, number_of_actions)
        self.fc2_val = Linear(512, 1)

    def forward(self, x: Tensor) -> Tensor:
        latent = self.bn2d_conv1(relu(self.conv1(x)))
        latent = self.bn2d_conv2(relu(self.conv2(latent)))
        latent = self.bn2d_conv3(relu(self.conv3(latent)))
        flat = latent.view(x.size(0), -1)
        adv = self.bn1d_fc1_adv(relu(self.fc1_adv(flat)))
        val = self.bn1d_fc1_val(relu(self.fc1_val(flat)))
        adv = self.fc2_adv(adv)
        val = self.fc2_val(val).expand(flat.size(0), self.fc2_adv.out_features)
        q = val + adv - adv.mean(1, keepdim=True)
        return q

    def get_action(self, x: Tensor, epsilon: float) -> int:
        if random() < epsilon:
            return randrange(self.fc2_adv.out_features)
        with no_grad():
            self.eval()
            q_val = self.forward(x=x)
            return q_val.max(1)[1].item()

    def get_conv_output_size(self, width: int, height: int) -> int:
        img = zeros((1, height, width), requires_grad=False)
        out = self.conv1(img)
        out = self.conv2(out)
        out = self.conv3(out)
        return out.view(1, -1).size(1)
