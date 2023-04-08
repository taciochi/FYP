from torch import zeros, Tensor
from torch.nn import Conv2d, Linear
from torch.nn.functional import relu

from classes.Globals import Globals
from interfaces.model.Model import Model


class DuelingDQN(Model):
    def __init__(self, in_channels, num_actions):
        super(DuelingDQN, self).__init__()

        self.conv1 = Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = Conv2d(64, 64, kernel_size=3, stride=1)

        fc_input = self.get_conv_output_size()

        self.fc1_adv = Linear(fc_input, 512)
        self.fc1_val = Linear(fc_input, 512)
        self.fc2_adv = Linear(512, num_actions)
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
        q = val + adv - adv.mean(1, keepdim=True).expand(latent.size(0), self.fc2_adv.out_features)
        return q

    def get_conv_output_size(self) -> int:
        img = zeros((1, Globals.IMG_SIZE[0], Globals.IMG_SIZE[1]), requires_grad=False)
        out = self.conv1(img)
        out = self.conv2(out)
        out = self.conv3(out)
        return out.view(1, -1).size(1)