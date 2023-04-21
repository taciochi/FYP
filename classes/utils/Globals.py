from typing import Dict, List, Union

from ple import PLE
from torch import device
from torch.cuda import is_available

from classes.model.linear.LinearDQN import LinearDQN
from classes.model.convolutional.ConvDQN import ConvolutionalDQN
from classes.model.linear.LinearDuelingDQN import LinearDuelingDQN
from classes.model.convolutional.ConvDuelingDQN import ConvolutionalDuelingDQN


class Globals:
    DEVICE_TYPE: device = device('cuda:0' if is_available() else 'cpu')

    BRAIN_DIR_PATH: str = 'brains/'
    BRAIN_COPTER_DIR_PATH: str = BRAIN_DIR_PATH + 'copter/'
    BRAIN_FLAPPY_DIR_PATH: str = BRAIN_DIR_PATH + 'flappy/'

    OUTCOMES_DIR_PATH: str = 'outcomes/'
    OUTCOMES_FILE_NAME_START: str = OUTCOMES_DIR_PATH + 'outcomes_'

    BRAIN_TYPES: List[str] = ['convolutional_dueling_dqn', 'convolutional_dqn', 'linear_dueling_dqn', 'linear_dqn']
    IMG_SIZE: int = 64

    TRAINING_REWARD_VALUES: Dict[str, float] = {
        'negative': -8.0,
        'positive': 10.0,
        'tick': 0.0,
        'loss': -8,
        'win': 10.0
    }
    PLAYING_REWARD_VALUES: Dict[str, float] = {
        'negative': 0.0,
        'positive': 10.0002,
        'tick': 0.0,
        'loss': 0,
        'win': 10.0002
    }

    @staticmethod
    def get_brain(brain_type: str, number_of_actions: int) -> Union[ConvolutionalDQN, ConvolutionalDuelingDQN,
                                                                    LinearDQN, LinearDuelingDQN]:
        if 'linear' in brain_type:
            return LinearDuelingDQN(number_of_observations=7,
                                    number_of_actions=number_of_actions) if 'dueling' in brain_type else \
                LinearDQN(number_of_observations=7, number_of_actions=number_of_actions)
        return ConvolutionalDuelingDQN(in_channels=1, number_of_actions=number_of_actions, game_height=Globals.IMG_SIZE,
                                       game_width=Globals.IMG_SIZE) if 'dueling' in brain_type else \
            ConvolutionalDQN(in_channels=1, number_of_actions=number_of_actions, game_width=Globals.IMG_SIZE,
                             game_height=Globals.IMG_SIZE)
