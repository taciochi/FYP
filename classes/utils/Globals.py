from typing import Dict, List

from torch import device
from torch.cuda import is_available


class Globals:
    DEVICE_TYPE: device = device('cuda:0' if is_available() else 'cpu')

    BRAIN_DIR_PATH: str = 'brains/'
    BRAIN_COPTER_DIR_PATH: str = BRAIN_DIR_PATH + 'copter/'
    BRAIN_FLAPPY_DIR_PATH: str = BRAIN_DIR_PATH + 'flappy/'
    BRAIN_TYPES: List[str] = ['linear_dqn']
    # BRAIN_TYPES: List[str] = ['linear_dqn', 'linear_dueling_dqn', 'convolutional_dqn', 'convolutional_dueling_dqn']

    GAME_WIDTH: int = 256
    GAME_HEIGHT: int = 512

    GAME_NAMES: List[str] = ['flappy', 'copter']

    TRAINING_REWARD_VALUES: Dict[str, float] = {
        'negative': -8.0,
        'positive': 10.0,
        'tick': 0.01,
        'loss': -8,
        'win': 10.0
    }
    TESTING_REWARD_VALUES: Dict[str, float] = {
        'negative': 0.0,
        'positive': 10.0,
        'tick': 0.0,
        'loss': 0,
        'win': 10.0
    }
