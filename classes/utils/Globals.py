from typing import Dict, List

from torch import device
from torch.cuda import is_available


class Globals:
    DEVICE_TYPE: device = device('cuda:0' if is_available() else 'cpu')

    BRAIN_DIR_PATH: str = 'brains/'
    BRAIN_COPTER_DIR_PATH: str = BRAIN_DIR_PATH + 'copter/'
    BRAIN_FLAPPY_DIR_PATH: str = BRAIN_DIR_PATH + 'flappy/'
    OUTCOMES_DIR_PATH: str = 'outcomes/'
    OUTCOMES_FILE_NAME_START: str = OUTCOMES_DIR_PATH + 'outcomes_'
    BRAIN_TYPES: List[str] = ['linear_dueling_dqn', 'linear_dqn']
    # BRAIN_TYPES: List[str] = ['linear_dqn', 'linear_dueling_dqn', 'convolutional_dqn', 'convolutional_dueling_dqn']

    TRAINING_REWARD_VALUES: Dict[str, float] = {
        'negative': -8.0,
        'positive': 10.0,
        'tick': 1.0,
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
