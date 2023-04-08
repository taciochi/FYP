from typing import Dict, Tuple

from torch import device
from torch.cuda import is_available


class Globals:
    SEED_VAL: int = 3
    GAMMA: float = 0.9
    BATCH_SIZE: int = 75
    GAME_WIDTH: int = 256
    GAME_HEIGHT: int = 512
    OUT_CHANNELS: int = 32
    REPLAY_SIZE: int = 20
    MIN_EPSILON: float = 0.01
    INIT_EPSILON: float = 0.5
    MEMORY_SIZE: int = 1_000
    EPSILON_DECAY: float = 0.97
    COPTER_BRAIN_FILE_NAME: str = 'copter.pth'
    FLAPPY_BRAIN_FILE_NAME: str = 'flappy.pth'
    NUMBER_OF_EPISODES: int = 1_200
    BRAIN_DIR_PATH: str = 'brains/'
    BRAIN_PATH: str = BRAIN_DIR_PATH + 'brain_'
    IMG_SIZE: Tuple[int, int] = (84, 84)
    DEVICE_TYPE: device = device('cuda:0' if is_available() else 'cpu')
    TRAINING_REWARD_VALUES: Dict[str, float] = {
        'negative': -8.0,
        'positive': 10.0,
        'tick': 0.05,
        'loss': -8,
        'win': 10.0
    }
    PLAYING_REWARD_VALUES: Dict[str, float] = {
        'negative': 0.0,
        'positive': 10.0,
        'tick': 0.00,
        'loss': 0,
        'win': 10.0
    }
