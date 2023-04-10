from typing import Dict, Tuple

from torch import device
from torch.cuda import is_available


class Globals:
    SEED_VAL: int = 3
    GAMMA: float = 0.9
    BATCH_SIZE: int = 75
    GAME_HEIGHT: int = 512
    MIN_EPSILON: float = 0.01
    INIT_EPSILON: float = 0.5
    MEMORY_SIZE: int = 1_000
    EPSILON_DECAY: float = 0.97
    NUMBER_OF_EPISODES: int = 1_200
    BRAIN_DIR_PATH: str = 'brains/'
    DEVICE_TYPE: device = device('cuda:0' if is_available() else 'cpu')
    REWARD_VALUES: Dict[str, float] = {
        'negative': -8.0,
        'positive': 10.0,
        'tick': 0.05,
        'loss': -8,
        'win': 10.0
    }
