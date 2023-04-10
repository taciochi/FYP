from typing import Dict, Tuple

from torch import device
from torch.cuda import is_available


class Globals:
    BRAIN_DIR_PATH: str = 'brains/'
    DEVICE_TYPE: device = device('cuda:0' if is_available() else 'cpu')
    REWARD_VALUES: Dict[str, float] = {
        'negative': -8.0,
        'positive': 10.0,
        'tick': 0.05,
        'loss': -8,
        'win': 10.0
    }
