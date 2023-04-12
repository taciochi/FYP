from torch import Tensor, tensor

from interfaces.data.game_state.GameStatePreprocessor import GameStatePreprocessor


class PixelCopterGameStatePreprocessor(GameStatePreprocessor):

    @staticmethod
    def extract_state_info(game_state: dict) -> Tensor:
        y_position: float = game_state['player_y']
        velocity: float = game_state['player_vel']
        distance_to_ceiling: float = game_state['player_dist_to_ceil']
        distance_to_floor: float = game_state['player_dist_to_floor']
        distance_to_gate: float = game_state['next_gate_dist_to_player']
        gate_block_top: float = game_state['next_gate_block_top']
        gate_block_bottom: float = game_state['next_gate_block_bottom']

        return tensor([y_position, velocity, distance_to_ceiling, distance_to_floor, distance_to_gate, gate_block_top,
                       gate_block_bottom], requires_grad=False)
