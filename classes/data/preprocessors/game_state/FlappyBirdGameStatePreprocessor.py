from torch import Tensor, tensor

from interfaces.data.game_state.GameStatePreprocessor import GameStatePreprocessor


class FlappyBirdGameStatePreprocessor(GameStatePreprocessor):

    @staticmethod
    def extract_state_info(game_state: dict) -> Tensor:
        y_position: float = game_state['player_y']
        velocity: float = game_state['player_vel']
        next_pipe_top_y: float = game_state['next_pipe_top_y']
        next_pipe_bottom_y: float = game_state['next_pipe_bottom_y']
        distance_to_next_pipe: float = game_state['next_pipe_dist_to_player']
        next_next_pipe_top_y: float = game_state['next_next_pipe_top_y']
        next_next_pipe_bottom_y: float = game_state['next_next_pipe_bottom_y']

        return tensor([y_position, velocity, next_pipe_top_y, next_pipe_bottom_y, distance_to_next_pipe,
                       next_next_pipe_top_y, next_next_pipe_bottom_y], requires_grad=False)
