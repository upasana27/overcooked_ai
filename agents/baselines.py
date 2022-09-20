from arguments import get_arguments
from rl_agents import SingleAgentTrainer, MultipleAgentsTrainer


class SelfPlay(MultipleAgentsTrainer):
    def __init__(self):
        super(SelfPlay, self).__init__(args, name='selfplay', num_agents=1, use_lstm=False)

        @



