import json
import pandas as pd
import pygame
from pygame import K_UP, K_LEFT, K_RIGHT, K_DOWN, K_SPACE, K_w, K_a, K_s, K_d, K_RSHIFT
from pygame.locals import HWSURFACE, DOUBLEBUF, RESIZABLE, QUIT, VIDEORESIZE
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, Direction, Action
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.agents.agent import StayAgent, RandomAgent, AgentFromPolicy, GreedyHumanModel
# from overcooked_ai_py.planning.planners import MediumLevelPlanner
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
from overcooked_ai_py.mdp.layout_generator import LayoutGenerator
from overcooked_ai_py.utils import load_dict_from_file


no_counters_params = {
    'start_orientations': False,
    'wait_allowed': False,
    'counter_goals': [],
    'counter_drop': [],
    'counter_pickup': [],
    'same_motion_goals': True
}

valid_counters = [(5, 3)]
one_counter_params = {
    'start_orientations': False,
    'wait_allowed': False,
    'counter_goals': valid_counters,
    'counter_drop': valid_counters,
    'counter_pickup': [],
    'same_motion_goals': True
}


class App:
    """Class to run an Overcooked Gridworld game, leaving one of the players as fixed.
    Useful for debugging. Most of the code from http://pygametutorials.wikidot.com/tutorials-basic."""

    def __init__(self, layout_name, slowmo_rate):
        self._running = True
        self._display_surf = None
        self.layout_name = layout_name
        self.env = OvercookedEnv.from_mdp(OvercookedGridworld.from_layout_name(self.layout_name), horizon=1200)
        self.agent = None
        self.slowmo_rate = slowmo_rate
        # print("Human player index:", p_idx)
        self.fps = 30 // slowmo_rate
        self.score = 0
        self.curr_tick = 0
        self.trajectory = []
        self.joint_action = [None, None]

    def on_init(self):
        pygame.init()
        surface = StateVisualizer().render_state(self.env.state, grid=self.env.mdp.terrain_mtx)
        self.window = pygame.display.set_mode(surface.get_size(), HWSURFACE | DOUBLEBUF | RESIZABLE)
        self.window.blit(surface, (0, 0))
        pygame.display.flip()
        # pygame.display.set_mode((100, 100))

        # Adding pre-trained agent as teammate
        #self.agent.set_agent_index(self.agent_idx)
        #self.agent.set_mdp(self.env.mdp)
        self._running = True

    def on_event(self, event, pidx):
        if event.type == pygame.KEYDOWN:
            print(event)
            pressed_key = event.dict['key']
            action = None

            if pressed_key == K_UP:
                action = Direction.NORTH
            elif pressed_key == K_RIGHT:
                action = Direction.EAST
            elif pressed_key == K_DOWN:
                action = Direction.SOUTH
            elif pressed_key == K_LEFT:
                action = Direction.WEST
            elif pressed_key == K_SPACE:
                action = Action.INTERACT
            self.joint_action[pidx] = action

            # if action is not None:
            #     self.joint_action[0] = action
            # action = None
            #
            # if pressed_key == K_w:
            #     action = Direction.NORTH
            # elif pressed_key == K_d:
            #     action = Direction.EAST
            # elif pressed_key == K_s:
            #     action = Direction.SOUTH
            # elif pressed_key == K_a:
            #     action = Direction.WEST
            # elif pressed_key == K_SPACE:
            #     action = Action.INTERACT
            # if action is not None:
            #     self.joint_action[1] = action

        if event.type == pygame.QUIT:
            self._running = False

    def step_env(self, joint_action):
        prev_state = self.env.state
        new_state, reward, done, info = self.env.step(joint_action)

        # prev_state, joint_action, info = super(OvercookedPsiturk, self).apply_actions()

        # Log data to send to psiturk client
        curr_reward = sum(info['sparse_r_by_agent'])
        self.score += curr_reward
        transition = {
            "state" : json.dumps(prev_state.to_dict()),
            "joint_action" : json.dumps(joint_action),
            "reward" : curr_reward,
            "time_left" : max((1200 - self.curr_tick) / self.fps, 0),
            "score" : self.score,
            "time_elapsed" : self.curr_tick / self.fps,
            "cur_gameloop" : self.curr_tick,
            "layout" : json.dumps(self.env.mdp.terrain_mtx),
            "layout_name" : self.layout_name,
            "trial_id" : -1 # TODO this is just for testing self.trial_id,
            # "player_0_id" : self.players[0],
            # "player_1_id" : self.players[1],
            # "player_0_is_human" : self.players[0] in self.human_players,
            # "player_1_is_human" : self.players[1] in self.human_players
        }

        self.trajectory.append(transition)
        return done

    def on_loop(self):
        # for i in range(2):
        #     self.joint_action[i] = self.joint_action[i] or Action.STAY
        assert(all(self.joint_action))

        if all(self.joint_action):
            done = self.step_env(self.joint_action)
            self.joint_action = [None, None]
            self.curr_tick += 1

            if done:
                self._running = False


    def on_render(self, pidx=None):
        p0_action = Action.ACTION_TO_INDEX[self.joint_action[0]] if pidx == 1 else None
        surface = StateVisualizer().render_state(self.env.state, grid=self.env.mdp.terrain_mtx, pidx=pidx, hud_data={"timestep": self.curr_tick}, p0_action=p0_action)
        self.window = pygame.display.set_mode(surface.get_size(), HWSURFACE | DOUBLEBUF | RESIZABLE)
        self.window.blit(surface, (0, 0))
        pygame.display.flip()


    def on_cleanup(self):
        pygame.quit()

    def on_execute(self):
        if self.on_init() == False:
            self._running = False

        sleep_time = 1000 // self.fps
        while (self._running):
            self.joint_action = [None, None]
            for i in range(2):
                self.on_render(pidx=i)
                while self.joint_action[i] is None:
                    for event in pygame.event.get():
                        self.on_event(event, i)
                    pygame.event.pump()
            self.on_loop()

            # for event in pygame.event.get():
            #     self.on_event(event)
            # self.on_loop()
            # self.on_render()
            # pygame.event.pump()
            # pygame.time.wait(sleep_time)

        self.save_trajectory()

        self.on_cleanup()

    def save_trajectory(self):
        df = pd.DataFrame(self.trajectory)
        df.to_pickle('./data/test.pickle')


def setup_game(env_name, player_idx):
    # if run_type == "ppo":
    #     print("Seed", run_seed)
    #     agent, config = get_ppo_agent(run_dir, run_seed, best=True)
    # elif run_type == "pbt":
    #     run_path = "data/" + run_type + "_runs/" + run_dir + "/seed_{}".format(run_seed)
    #     config = load_dict_from_file(run_path + "/config.txt")
    #
    #     agent_path = run_path + '/agent' + str(agent_num) + "/best"
    #     agent = get_agent_from_saved_model(agent_path, config["sim_threads"])
    # elif run_type == "bc":
    #     agent, config = get_bc_agent_from_saved(run_dir)
    # else:
    #     raise ValueError("Unrecognized run type")
    agent = None

    # base_mdp = OvercookedGridworld.from_layout_name("cramped_room")
    # self.mlam = MediumLevelActionManager.from_pickle_or_compute(self.base_mdp, NO_COUNTERS_PARAMS, force_compute=True)
    # self.env = OvercookedEnv.from_mdp(self.base_mdp, **DEFAULT_ENV_PARAMS, info_level=0)
    # self.greedy_human_model_pair = AgentPair(GreedyHumanModel(self.mlam), GreedyHumanModel(self.mlam))
    # np.random.seed(0)
    # env = OvercookedEnv.from_mdp(OvercookedGridworld.from_layout_name(env_name), horizon=1200)
    # return env, agent, p_idx


if __name__ == "__main__":
    """
    Sample commands
    -> pbt
    python overcooked_interactive.py -t pbt -r pbt_simple -a 0 -s 8015
    ->
    python overcooked_interactive.py -t ppo -r ppo_sp_simple -s 386
    -> BC
    python overcooked_interactive.py -t bc -r simple_bc_test_seed4
    """
    parser = ArgumentParser()
    # parser.add_argument("-t", "--type", dest="type",
    #                     help="type of run, (i.e. pbt, bc, ppo, etc)", required=True)
    # parser.add_argument("-r", "--run_dir", dest="run",
    #                     help="name of run dir in data/*_runs/", required=True)
    parser.add_argument("-no_slowed", "--no_slowed_down", dest="slow",
                        help="Slow down time for human to simulate actual test time", action='store_false')
    parser.add_argument("-s", "--seed", dest="seed", required=False, default=0)
    parser.add_argument("-a", "--agent_num", dest="agent_num", default=0)
    parser.add_argument("-i", "--idx", dest="idx", default=0)

    args = parser.parse_args()
    # run_type, run_dir, slow_time, run_seed, agent_num, p_idx = args.type, args.run, bool(args.slow), int(
    #     args.seed), int(args.agent_num), int(args.idx)
    slow_time = False

    layout_name = 'tf_test_4'
    slowmo_rate = 8

    # env, agent, p_idx = setup_game(layout_name, p_idx=0)

    theApp = App(layout_name, slowmo_rate)
    print("Slowed time:", slow_time)
    theApp.on_execute()