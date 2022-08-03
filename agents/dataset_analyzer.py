import json
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedState, OvercookedGridworld, Direction, Action
from pathlib import Path
import pandas as pd


layouts = 'all'
data_path = '../data/' # args.base_dir / args.data_path / args.dataset
filename = '2019_hh_trials_all.pickle'
main_trials = pd.read_pickle(Path(data_path) / filename)
if filename == '2019_hh_trials_all.pickle':
    main_trials.loc[main_trials.layout_name == 'random0', 'layout_name'] = 'forced_coordination'
    main_trials.loc[main_trials.layout_name == 'random3', 'layout_name'] = 'counter_circuit'
print(f'Number of all trials: {len(main_trials)}')
if layouts != 'all':
    main_trials = main_trials[main_trials['layout_name'].isin(layouts)]

# Remove all transitions where both players noop-ed
noop_trials = main_trials[main_trials['joint_action'] != '[[0, 0], [0, 0]]']
print(f'Number of {str(layouts)} trials without double noops: {len(noop_trials)}')
# print(main_trials['layout_name'])

action_ratios = {k: 0 for k in Action.ALL_ACTIONS}
all_noops, p1_noops, p2_noops, double_noops = 0, 0, 0, 0

def str_to_actions(joint_action):
    """
    Convert df cell format of a joint action to a joint action as a tuple of indices.
    Used to convert pickle files which are stored as strings into np.arrays
    """
    global all_noops, p1_noops, p2_noops, double_noops
    try:
        joint_action = json.loads(joint_action)
    except json.decoder.JSONDecodeError:
        # Hacky fix taken from https://github.com/HumanCompatibleAI/human_aware_rl/blob/master/human_aware_rl/human/data_processing_utils.py#L29
        joint_action = eval(joint_action)
    for i in range(2):
        if type(joint_action[i]) is list:
            joint_action[i] = tuple(joint_action[i])
        if type(joint_action[i]) is str:
            joint_action[i] = joint_action[i].lower()
        assert joint_action[i] in Action.ALL_ACTIONS
        action_ratios[joint_action[i]] += 1
        if joint_action[i] == Action.STAY:
            all_noops += 1
            if i == 0:
                p1_noops += 1
            elif i == 1:
                p2_noops += 1
    if joint_action[0] == Action.STAY and joint_action[1] == Action.STAY:
        double_noops += 1

def str_to_obss(df):
    """
    Convert from a df cell format of a state to an Overcooked State
    Used to convert pickle files which are stored as strings into overcooked states
    """
    state = df['state']
    if type(state) is str:
        state = json.loads(state)
    state = OvercookedState.from_dict(state)
    env = layout_to_env[df['layout_name']]
    visual_obs, agent_obs = encode_state_fn(env.mdp, state, grid_shape, args.horizon)
    df['state'] = state
    df['visual_obs'] = visual_obs
    df['agent_obs'] = agent_obs
    return df

main_trials['joint_action'] = main_trials['joint_action'].apply(str_to_actions)

print(f'p1 noops: {p1_noops} / {len(main_trials)} = {p1_noops / len(main_trials):.3f}%')
print(f'p2 noops: {p2_noops} / {len(main_trials)} = {p2_noops / len(main_trials):.3f}%')
print(f'all noops: {all_noops} / {2*len(main_trials)} = {all_noops / (2*len(main_trials)):.3f}%')
print(f'double noops: {double_noops} / {len(main_trials)} = {double_noops / len(main_trials):.3f}%')