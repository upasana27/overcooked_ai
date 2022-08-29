import argparse
from pathlib import Path


def get_arguments():
    """
    Arguments for training agents
    :return:
    """
    parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
    parser.add_argument('--layout-name', default='asymmetric_advantages',  help='Overcooked map to use')
    parser.add_argument('--use-subtasks', action='store_true', help='Condition IL agents on subtasks (default: False)')
    parser.add_argument('--policy-selection', type=str, default='CEM',
                        help='Which policy selection algorithm to use. Options: "CEM", "PLASTIC". Default: "CEM"')
    parser.add_argument('--subtask-selection', type=str, default='weighted',
                        help='Which subtask selection algorithm to use. Options: "weighted", "dist". Default: "dist"')
    parser.add_argument('--horizon', type=int, default=400, help='Max timesteps in a rollout')
    parser.add_argument('--encoding-fn', type=str, default='dense_lossless',
                        help='Encoding scheme to use. Options: "dense_lossless", "OAI_lossless", "OAI_feats"')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--batch-size', type=int, default=256, help='learning rate')
    parser.add_argument('--eta', type=float, default=0.1, help='eta used in plastic policy update.')
    parser.add_argument('--exp-name', type=str, default='default_exp',
                        help='Name of experiment. Used to tag save files.')
    parser.add_argument('--base-dir', type=str, default=Path.cwd(),
                        help='Base directory to save all models, data, tensorboard metrics.')
    parser.add_argument('--data-path', type=str, default='data',
                        help='Path from base_dir to where the expert data is stored')
    parser.add_argument('--dataset', type=str, default='2019_hh_trials_all.pickle',
                        help='Which set of expert data to use. '
                             'See https://github.com/HumanCompatibleAI/human_aware_rl/tree/master/human_aware_rl/static/human_data for options')
    parser.add_argument('--num_workers', type=int, default=4, metavar='N',
                        help='number of workers for pytorch train_dataloader (default: 4)')
    parser.add_argument('--wandb-mode', type=str, default='online',
                        help='Wandb mode. One of ["online", "offline", "disabled"')


    args = parser.parse_args()
    args.base_dir = Path(args.base_dir)
    return args
