#!/bin/bash
#SBATCH --qos=blanca-kann
#SBATCH --time=24:00:00
#SBATCH --gres=gpu
#SBATCH --ntasks=4
#SBATCH --mem=40G
#SBATCH --output=oai.%j.out
source /curc/sw/anaconda3/latest
conda activate oai
python3 agents/adaptable_agents.py --base-dir /projects/star7023/oai --layout tf_test_5_5 --encoding-fn OAI_lossless --use-subtasks --exp-name bc_1200_oai_lossless --horizon 1200
