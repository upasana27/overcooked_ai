#!/bin/bash
#SBATCH --qos=blanca-kann
#SBATCH --time=4-00:00:00
#SBATCH --gres=gpu
#SBATCH --ntasks=4
#SBATCH --mem=40G
#SBATCH --output=oai.%j.out
source /curc/sw/anaconda3/latest
conda activate oai
python3 agents/rl_agents.py --base-dir /projects/star7023/oai --layout counter_circuit_o_1order --encoding-fn OAI_lossless --use-subtasks --exp-name lstm --horizon 1200
