import numpy as np

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO


lstm_model = RecurrentPPO("MlpLstmPolicy", "CartPole-v1", verbose=1, n_steps=2048, batch_size=64)
lstm_model.learn(10000)

env = lstm_model.get_env()
mean_reward, std_reward = evaluate_policy(lstm_model, env, n_eval_episodes=20, warn=False)
print('--- LSTM MODEL: REWARD', mean_reward)


mean_reward = []

for i in range(20):
    obs = env.reset()
    # cell and hidden state of the LSTM
    lstm_states = None
    num_envs = 1
    # Episode start signals are used to reset the lstm states
    episode_starts = np.ones((num_envs,), dtype=bool)
    cum_reward = 0
    while True:
        action, lstm_states = lstm_model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        cum_reward += rewards
        episode_starts = dones
        if dones:
            break
    # env.render()
    mean_reward.append(cum_reward)

print('--- LSTM MODEL CORRECT: REWARD', np.mean(mean_reward))



mlp_model = PPO("MlpPolicy", "CartPole-v1", verbose=1, n_steps=2048, batch_size=64)
mlp_model.learn(10000)

env = mlp_model.get_env()
mean_reward, std_reward = evaluate_policy(mlp_model, env, n_eval_episodes=20, warn=False)
print('--- MLP MODEL: REWARD', mean_reward)

# model.save("ppo_recurrent")
# del model # remove to demonstrate saving and loading
#
# model = RecurrentPPO.load("ppo_recurrent")
#
