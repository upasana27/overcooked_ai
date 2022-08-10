from agent import OAIAgent
from behavioral_cloning import BehaviouralCloningPolicy, BehaviouralCloningAgent

import numpy as np

class BCSubtaskAdaptor(BehaviouralCloningAgent):
    def __init__(self, device, visual_obs_shape, agent_obs_shape, args):
        super(BehaviouralCloningAgent, self).__init__(device, visual_obs_shape, agent_obs_shape, args)
        self.subtask_adaptor = None # TODO
        self.player_idx = None
        self.trajectory = []

    def predict(self, obs, sample=True):
        obs = (*obs, self.curr_subtask) if self.use_subtasks else obs
        obs = [o.unsqueeze(dim=0).to(self.device) for o in obs]
        logits = self.forward(obs)
        if self.use_subtasks:
            action_logits, subtask_logits = logits
            # Update predicted subtask
            if Action.INDEX_TO_ACTION[action] == Action.INTERACT:
                adapted_subtask_logits = self.subtask_adaptor(subtask_logits)
                ps = th.zeros_like(adapted_subtask_logits.squeeze())
                ps[th.argmax(adapted_subtask_logits.detach().squeeze(), dim=-1)] = 1
                self.adapted_subtask_logits = ps.float()
        else:
            action_logits = logits
        return Categorical(logits=action_logits).sample() if sample else th.argmax(action_logits, dim=-1)

    def step(self, state, joint_action):
        if self.player_idx is None:
            raise ValueError('Player idx must be set before BCSubtaskAdaptor.step can be called')
        self.trajectory.append(joint_action)

    def reset(self):
        self.trajectory = []
        # Predicted subtask to perform next, stars as unknown
        unknown_task_id = th.tensor(Subtasks.SUBTASKS_TO_IDS['unknown']).to(self.device)
        self.curr_subtask = F.one_hot(unknown_task_id, num_classes=Subtasks.NUM_SUBTASKS)

class TypeBasedAdaptor(OAIAgent):
    def __init__(self, models, args):
        super(TypeBasedAdaptor, self).__init__()
        self.args = args
        self.models = models
        self.policy = np.random.choice(self.models)

    def predict(self, obs: th.Tensor) -> Tuple[int, Union[th.Tensor, None]]:
        """
        Given an observation return the index of the action and the agent state if the agent is recurrent.
        """
        return self.policy.predict(obs)

    def step(self, state, joint_action):
        if self.player_idx is None:
            raise ValueError('Player idx must be set before BCSubtaskAdaptor.step can be called')
        teammate_action = joint_action[self.player_idx]
        self.update_policy_dist(state, teammate_action)

    def reset(self):
        # NOTE this is for a reset between episodes. Create a new TypeBasedAdaptor if this is for a new human
        self.policy = self.select_policy()

    def update_policy_dist(self, state, action):
        # TODO updates the estimated probability of which model best resembles teammate
        pass

    def calculate_selfplay_table(self):
        # TODO calculate the score for each model playing with each other
        pass

    def select_policy(self):
        # TODO use distribution to find the most similar model to human,
        #      then select the most complementary model using the selfplay_table
        pass

    def save(self, path: str):
        """All models should be pretrained, so saving shouldn't be necessary"""
        pass

    def load(self, path: str):
        """All models should be pretrained and passed in already loaded, so loading shouldn't be necessary"""
        pass