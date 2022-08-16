import pandas as pd
import numpy as np
from sklearn import preprocessing
from overcooked_ai_py.mdp.overcooked_mdp import Action
class PlasticAgent():
    '''
    create a Plastic Policy Agent
    -Input required-
        if static distribution used:
            List of models provided in csv format, with two columns- state
            and action taken by agent in that state, each row is a sample 
            from which we extract transition function
        else:
            List of models provided as state and action distribution
    -param models: List of models provided
    -param num_models: Number of models for creating Behavior Distribution
    -param tf_probs: Extract transition function from list of models
    -param BehaviorDist: Probability Distribution over all models
    -param eta: Value of Eta, default=0.2
    '''
    def __init__(self, num_models, models, is_static = False):
        self.models = models
        self.num_models = num_models
        self.is_static = is_static
        if is_static:
            self.tf_probs = self.extract_trans()
        self.behavior_dist= self.init_behavior_dist(self.num_models)
        print("initial distribution:", self.behavior_dist)
        self.eta = 0.1
    
    def extract_trans(self):
        # Initialize empty transition function for all models
        tf_probs = []
        for model in self.models:
            # Initialize empty transition function for this model
            counts = {}
            # iterate over all model rows
            for index, row in model.iterrows():
                if row.at['state'] not in counts:
                    # If this state did not exist, initialize key
                    counts[row.at['state']] = np.zeros(len(Action.ALL_ACTIONS))
                # If this state and action already existed, add to instances count
                # print( type(row.at['joint_action']))
                action_idx = int(row.at['joint_action'][1])
                counts[row.at['state']][action_idx] +=1
            
            tf_prob = {}
            for state,action_counts in counts.items():
                # for each state, sum instances over all actions and calculate probability of taking each action from that state
                tf_prob[state] = action_counts/np.sum(action_counts)
            tf_probs.append(tf_prob)
            print(type(tf_probs))
        return tf_probs
    
    def init_behavior_dist(self,n):
        # initialize initial Probability Distribution as uniform
        return [1/n]*n 

    def update_beliefs(self,state,action):
        for index, model in enumerate(self.models) :
            if is_static:
                tf = self.tf_probs[index]
                # Check if there is matching (state,action) pair
                action_idx = action
                try:
                    prob_from_tf = tf[state][action_idx] 
                except:
                    prob_from_tf = 0
            # calculate loss for model
            loss_model = 1 - prob_from_tf
            # Update Probability Distribution according to loss for that model
            self.behavior_dist[index] *= (1-self.eta*loss_model)
        # Normalize Probabiity Distribution , review this code
        self.behavior_dist = self.behavior_dist / np.sum(self.behavior_dist )
    
def main():
    models = []
    # Create list of models
    for i in range(1,4,1):
        model = pd.read_csv (("{}{}{}".format('./data/model_', i, '.csv')))
        models.append(model)

    agent = PlasticAgent(3, models, True)
    for index,row in models[1].iterrows():
        agent.update_beliefs(row.at['state'], int(row.at['joint_action'][1]))
    print("final distribution",agent.behavior_dist)
if __name__ == '__main__':
    main()


        