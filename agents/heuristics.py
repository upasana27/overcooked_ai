from overcooked_dataset import Subtasks
def compute_position_heuristic(mdp, state):
    ''''
    Position Heuristic calculates the minimum number of 
    steps required to go from current agent location to the 
    start position to end position for each subtask:
        get_onion_from_dispenser: Player position to onion dispenser
        put_onion_in_pot: Closest onion dispenser to pot
        put_onion_closer: Player position to onion dispenser to closest counter available to both players
        get_onion_from_counter: Closest counter to nearest pot
        get_plate_from_dish_rack: Player position to dish dispenser
        get_soup: Closest dish dispenser to pot
        put_plate_closer: Player position to dish dispenser to closest counter available to both players
        get_plate_from_counter: Closest counter to pot
        put_soup_closer: Player position to pot to closest counter available to both players
        get_soup_from_counter: Closest counter to serving location
        serve_soup: Player position to closest pot to serving location
        unknown: Not computed
    Inputs:
        grid(OvercookedGridworld): instance of Gridworld created as example:
                                   OvercookedGridworld.from_layout_name(args.layout)
        state (OvercookedState): the current state for computing the heuristic
    Output:
        Dictionary with keys for each subtask 
    '''
    planner = MotionPlanner(mdp)
    free_counter_locations = mdp.find_free_counters_valid_for_both_players(state, planner)
    players_pos_or_list = state.players_pos_and_or
    players_pos_list = state.player_positions
    pos_heur = dict.fromkeys(Subtasks.SUBTASKS)
    for i,player in enumerate(state.players):
        # current player positions and orientations
        player_pos_or = players_pos_or_list[i]
        player_pos = players_pos_list[i]

        #calculate heuristics for onion class, find onion dispenser locations and pot locations
        onion_disp_loc = mdp.get_onion_dispenser_locations()
        pot_loc = mdp.get_pot_locations()
        # closest onion dispenser to current player location
        min_steps_disp, best_disp = planner.min_cost_to_feature(player_pos_or, onion_disp_loc, True)
        # closest pot location to the closest onion dispenser
        min_steps_pot = planner.min_cost_between_features([best_disp], pot_loc)
        # closest free counter to the closest onion dispenser
        min_steps_disp_to_counter, best_counter = planner.min_cost_between_features([best_disp], free_counter_locations, True)
        # closest pot location to closest free counter to put onion in
        min_steps_counter_to_pot = planner.min_cost_between_features([best_counter], pot_loc)
        pos_heur['get_onion_from_dispenser'] = min_steps_disp
        pos_heur['put_onion_in_pot'] = min_steps_pot
        if min_steps_disp_to_counter ==None:
            pos_heur['put_onion_closer'] = min_steps_disp 
        pos_heur['get_onion_from_counter'] = min_steps_counter_to_pot


        #calculate heuristics for dish class, find dish dispenser locations
        dish_disp_loc = mdp.get_dish_dispenser_locations()
        # closest dish dispenser to current player location
        min_steps_disp, best_disp = planner.min_cost_to_feature(player_pos_or, dish_disp_loc, True)
        # closest pot location to the closest dish dispenser
        min_steps_pot = planner.min_cost_between_features([best_disp], pot_loc)
        # closest free counter to the closest dish dispenser
        min_steps_disp_to_counter, best_counter = planner.min_cost_between_features([best_disp], free_counter_locations, True)
        # closest pot location to closest free counter to collect soup
        min_steps_counter_to_pot = planner.min_cost_between_features([best_counter], pot_loc)
        pos_heur['get_plate_from_dish_rack'] = min_steps_disp
        pos_heur['get_soup'] = min_steps_pot 
        pos_heur['put_plate_closer'] = min_steps_disp + min_steps_disp_to_counter
        pos_heur['get_plate_from_counter'] = min_steps_counter_to_pot

        #calculate heuristics for soup class, find serving locations
        serving_locations = mdp.get_serving_locations()
        # find closest pot to current player location
        min_steps_to_pot, best_pot_pos = planner.min_cost_to_feature(player_pos_or, pot_loc, True)
        # distance between closest pot and serving locations
        min_steps_to_serving = planner.min_cost_between_features([best_pot_pos], serving_locations)
        # closest free counter to the closest pot
        min_steps_pot_to_counter, best_counter = planner.min_cost_between_features([best_pot_pos], free_counter_locations, True)
        # closest seving location to closest free counter to serve soup
        min_steps_counter_to_serving = planner.min_cost_between_features([best_counter], serving_locations)
        pos_heur['serve_soup'] = min_steps_to_pot + min_steps_to_serving
        pos_heur['put_soup_closer'] = min_steps_pot_to_counter
        pos_heur['get_soup_from_counter'] = min_steps_counter_to_serving
    return pos_heur

def compute_history_heuristic(history):
    '''
    Considering history of our agent's actions
    should we consider actions of other agent also?
    Input: List of dictionaries for all agents
           Each key denotes subtasks and values
           are the number of each subtask
           performed by the agent
    Output: Dictionary with keys for each subtask 

    '''
    history_heur =  history[0]
    return history_heur
    

def compute_layout_heuristic(mdp,state):
    '''
    Layout Heuristic calculates the number of steps 
    required for independent vs teamwork task the 
    start position to end position by comparing 
    the steps required in the layout for each subtask:
        Independent Yasks:
        get_onion_from_dispenser, put_onion_in_pot: Best path between onion dispensers and pots
        get_plate_from_dish_rack, get_soup: Best path between dish dispensers and pots
        serve_soup: Best path from pots to serving locations
        Teamwork Tasks:
        put_onion_closer: Best path between onion dispensers and counters available to both players 
        get_onion_from_counter: Best path between closest counter and pots                
        put_plate_closer: Best path between dish dispensers and counters available to both players
        get_plate_from_counter: Best path between slosest counter and pots
        put_soup_closer: Best path between pots and counters available to both players
        get_soup_from_counter: Best path between closest counter and serving locations
        unknown: Not computed
    Inputs:
        grid(OvercookedGridworld): instance of Gridworld created as example:
                                   OvercookedGridworld.from_layout_name(args.layout)
        state (OvercookedState): the current state for computing the heuristic
    Output:
        Dictionary with keys for each subtask 
    '''
    planner = MotionPlanner(mdp)
    free_counter_locations = mdp.find_free_counters_valid_for_both_players(state, planner)
    players_pos_or_list = state.players_pos_and_or
    players_pos_list = state.player_positions
    layout_heur = dict.fromkeys(Subtasks.SUBTASKS)
    # calculate heuristics for onion class
    ### for independent task execution ###
    onion_disp_loc = self.mdp.get_onion_dispenser_locations()
    pot_loc = mdp.get_pot_locations()
    # distance between onion dispensers and pot locations
    min_steps_onion_indep = planner.min_cost_between_features(onion_disp_loc, pot_loc)
    ### for teamwork task execution  ###
    # distance between onion dispensers and free counters
    min_steps_onion_to_counter, best_counter  = planner.min_cost_between_features(onion_disp_loc, free_counter_locations, True)
    # distance between closest counter and pots
    min_steps_counter_to_pot = planner.min_cost_between_features([best_counter], pot_loc)

    layout_heur['get_onion_from_dispenser'] = min_steps_onion_indep
    layout_heur['put_onion_in_pot'] = min_steps_onion_indep
    layout_heur['put_onion_closer'] = min_steps_onion_to_counter
    layout_heur['get_onion_from_counter'] = min_steps_counter_to_pot

    # calculate heuristics for dish class
    ### for independent task execution ###
    dish_disp_loc = mdp.get_dish_dispenser_locations()
    pot_loc = mdp.get_pot_locations()
    # distance between dish dispensers and pots
    min_steps_dish_indep = planner.min_cost_between_features(dish_disp_loc, pot_loc)
    ### for teamwork task execution ###
    # distance between dish dispensers and free counters
    min_steps_disp_to_counter, best_counter  = planner.min_cost_between_features(dish_disp_loc, free_counter_locations, True)
    # distance between closest counter and pots
    min_steps_counter_to_pot = planner.min_cost_between_features([best_counter], pot_loc)

    layout_heur['get_plate_from_dish_rack'] = min_steps_dish_indep
    layout_heur['get_soup'] = min_steps_dish_indep
    layout_heur['put_plate_closer'] = min_steps_disp_to_counter
    layout_heur['get_plate_from_counter'] = min_steps_counter_to_pot

    # calculate heuristics for soup class
    ### for independent task execution ###
    serving_locations = mdp.get_serving_locations()
    pot_loc = mdp.get_pot_locations()
    # distance between pots and serving locations
    min_steps_soup_indep = planner.min_cost_between_features(pot_loc, serving_locations)
    ### for teamwork task execution ###
    # distance between pots and free counters
    min_steps_pot_to_counter, best_counter  = planner.min_cost_between_features(pot_loc, free_counter_locations, True)
    # distance between closest counter and serving locations
    min_steps_counter_to_serving = planner.min_cost_between_features([best_counter], serving_locations)

    layout_heur['serve_soup'] = min_steps_soup_indep
    layout_heur['put_soup_closer'] = min_steps_pot_to_counter
    layout_heur['get_soup_from_counter'] = min_steps_counter_to_serving
    return layout_heur   
    
   
        

