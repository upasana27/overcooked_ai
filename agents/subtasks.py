import numpy as np

class Subtasks:
    SUBTASKS = ['get_onion_from_dispenser', 'get_onion_from_counter', 'put_onion_in_pot', 'put_onion_closer',
                'get_plate_from_dish_rack', 'get_plate_from_counter', 'put_plate_closer', 'get_soup',
                'get_soup_from_counter', 'put_soup_closer', 'serve_soup', 'unknown']
    NUM_SUBTASKS = len(SUBTASKS)
    SUBTASKS_TO_IDS = {s: i for i, s in enumerate(SUBTASKS)}
    IDS_TO_SUBTASKS = {v: k for k, v in SUBTASKS_TO_IDS.items()}

def facing(layout, player):
    '''Returns terrain type that the agent is facing'''
    x, y = np.array(player.position) + np.array(player.orientation)
    layout = [[t for t in row.strip("[]'")] for row in layout.split("', '")]
    return layout[y][x]

def calculate_completed_subtask(layout, prev_state, curr_state, p_idx):
    '''
    Find out which subtask has been completed between prev_state and curr_state for player with index p_idx
    :param layout: layout of the env
    :param prev_state: previous state
    :param curr_state: current state
    :param p_idx: player index
    :return: Completed subtask ID, or None if no subtask was completed
    '''
    prev_obj = prev_state.players[p_idx].held_object.name if prev_state.players[p_idx].held_object else None
    curr_obj = curr_state.players[p_idx].held_object.name if curr_state.players[p_idx].held_object else None
    tile_in_front = facing(layout, prev_state.players[p_idx])
    # Object held didn't change -- This interaction didn't actually transition to a new subtask
    if prev_obj == curr_obj:
        subtask = None
    # Pick up an onion
    elif prev_obj is None and curr_obj == 'onion':
        # Facing an onion dispenser
        if tile_in_front == 'O':
            subtask = 'get_onion_from_dispenser'
        # Facing a counter
        elif tile_in_front == 'X':
            subtask = 'get_onion_from_counter'
        else:
            raise ValueError(f'Unexpected transition. {prev_obj} -> {curr_obj} while facing {tile_in_front}')
    # Place an onion
    elif prev_obj == 'onion' and curr_obj is None:
        # Facing a pot
        if tile_in_front == 'P':
            subtask = 'put_onion_in_pot'
        # Facing a counter
        elif tile_in_front == 'X':
            subtask = 'put_onion_closer'
        else:
            raise ValueError(f'Unexpected transition. {prev_obj} -> {curr_obj} while facing {tile_in_front}')
    # Pick up a dish
    elif prev_obj is None and curr_obj == 'dish':
        # Facing a dish dispenser
        if tile_in_front == 'D':
            subtask = 'get_plate_from_dish_rack'
        # Facing a counter
        elif tile_in_front == 'X':
            subtask = 'get_plate_from_counter'
        else:
            raise ValueError(f'Unexpected transition. {prev_obj} -> {curr_obj} while facing {tile_in_front}')
    # Place a dish
    elif prev_obj == 'dish' and curr_obj is None:
        # Facing a counter
        if tile_in_front == 'X':
            subtask = 'put_plate_closer'
        else:
            raise ValueError(f'Unexpected transition. {prev_obj} -> {curr_obj} while facing {tile_in_front}')
    # Pick up soup from pot using plate
    elif prev_obj == 'dish' and curr_obj == 'soup':
        # Facing a counter
        if tile_in_front == 'P':
            subtask = 'get_soup'
        else:
            raise ValueError(f'Unexpected transition. {prev_obj} -> {curr_obj} while facing {tile_in_front}')
    # Pick up soup from counter
    elif prev_obj is None and curr_obj == 'soup':
        # Facing a counter
        if tile_in_front == 'X':
            subtask = 'get_soup_from_counter'
        else:
            raise ValueError(f'Unexpected transition. {prev_obj} -> {curr_obj} while facing {tile_in_front}')
    # Place soup
    elif prev_obj == 'soup' and curr_obj is None:
        # Facing a service station
        if tile_in_front == 'S':
            subtask = 'serve_soup'
        # Facing a counter
        elif tile_in_front == 'X':
            subtask = 'put_soup_closer'
        else:
            raise ValueError(f'Unexpected transition. {prev_obj} -> {curr_obj} while facing {tile_in_front}')
    else:
        raise ValueError(f'Unexpected transition. {prev_obj} -> {curr_obj}.')

    if subtask:
        subtask = Subtasks.SUBTASKS_TO_IDS[subtask]

    return subtask