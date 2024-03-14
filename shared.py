from util import PriorityQueue
from capture import GameState

import copy
import random
import logging

from contest.game import Actions
from capture import AgentRules


####################
# Shared functions #
####################

def get_legal_actions_own(agent, game_state: GameState, pos):
    """
    Our own version of get_legal_actions that makes a copy of pacman's
    config and overrides the pacman's position in order to know the legal
    actions in any given position (making it possible to be used in A*)
    """
    agent_state = game_state.get_agent_state(agent.index)

    conf = copy.deepcopy(agent_state.configuration)
    conf.pos = pos

    possible_actions = Actions.get_possible_actions(conf, game_state.data.layout.walls)
    return AgentRules.filter_for_allowed_actions(possible_actions)


def aStarSearch(agent, initial_position, food_position, game_state: GameState):
    """A* implementation from last assignment's"""
    # Queue of states and their corresponding action traces
    frontier = PriorityQueue()
    visited_states = set()  # List of visited nodes that we don't want to check again

    frontier.push((initial_position, [], 0), 0)

    while not frontier.isEmpty():
        # Get the first element in the queue (the one with least cost)
        pos, actions, cost_until_now = frontier.pop()

        if pos not in visited_states:  # Otherwise, don't visit it
            if pos == food_position:
                return pos, actions  # Return the action trace

            visited_states.add(pos)  # Add the state to the visited states

            legal_actions = get_legal_actions_own(agent, game_state, pos)
            successor_states = []
            (x, y) = pos
            for action in legal_actions:
                if action == 'North':
                    successor_states.append(((x, y + 1), action, 1))
                elif action == 'South':
                    successor_states.append(((x, y - 1), action, 1))
                elif action == 'East':
                    successor_states.append(((x + 1, y), action, 1))
                elif action == 'West':
                    successor_states.append(((x - 1, y), action, 1))
                # We disregard 'Stop'

            for (successor, action, stepCost) in successor_states:
                # We add to the queue:
                # - The child
                # - The trace of actions until this moment (and the action that generated the child)
                # - The cost until now plus the cost of the current step
                priority = cost_until_now + stepCost + agent.heuristic(successor, game_state)
                # Update replaces, pushes or does nothing depending on the priority for us
                frontier.update((successor, actions + [action], priority), priority)

    # If the queue is empty, the goal node doesn't exist
    return []


def in_our_field(agent, position, game_state):
    """
    Returns whether a position is in our field or not
    """
    (x, _) = position
    if agent.red:
        return x < int(game_state.data.layout.width / 2)  # This is horrible
    else:
        return x >= int(game_state.data.layout.width / 2)


def get_closest_point_in_our_field(agent, game_state):
    if agent.red:
        x = int(game_state.data.layout.width / 2) - 1  # This is horrible (x2)
    else:
        x = int(game_state.data.layout.width / 2)

    min_distance = 999999
    min_point = None
    for y in range(1, game_state.data.layout.height):
        if not game_state.data.layout.walls[x][y]:
            distance_to_point = agent.get_maze_distance(game_state.get_agent_position(agent.index), (x, y))
            if distance_to_point < min_distance:
                min_distance = distance_to_point
                min_point = (x, y)

    return min_point


def get_column_slice(col, game_state, offset=None):
    if offset is not None:
        col += offset

    valid_positions = []
    for y in range(1, game_state.data.layout.height):
        if not game_state.data.layout.walls[col][y]:
            valid_positions.append((col, y))

    return valid_positions


def avoid_enemy_collision(agent, current_pos, next_action, game_state):
    """
    Checks if an A* decision was to go towards an enemy, and attempts to override
    it if possible
    """
    (x, y) = current_pos
    if next_action == 'North':
        a_star_pos = (x, y + 1)
    elif next_action == 'South':
        a_star_pos = (x, y - 1)
    elif next_action == 'East':
        a_star_pos = (x + 1, y)
    elif next_action == 'West':
        a_star_pos = (x - 1, y)

    possible_actions_and_positions = []
    actions = game_state.get_legal_actions(agent.index)
    for action in actions:
        if action == 'North':
            possible_actions_and_positions.append((action, (x, y + 1)))
        elif action == 'South':
            possible_actions_and_positions.append((action, (x, y - 1)))
        elif action == 'East':
            possible_actions_and_positions.append((action, (x + 1, y)))
        elif action == 'West':
            possible_actions_and_positions.append((action, (x - 1, y)))


    for agent_index in agent.get_opponents(game_state):
        enemy_position = game_state.get_agent_position(agent_index)
        if (enemy_position is not None and not in_our_field(agent, enemy_position, game_state)):
            # Check if the action causes a collision with the enemy
            for (action, position) in possible_actions_and_positions:
                if agent.get_maze_distance(position, enemy_position) <= 1: # The agent can reach us in his next move if we move there too
                    possible_actions_and_positions.remove((action, position))

    if len(possible_actions_and_positions) > 0:
        # Check if A*'s position is within the valid ones. If so, return its action
        for action, position in possible_actions_and_positions:
            if a_star_pos == position:
                return action

        action, position = random.choice(possible_actions_and_positions)
        if agent.red:
            logging.info(f"Red: Attacker: A* made me rush towards an enemy, doing an alternative action: {action}")
        else:
            logging.info(f"Blue: Attacker: A* made me rush towards an enemy, doing an alternative action: {action}")
        return action
    
    # We could stop, but it's faster to just die next turn
    return next_action


def get_food_positions_enemy(agent, game_state: GameState):
    isFood = agent.get_food(game_state)
    food_state = [row for row in agent.get_food(game_state)]
    food_positions = []

    numRows = len(food_state[0])
    numColumns = len(food_state)

    for j in range(0, numRows - 1):
        for i in range(0, numColumns - 1):
            if isFood[i][j]:
                food_positions.append((i, j))

    # Also add the capsules as an additional objective
    food_positions.extend(agent.get_capsules(game_state))
    return food_positions


def get_food_positions_ours(self, game_state: GameState):
    isFood = self.get_food_you_are_defending(game_state)
    food_state = [row for row in self.get_food_you_are_defending(game_state)]
    food_positions = []

    numRows = len(food_state[0])
    numColumns = len(food_state)

    for j in range(0, numRows - 1):
        for i in range(0, numColumns - 1):
            if isFood[i][j]:
                food_positions.append((i, j))

    # Also add the capsules as an additional objective
    food_positions.extend(self.get_capsules_you_are_defending(game_state))
    return food_positions


def a_star_to_food_position(agent, game_state: GameState, food_positions_function, randomize=False,
                            initial_position_function=None):
    """
    Runs A* to the closest food position given by the function provided, returning
    the whole sequence of actions

    If randomize==True, go to one of the top 3 closest foods randomly
    """
    if initial_position_function is None:
        initial_position = game_state.get_agent_position(agent.index)
    else:
        initial_position = initial_position_function(agent, game_state)

    distances_array = [
        (agent.get_maze_distance(initial_position, food_position), food_position) for
        food_position in food_positions_function(agent, game_state)]
    distances_array = sorted(distances_array, key=lambda x: x[0])

    if randomize:
        (_, min_food_position) = random.choice(distances_array[:3])
    else:
        (_, min_food_position) = distances_array[0]

    destination, next_actions = aStarSearch(agent, game_state.get_agent_position(agent.index),
                                            min_food_position,
                                            game_state)

    return destination, next_actions


def get_our_position(agent, game_state: GameState):
    return game_state.get_agent_position(agent.index)


def get_my_initial_position(agent, game_state: GameState):
    return game_state.get_initial_agent_position(agent.index)


def get_initial_enemy_position(agent, game_state: GameState):
    return game_state.get_initial_agent_position(agent.get_opponents(game_state)[0])
