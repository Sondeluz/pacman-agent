import logging
import random

from capture import GameState
from captureAgents import CaptureAgent

from shared import (aStarSearch,
                    in_our_field,
                    a_star_to_food_position,
                    avoid_enemy_collision,
                    get_closest_point_in_our_field,
                    get_food_positions_enemy,
                    get_our_position,
                    get_initial_enemy_position,
                    get_food_positions_ours,
                    get_legal_actions_own,
                    get_column_slice)

COST_HEURISTIC_CROSSING_ENEMY_FIELD = 5
CAPSULE_EFFECT_DURATION = 40
DISTANCE_FROM_ENEMY_TO_FLEE = 5  # If with capsule
COST_HEURISTIC_ENEMY_CLOSE = 5


class DefendAgent(CaptureAgent):
    """
    Our defend agent, exclusively based on A* and planning rules
    """
    logging.getLogger().setLevel(logging.INFO)

    # Sequence of actions to be executed at the start of the game (go towards an
    # initial position, with no changes in-between turns)
    initial_actions = []
    initial_goal = None

    last_patrolled_point = None
    patrol_point_1 = None
    patrol_point_2 = None

    # Food tracking counters
    food_positions_in_last_turn = None
    eaten_food_position = None

    # Capsule tracking counters
    capsules_in_last_turn = 0
    turns_with_capsule_effect = 0

    def heuristic(self, pos, game_state: GameState):
        """
        Defender's A* heuristic
        """
        if not in_our_field(self, pos, game_state):
            return COST_HEURISTIC_CROSSING_ENEMY_FIELD

        # for agent_index in self.get_opponents(game_state):
        #    enemy_position = game_state.get_agent_position(agent_index)

        #    if enemy_position is not None:
        #        if in_our_field(self, enemy_position, game_state) and self.capsules_in_last_turn > 0:
        #            distance_to_enemy = self.get_maze_distance(pos, enemy_position)
        #            if distance_to_enemy < DISTANCE_FROM_ENEMY_TO_FLEE:
        #                return COST_HEURISTIC_ENEMY_CLOSE

        return 1

    def register_initial_state(self, game_state: GameState):
        CaptureAgent.register_initial_state(self, game_state)
        self.food_positions_in_last_turn = set(get_food_positions_ours(self, game_state))
        self.capsules_in_last_turn = len(self.get_capsules_you_are_defending(game_state))
        self.turns_with_capsule_effect = 0

    def do_vertical_patrol(self, game_state):
        if self.patrol_point_1 is None and self.patrol_point_2 is None:  # Only happens on the testCapture map
            return "Stop"
        distance_to_pp_1 = self.get_maze_distance(get_our_position(self, game_state), self.patrol_point_1)
        distance_to_pp_2 = self.get_maze_distance(get_our_position(self, game_state), self.patrol_point_2)

        # It has visited none of them yet, choose the closest one
        if self.last_patrolled_point is None:
            if distance_to_pp_1 < distance_to_pp_2:
                goal = self.patrol_point_1
            else:
                goal = self.patrol_point_2

        # It has reached one of the patrol points, switch the patrol
        if distance_to_pp_1 == 0:
            self.last_patrolled_point = self.patrol_point_1
            goal = self.patrol_point_2

        elif distance_to_pp_2 == 0:
            self.last_patrolled_point = self.patrol_point_2
            goal = self.patrol_point_1

        # No patrol point reached yet, go to the patrol point we didn't visit previously
        elif self.last_patrolled_point == self.patrol_point_1:
            goal = self.patrol_point_2
        elif self.last_patrolled_point == self.patrol_point_2:
            goal = self.patrol_point_1

        logging.info(f"Defender: Patrolling to ({goal})")
        _, actions = aStarSearch(self, get_our_position(self, game_state), goal, game_state)
        return actions[0]

    def calculate_patrol_points(self, game_state):
        if len(get_food_positions_ours(self, game_state)) > 0:  # Else: It's the testCapture map
            self.initial_goal, _ = a_star_to_food_position(self, game_state, get_food_positions_ours,
                                                           initial_position_function=get_initial_enemy_position)
            logging.info(f"Defender: Start of match, going towards food at ({self.initial_goal}")
            _, actions = aStarSearch(self, get_our_position(self, game_state), self.initial_goal, game_state)
            self.initial_actions = actions

            # For debugging: start in a patrol point instead
            # _, actions = aStarSearch(self, get_our_position(self, game_state), (25, 2), game_state)
            # self.initial_actions = actions

            # Calculate the patrol points between the lowest and highest points of the column where
            # the closest food (from the enemy's initial position) is in
            (patrolling_column, _) = self.initial_goal
            valid_positions_in_column = get_column_slice(patrolling_column, game_state)
            valid_positions_in_column = sorted(valid_positions_in_column, key=lambda x: x[1])
            self.patrol_point_1 = valid_positions_in_column[0]
            self.patrol_point_2 = valid_positions_in_column[-1]

    def update_counters(self, game_state):
        if self.initial_goal is None:
            self.calculate_patrol_points(game_state)

        # Check if our food has been eaten
        food_positions_in_this_turn = set(get_food_positions_ours(self, game_state))
        foods_eaten = self.food_positions_in_last_turn - food_positions_in_this_turn
        self.food_positions_in_last_turn = food_positions_in_this_turn

        if len(foods_eaten) > 0:
            self.eaten_food_position = list(foods_eaten)[0]

        # Track capsule effect
        if self.capsules_in_last_turn > len(self.get_capsules_you_are_defending(game_state)):
            self.turns_with_capsule_effect = CAPSULE_EFFECT_DURATION
            self.capsules_in_last_turn = len(self.get_capsules_you_are_defending(game_state))
            logging.info(f"Defender: An attacker ate a capsule! remaining turns: {self.turns_with_capsule_effect}")

        elif self.turns_with_capsule_effect > 0:
            self.turns_with_capsule_effect -= 1
            logging.info(f"Defender: Capsule effect active, remaining turns: {self.turns_with_capsule_effect}")

    def decide_action_enemy_close_in_our_field(self, enemy_position, game_state):
        if self.turns_with_capsule_effect == 0:
            # Pursue it
            logging.info(f"Defender: Found an enemy at {enemy_position}, pursuing it")
            _, actions = aStarSearch(self, get_our_position(self, game_state), enemy_position,
                                     game_state)
            return actions[0]
        else:
            # The enemy has a capsule effect, but by the time we reach them they will have
            # run out of it, so we can pursue them
            if self.turns_with_capsule_effect < self.get_maze_distance(get_our_position(self, game_state),
                                                                       enemy_position):
                logging.info(
                    f"Defender: Found an enemy at {enemy_position} with capsule effect remaining for {self.turns_with_capsule_effect} turns but with distance {self.get_maze_distance(get_our_position(self, game_state), enemy_position)}, pursuing!")
                _, actions = aStarSearch(self, get_our_position(self, game_state),
                                         enemy_position, game_state)
                return actions[0]
            else:
                # Found enemies that can reach us while still having the capsule effect: It's dangerous
                # to pursue it, so go back to the patrol and hope to catch them when they return
                logging.info(
                    f"Defender: Found an enemy at {enemy_position} with capsule effect remaining for {self.turns_with_capsule_effect} turns but with distance {self.get_maze_distance(get_our_position(self, game_state), enemy_position)}, doing the patrol!")
                return self.do_vertical_patrol(game_state)

    def choose_action(self, game_state: GameState):
        self.update_counters(game_state)

        if len(self.initial_actions) != 0:
            # It's the start of the game, and we still have a set of actions to follow towards
            # the initial food
            return self.initial_actions.pop(0)
        else:
            # Check if there are visible enemies close
            for agent_index in self.get_opponents(game_state):
                enemy_position = game_state.get_agent_position(agent_index)

                if enemy_position is not None:
                    if in_our_field(self, enemy_position, game_state):
                        return self.decide_action_enemy_close_in_our_field(enemy_position, game_state)

            # No enemies close, check if the foods are disappearing (an enemy may be inside in an unknown location)
            if self.eaten_food_position is not None:  # They have eaten one of our foods!
                if self.eaten_food_position != get_our_position(self, game_state):
                    # Go towards it until we reach it, see an enemy or another food is eaten
                    logging.info(
                        f"Defender: They have eaten one of our foods, going towards it ({self.eaten_food_position})")
                    _, actions = aStarSearch(self, get_our_position(self, game_state), self.eaten_food_position,
                                             game_state)
                    return actions[0]
                else:
                    logging.info(f"Defender: Reached food position!")
                    self.eaten_food_position = None

            # Nothing is happening: patrol
            return self.do_vertical_patrol(game_state)
