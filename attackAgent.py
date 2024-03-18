import logging

from capture import GameState
from captureAgents import CaptureAgent

from shared import (aStarSearch,
                    in_our_field,
                    a_star_to_food_position,
                    avoid_enemy_collision,
                    get_closest_point_in_our_field,
                    get_food_positions_enemy,
                    get_our_position,
                    get_my_initial_position,
                    get_column_slice,
                    get_food_positions_ours,
                    get_initial_enemy_position)


# Attacker's heuristic configuration
DISTANCE_FROM_ENEMY_TO_FLEE = 6
COST_HEURISTIC_ENEMY_CLOSE = 5
FOOD_EATEN_TO_RETURN = 5
CAPSULE_EFFECT_DURATION = 35  # Less than the actual duration, to avoid pacman being too careless
TURNS_FLEEING_TOO_MUCH = 4
TURNS_HAS_TO_FLEE = 6


class AttackAgent(CaptureAgent):
    """
    Our Attack agent, exclusively based on A* and planning rules
    The overall strategy is as follows:
        - At the start of the game, randomly go to the top closest enemy foods 
          (the defenders should usually go to the closest one too, so this way we can "confuse" them)
        - When attacking:
            - If there are no enemies close, we go to the closest food or capsule
            - If there are enemies close, start going towards the closest point in our field

        As special behavior modifiers, the attacker will:
            - Ignore enemies when having a capsule for CAPSULE_EFFECT_DURATION turns (less than the real ones, to avoid being too reckless)
            - Go back to our field when having eaten FOOD_EATEN_TO_RETURN foods
                - While going back and there are no enemies close, try to eat foods within a distance 1 even
                  if A* told us to do another action
            - Avoid calculating routes within the enemy field that cross through enemy 
              surrounding areas (via A* heuristics)
            - When trying to go towards the enemy field, if the attacker has been fleeing for many consecutive turns, it will
              go to different close positions in our field and attack from there again, in order to confuse the defender 
              and avoid deadlocks

        There are also failsafes built on top of A*, so that:
            - If A* attempts to go towards a position that is reachable by an enemy in the next turn(s), go to another one
            - Although the agent is very efficient, we have made small improvements such as storing paths when going towards an initial position,
              with added failsafes when pacman dies
                
        All heuristic and misc. values that affect the behavior are configurable in the global variables above
    """
    logging.getLogger().setLevel(logging.INFO)

    # Food tracking counters
    food_in_last_turn = 0
    food_eaten = 0

    # Capsule tracking counters
    capsules_in_last_turn = 0
    turns_with_capsule_effect = 0

    # Sequence of actions to be executed at the start of the game (go towards an
    # initial position, with no changes in-between turns)
    first_actions = []
    already_randomized = False

    turn_counter = 0
    has_fled = []

    fleeing_point = None
    turns_has_to_flee = 0
    last_fled_turn_checked = 0

    def register_initial_state(self, game_state: GameState):
        CaptureAgent.register_initial_state(self, game_state)

        self.food_in_last_turn = len(get_food_positions_enemy(self, game_state))
        self.capsules_in_last_turn = len(self.get_capsules(game_state))
        self.turns_with_capsule_effect = 0

    def heuristic(self, pos, game_state: GameState):
        """
        Attacker's A* heuristic that forces the path cost be higher for positions
        within the FLEE distance of an enemy.

        Note: we could also do this via stepCost, but this way we keep A* generic
        """

        for agent_index in self.get_opponents(game_state):
            enemy_position = game_state.get_agent_position(agent_index)

            if enemy_position is not None:
                if not in_our_field(self, enemy_position, game_state):
                    distance_to_enemy = self.get_maze_distance(pos, enemy_position)
                    if distance_to_enemy < DISTANCE_FROM_ENEMY_TO_FLEE:
                        return COST_HEURISTIC_ENEMY_CLOSE

        return 1

    def update_counters(self, game_state):
        self.turn_counter += 1
        if self.turns_has_to_flee > 0:
            self.turns_has_to_flee -= 1

        if self.turns_with_capsule_effect > 0:
            self.turns_with_capsule_effect -= 1
            logging.info(f"Attacker: Capsule effect active, remaining turns: {self.turns_with_capsule_effect}")

        if self.capsules_in_last_turn > len(self.get_capsules(game_state)):
            self.turns_with_capsule_effect = CAPSULE_EFFECT_DURATION
            self.capsules_in_last_turn = len(self.get_capsules(game_state))
            logging.info(f"Attacker: Ate a capsule! remaining turns: {self.turns_with_capsule_effect}")

        if in_our_field(self, get_our_position(self, game_state), game_state):
            self.food_eaten = 0
        else:
            self.first_actions = []

        if self.already_randomized and get_our_position(self, game_state) == get_my_initial_position(self, game_state):
            logging.info(
                f"Attacker: I have been killed while going to my initial position or fleeing! Resetting actions...")
            self.first_actions = []
            self.turns_has_to_flee = 0

        food_in_current_turn = len(get_food_positions_enemy(self, game_state))
        if food_in_current_turn < self.food_in_last_turn:
            self.food_in_last_turn = food_in_current_turn
            self.food_eaten += 1
            logging.info(f"Attacker: Ate a food in last turn! Food eaten: {self.food_eaten}")

    def has_been_fleeing_too_much(self, game_state):
        if in_our_field(self, get_our_position(self, game_state), game_state):
            if len(self.has_fled) > 4:
                if self.has_fled[-1] != self.last_fled_turn_checked:
                    self.last_fled_turn_checked = self.has_fled[-1]
                    return (self.has_fled[-1] - self.has_fled[-2] <= TURNS_FLEEING_TOO_MUCH) and (
                                self.has_fled[-2] - self.has_fled[-3] <= TURNS_FLEEING_TOO_MUCH)

        return False

    def decide_action_return(self, game_state):
        """
        Return to the closest point in our field, deciding whether to flee or
        opportunistically grab extra food along the way
        """
        closest_point_in_our_field = get_closest_point_in_our_field(self, game_state)
        next_pos, next_actions = aStarSearch(self, get_our_position(self, game_state),
                                             closest_point_in_our_field,
                                             game_state)

        has_to_flee = False
        for agent_index in self.get_opponents(game_state):
            enemy_position = game_state.get_agent_position(agent_index)
            if enemy_position is not None:
                if self.get_maze_distance(get_our_position(self, game_state),
                                          enemy_position) < DISTANCE_FROM_ENEMY_TO_FLEE:
                    has_to_flee = True
                    break

        # If there are no enemies close and there is reachable food within one position, go for it even
        # if it is not in our path
        if not has_to_flee:
            for food_pos in get_food_positions_enemy(self, game_state):
                (x, y) = get_our_position(self, game_state)
                if self.get_maze_distance((x, y), food_pos) == 1:
                    logging.info(
                        f"Attacker: Ate {self.food_eaten} foods, returning to our field ({closest_point_in_our_field}). Doing a detour due to a nearby food ({food_pos}) with no enemies close")

                    if food_pos == (x, y + 1):
                        next_action = 'North'
                    elif food_pos == (x, y - 1):
                        next_action = 'South'
                    elif food_pos == (x + 1, y):
                        next_action = 'East'
                    elif food_pos == (x - 1, y):
                        next_action = 'West'

                    return next_action

        # else: there are enemies close or no nearby food was found, do not risk it
        logging.info(
            f"Attacker: Ate {self.food_eaten} foods, returning to our field ({closest_point_in_our_field}). Enemies close: {has_to_flee}")
        if len(next_actions) == 0:  # Only happens in the testCapture map
            return "Stop"
        return avoid_enemy_collision(self, get_our_position(self, game_state), next_actions[0], game_state)

    def decide_action_attack(self, game_state):
        """
        Go towards the closest enemy food. If it's the game start, instead of going towards the closest
        one go towards a random food within the top 3 closest ones (this may confuse the enemy defender
        if their heuristic is to defend the closest food from our starting position)
        """
        if len(self.first_actions) == 0 and not self.already_randomized:
            self.already_randomized = True
            _, self.first_actions = a_star_to_food_position(self, game_state,
                                                            get_food_positions_enemy,
                                                            randomize=True)
            logging.info("Attacker: Start of match, going to a random enemy food...")

        if len(self.first_actions) > 0:
            # It's the start of the game, and we still have a set of actions to follow towards
            # the initial food
            return avoid_enemy_collision(self, get_our_position(self, game_state),
                                         self.first_actions.pop(0),
                                         game_state)
        elif self.turns_has_to_flee > 0:
            # Continue going towards the fleeing point
            logging.info(f"Attacker: I have to keep fleeing towards a defender's patrol point for {self.turns_has_to_flee} turns, enemies close")
            if self.fleeing_point == get_our_position(self, game_state): # Go to a new one
                self.fleeing_point = self.get_fleeing_point(game_state)

            _, fleeing_actions = aStarSearch(self,
                                                      get_our_position(self, game_state),
                                                      self.fleeing_point,
                                                      game_state)
            return fleeing_actions[0]

        if self.has_been_fleeing_too_much(game_state):
            if len(get_food_positions_ours(self, game_state)) == 0:  # They have eaten all our foods
                return "Stop"

            self.fleeing_point = self.get_fleeing_point(game_state)

            self.turns_has_to_flee = TURNS_HAS_TO_FLEE

            logging.info(f"Attacker: I have been fleeing too much, going to random close enemy food ({self.fleeing_point})")

            _, fleeing_actions = aStarSearch(self,
                                                  get_our_position(self, game_state),
                                                  self.fleeing_point,
                                                  game_state)
            return fleeing_actions[0]
        else:
            dest, next_actions = a_star_to_food_position(self, game_state, get_food_positions_enemy, randomize=False)
            logging.info(f"Attacker: Going to closest enemy food ({dest})")

        return avoid_enemy_collision(self, get_our_position(self, game_state), next_actions[0], game_state)

    def get_fleeing_point(self, game_state):
        # Returns the highest or lowest position located in the column where
        # the closest food in our field from the enemy's start position perspective
        # is located in (whichever is the furthest)

        closest_food_from_enemy, _ = a_star_to_food_position(self,
                                                             game_state,
                                                             get_food_positions_ours,
                                                             initial_position_function=get_initial_enemy_position)
        (x, _) = closest_food_from_enemy
        valid_positions_in_column = get_column_slice(x, game_state)
        valid_positions_in_column = sorted(valid_positions_in_column, key=lambda x: x[1])
        patrol_point_1 = valid_positions_in_column[0]
        patrol_point_2 = valid_positions_in_column[-1]
        distance_to_pp_1 = self.get_maze_distance(get_our_position(self, game_state), patrol_point_1)
        distance_to_pp_2 = self.get_maze_distance(get_our_position(self, game_state), patrol_point_2)
        if distance_to_pp_1 > distance_to_pp_2:
            fleeing_point = patrol_point_1
        else:
            fleeing_point = patrol_point_2
        return fleeing_point

    def decide_action_in_enemy_field(self, game_state):
        # We don't have the capsule effect
        if self.turns_with_capsule_effect == 0:
            # Check if there are enemies close (visible)
            for agent_index in self.get_opponents(game_state):
                enemy_position = game_state.get_agent_position(agent_index)

                if enemy_position is not None:
                    if not in_our_field(self, enemy_position, game_state):
                        distance_to_enemy = self.get_maze_distance(get_our_position(self, game_state),
                                                                   enemy_position)
                        if distance_to_enemy < DISTANCE_FROM_ENEMY_TO_FLEE:
                            # Attempt to flee towards our field
                            flee_point = get_closest_point_in_our_field(self, game_state)

                            logging.info(
                                f"Attacker: Fleeing due to enemy within {distance_to_enemy} actions, going to {flee_point}")

                            self.has_fled.append(self.turn_counter)

                            next_pos, next_actions = aStarSearch(self,
                                                                 get_our_position(self, game_state),
                                                                 flee_point,
                                                                 game_state)

                            return avoid_enemy_collision(self,
                                                         get_our_position(self, game_state),
                                                         next_actions[0],
                                                         game_state)

        # Else: we are in the enemy field, but we still have the capsule effect or no enemies
        # close, so we don't care about them for now
        return self.decide_action_attack(game_state)

    def choose_action(self, game_state: GameState):
        # Update the counters on every turn
        self.update_counters(game_state)

        # Decide which course of action to follow
        if self.food_eaten > FOOD_EATEN_TO_RETURN or len(get_food_positions_enemy(self, game_state)) == 0:
            return self.decide_action_return(game_state)

        elif in_our_field(self, get_our_position(self, game_state), game_state):
            return self.decide_action_attack(game_state)

        else:
            return self.decide_action_in_enemy_field(game_state)
