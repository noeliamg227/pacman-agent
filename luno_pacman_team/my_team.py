import random
import util
from capture_agents import CaptureAgent
from game import Directions
from util import nearest_point
import heapq
#################
# Team creation #
#################


def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
    """
    Creates a team with two agents. The agents are defined by the parameters
    'first' and 'second'. Additionally, 'first_index' and 'second_index' are their respective indexes.
    """
    return [eval(first)(first_index), eval(second)(second_index)]




def a_star_search(game_state, start, goals, enemies):
    """
    Implements A* algorithm to find the safest path to a goal while avoiding enemies.
    :param game_state: Current game state.
    :param start: Agent's starting position.
    :param goals: List of goal positions (food or base).
    :param enemies: List of enemy positions.
    :return: First action of the optimal path or None if no path exists.
    """
    frontier = []
    heapq.heappush(frontier, (0, start, []))  # (Accumulated cost, current position, path)
    visited = set()

    while frontier:
        cost, current, path = heapq.heappop(frontier)

        if current in visited:
            continue
        visited.add(current)

        # If a goal is reached, return the first move of the path
        if current in goals:
            return path[0] if path else None

        # Expand neighbors
        for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            next_pos = get_next_position(current, direction)

            if game_state.has_wall(int(next_pos[0]), int(next_pos[1])) or next_pos in visited:
                continue

            # Penalty if an enemy is nearby
            enemy_penalty = 1000 if any(util.manhattan_distance(next_pos, e) <= 1 for e in enemies) else 0
            new_cost = cost + 1 + enemy_penalty
            heapq.heappush(frontier, (new_cost, next_pos, path + [direction]))

    return None  # No safe path found

def get_next_position(position, direction):
    """
    Computes the new position given a direction.
    """
    x, y = position
    if direction == Directions.NORTH:
        return (x, y + 1)
    elif direction == Directions.SOUTH:
        return (x, y - 1)
    elif direction == Directions.EAST:
        return (x + 1, y)
    elif direction == Directions.WEST:
        return (x - 1, y)
    return position


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    Base class for reflex agents that choose actions to maximize their score.
    """
    def register_initial_state(self, game_state):
        # Store the agent's starting position
        self.start = game_state.get_agent_position(self.index)
        # Register the agent's initial state using the parent class method
        super().register_initial_state(game_state)

    def choose_action(self, game_state):
        # Get the legal actions available for the agent
        actions = game_state.get_legal_actions(self.index)
        # Evaluate each action and compute its value
        values = [self.evaluate(game_state, a) for a in actions]
        # Find the maximum value among all actions
        max_value = max(values)
        # Choose the actions that have the maximum value
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        # Get the number of food left on the board
        food_left = len(self.get_food(game_state).as_list())
        # Return a random choice among the best actions
        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        # Generate the successor state after taking the given action
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        # If the position is not a grid point, call the function again to ensure it is aligned with the grid
        if pos != nearest_point(pos):
            return successor.generate_successor(self.index, action)
        return successor

    def evaluate(self, game_state, action):
        # Evaluate the action by calculating a weighted sum of features
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        # Default feature set for reflex agents
        return util.Counter({'successor_score': self.get_score(game_state)})

    def get_weights(self, game_state, action):
        # Default weights for reflex agent
        return {'successor_score': 1.0}

class OffensiveReflexAgent(ReflexCaptureAgent):
    """
    An agent that focuses on collecting food while avoiding enemies.
    """
    def get_features(self, game_state, action):
        features = util.Counter()  # Initialize a dictionary to store features
        successor = self.get_successor(game_state, action)  # Get the successor state after the action
        food_list = self.get_food(successor).as_list()  # List of remaining food on the map
        my_pos = successor.get_agent_state(self.index).get_position()  # Get the agent's position

        # The more food left, the worse the score (we want to minimize remaining food)
        features['successor_score'] = -len(food_list)

        # If there is food available, calculate the minimum distance to food
        if food_list:
            features['distance_to_food'] = min(self.get_maze_distance(my_pos, food) for food in food_list)

        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(successor)]  # List of enemies
        defenders = [e for e in enemies if e.get_position() is not None and not e.is_pacman]  # Defenders (not pacman)

        # If there are defenders close by, prioritize returning to home base
        if defenders:
            min_defender_dist = min(self.get_maze_distance(my_pos, d.get_position()) for d in defenders)
            features['distance_to_defender'] = min_defender_dist
            
            # If the defender is too close, return home immediately
            if min_defender_dist < 5:
                features['return_home'] = 1
                # Calculate the distance to the home base
                home_base = self.get_home_base(successor)
                features['distance_to_home'] = self.get_maze_distance(my_pos, home_base)
            else:
                features['return_home'] = 0
        else:
            features['return_home'] = 0  # No defenders nearby, no need to return home

        return features

    def get_weights(self, game_state, action):
        return {
            'successor_score': 100,           # Maximize score (we want to collect food)
            'distance_to_food': -1,           # Penalize the distance to food (the further the food, the worse)
            'distance_to_defender': 10,       # Penalize the proximity to defenders
            'return_home': 1000,             # If close to a defender, prioritize returning home
            'distance_to_home': -10          # Penalize the distance to home base (the further, the worse)
        }

    def choose_action(self, game_state):
        my_pos = game_state.get_agent_state(self.index).get_position()
        food_list = self.get_food(game_state).as_list()
        enemies = [game_state.get_agent_state(i).get_position() for i in self.get_opponents(game_state) if game_state.get_agent_state(i).get_position()]
        carrying_food = game_state.get_agent_state(self.index).num_carrying  # how much food the pacman have

        # Find the home
        home_positions = self.get_home_positions(game_state)

        # Try to go home if we have food and the enemy is nearby
        if carrying_food > 0 and any(util.manhattan_distance(my_pos, e) <= 5 for e in enemies):
            best_action = a_star_search(game_state, my_pos, home_positions, enemies)
            if best_action:
                return best_action  # If we find a safe way to go home, go home

        # Search food with A*
        if food_list:
            best_action = a_star_search(game_state, my_pos, food_list, enemies)
            if best_action:
                return best_action  # If A* find a way we have to go by this way

        # If A* doesn't find a good way, use nrmal strategy
        return super().choose_action(game_state)

    def get_home_positions(self, game_state):
        """
        Find the positions of home
        """
        width = game_state.data.layout.width
        if self.red:
            home_x = width // 2 - 1  # Red
        else:
            home_x = width // 2  # blue
        
        return [(home_x, y) for y in range(game_state.data.layout.height) if not game_state.has_wall(home_x, y)]



class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    An agent focused on defending its territory.
    """
    def get_features(self, game_state, action):
        features = util.Counter()  # Initialize features dictionary
        successor = self.get_successor(game_state, action)  # Get the successor state after the action
        my_state = successor.get_agent_state(self.index)  # Get the agent's state
        my_pos = my_state.get_position()  # Get the agent's position

        features['on_defense'] = 0 if my_state.is_pacman else 1  # True if the agent is not a pacman, indicating it's on defense
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]  # List of enemy agents
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]  # List of pacman invaders

        features['num_invaders'] = len(invaders)  # Count of invaders

        if invaders:
            # Calculate the distance to the nearest invader
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)  # Minimum distance to an invader
            features['pursue_invader'] = 1 if features['invader_distance'] <= 3 else 0  # If invader is close, pursue

        # If the agent stops, penalize it
        if action == Directions.STOP:
            features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        # If the agent is reversing, penalize it
        if action == rev:
            features['reverse'] = 1
        
        return features

    def get_weights(self, game_state, action):
        return {
            'num_invaders': -1000,  # Strongly penalize the presence of invaders
            'on_defense': 100,      # Reward for being on defense (not a pacman)
            'invader_distance': -10, # Penalize if invaders are close
            'stop': -100,           # Penalize stopping
            'reverse': -2,          # Penalize reversing
            'pursue_invader': 500   # High weight for pursuing invaders
            }
