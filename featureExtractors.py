# featureExtractors.py
# --------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"Feature extractors for Pacman game states"

from game import Directions, Actions
import util
import heuristic
import BFS


class FeatureExtractor:
    def getFeatures(self, state, action):
        """
          Returns a dict from features to counts
          Usually, the count will just be 1.0 for
          indicator functions.
        """
        util.raiseNotDefined()


class SimpleExtractor(FeatureExtractor):
    """
    Used in neural network. Generates ghost distances, capsule distances, closest 3 foods and size, etc.
    [ghost dist, ghost scared dist, scared timer, capsule, food groups, ...]
    """
    def getFeatures(self, state, action):
        factors = heuristic.gatherFactors(state)
        pacman = state.generateSuccessor(0, action).getPacmanPosition()

        features = util.Counter()
        features["bias"] = 1.0

        # Ghosts
        self.getFeatureGhosts(factors, features, pacman, state)

        # Capsule values: distances sorted
        self.getFeatureCapsule(factors, features, pacman, state)

        # Food groups: Finds 3 closest food groups
        self.getFeatureFood(features, pacman, state)

        features.divideAll(10.0)
        return features

    # Feature extractor for explanations
    # Similar to getFeatures() but adds features for movement towards=-1/away=1
    # Ghost distance for each ghost is one variable
    def getFeaturesExplanations(self, state, action):
        factors = heuristic.gatherFactors(state)
        walls = state.getWalls()
        pacman = state.generateSuccessor(0, action).getPacmanPosition()
        old_pac_pos = state.getPacmanPosition()

        features = util.Counter()
        features["bias"] = 1.0

        # Ghosts
        self.getFeatureGhosts(factors, features, pacman, state, old_pac_pos)

        # Capsules
        self.getFeatureCapsule(factors, features, pacman, state, old_pac_pos)

        # Food
        self.getFeatureFood(features, pacman, state, old_pac_pos)

        features.divideAll(10.0)
        return features

    @staticmethod
    # Finds close scared and non-scared ghosts
    def getFeatureGhosts(factors, features, pacman, state, old_pac_pos=None):
        for i in range(len(factors["ghost_locs"])):
            # Ghost is scared
            if factors["scared"][i] > 0:
                farthest_consideration = 15
                cur_distance = min(len(BFS.BFS(pacman, factors["ghost_locs"][i], state)), farthest_consideration)

                # Distance booleans
                for j in range(farthest_consideration - cur_distance):
                    features["scared-ghost-" + str(i) + "-" + str(farthest_consideration - j) + "-away"] = 1

                # Directional information
                if old_pac_pos is not None:
                    features["scared-ghost-" + str(i) + "-towards"] = \
                        directional(factors["ghost_locs"][i],
                                    old_pac_pos,
                                    pacman,
                                    state)

            # Ghost is not scared
            # Need to determine if ghost is facing/is a threat to pacman
            else:
                # Find all potential moves for ghost
                legal_ghost_actions = state.getLegalActions(i + 1)
                legal_ghost_moves = []
                for action in legal_ghost_actions:
                    next_state = state.generateSuccessor(i + 1, action)
                    legal_ghost_moves.append(next_state.getGhostPosition(i + 1))

                possible_actions = Actions.getLegalNeighbors(state.getGhostPosition(i + 1), state.getWalls())

                # Find all non-potential moves for a ghost
                illegal_moves = []
                for possible_action in possible_actions:
                    if possible_action not in legal_ghost_moves:
                        illegal_moves.append(possible_action)

                # Runs BFS without the spots behind the current ghosts (ghosts can't go backward)
                farthest_consideration = 15
                path = BFS.BFS(pacman, factors["ghost_locs"][i], state, illegal_moves)

                # Finds current distance. Check for edge case where ghost is in house
                if path is []:
                    cur_distance = min(len(BFS.BFS(pacman, factors["ghost_locs"][i], state)), farthest_consideration)

                # Normal case. Illegal moves excluded
                else:
                    cur_distance = min(len(path), farthest_consideration)

                # Distance booleans
                for j in range(farthest_consideration - cur_distance):
                    features["ghost-" + str(i) + "-" + str(farthest_consideration - j) + "-away"] = 1

                # Directional information
                if old_pac_pos is not None:
                    features["ghost " + str(i) + " towards"] = \
                        directional(factors["ghost_locs"][i],
                                    old_pac_pos,
                                    pacman,
                                    state)

    @staticmethod
    # Returns capsule distances sorted by distance
    def getFeatureCapsule(factors, features, pacman, state, old_pac_pos=None):
        capsules = []
        for i in range(len(factors["capsule_locs"])):
            capsules.append(
                [float(len(BFS.BFS(pacman, factors["capsule_locs"][i], state))), factors["capsule_locs"][i]])
        capsules.sort()
        if len(capsules) > 0:
            farthest_consideration = 15
            for i in range(min(int(capsules[0][0]), farthest_consideration)):
                features["closest-capsule-" + str(farthest_consideration - i) + "-away"] = 1

            if old_pac_pos is not None:
                features["towards-capsule"] = \
                    directional(factors["capsule_locs"][0],
                                old_pac_pos,
                                pacman,
                                state)

    @staticmethod
    # Returns food groups: Finds 3 closest food groups and records if big or small
    def getFeatureFood(features, pacman, state, old_pac_pos=None):
        food_groups = BFS.coinGroup3s((int(pacman[0]), int(pacman[1])), state)
        food_groups.sort()
        # Records distance away and if big or small
        for i in range(len(food_groups)):
            farthest_consideration = 15
            for j in range(min(food_groups[i][0], farthest_consideration)):
                features["food-group-" + str(i) + "-" + str(farthest_consideration - j) + "-away"] = 1

            # Big or small
            if food_groups[i][1] < 5:
                features["food group " + str(i) + " size"] = 1

            # Eating or not
            if food_groups[i][0] == 0:
                features["eating"] = 1

        # Directional information
        if old_pac_pos is not None:
            food_groups_old = BFS.coinGroup3s((int(old_pac_pos[0]), int(old_pac_pos[1])), state)
            for i in range(len(food_groups)):
                if food_groups_old[i][0] > food_groups[i][0]:
                    features["towards food" + str(i)] = 1

    @staticmethod
    # Returns ghost values seperately
    def getFeatureGhostsSeperate(factors, features, pacman, state):
        for i in range(len(factors["ghost_locs"])):
            cur_distance = len(BFS.BFS(pacman, factors["ghost_locs"][i], state))
            if factors["scared"][i] > 0:
                features["ghost " + str(i) + " scared dist"] = min(cur_distance, 7)
            else:
                features["ghost " + str(i) + " dist"] = min(cur_distance, 7)
        for i in range(len(factors["ghost_locs"])):
            # Ghost is scared
            if factors["scared"][i] > 0:
                cur_distance = len(BFS.BFS(pacman, factors["ghost_locs"][i], state))
                # Scared ghost values
                if cur_distance <= 7:
                    features["ghost " + str(i) + " (scared) 7 away"] = 1
                    if cur_distance <= 5:
                        features["ghost " + str(i) + " (scared) 5 away"] = 1
                        if cur_distance <= 3:
                            features["ghost " + str(i) + " (scared) 3 away"] = 1
                            if cur_distance <= 2:
                                features["ghost " + str(i) + " (scared) 2 away"] = 1
                                if cur_distance <= 1:
                                    features["ghost " + str(i) + " (scared) 1 away"] = 1
                                    if cur_distance <= 0:
                                        features["ghost " + str(i) + " (scared) 0 away"] = 1

            # Ghost is not scared
            # Need to determine if ghost is facing/is a threat to pacman
            else:
                # Find all potential moves for ghost
                legal_ghost_actions = state.getLegalActions(i + 1)
                legal_ghost_moves = []
                for action in legal_ghost_actions:
                    next_state = state.generateSuccessor(i + 1, action)
                    legal_ghost_moves.append(next_state.getGhostPosition(i + 1))

                possible_actions = Actions.getLegalNeighbors(state.getGhostPosition(i + 1), state.getWalls())

                # Find all non-potential moves for a ghost
                illegal_moves = []
                for possible_action in possible_actions:
                    if possible_action not in legal_ghost_moves:
                        illegal_moves.append(possible_action)

                # Runs BFS without the spots behind the current ghosts (ghosts can't go backward)
                cur_distance = len(BFS.BFS(pacman, factors["ghost_locs"][i], state, illegal_moves))

                # Ghost values
                if cur_distance <= 7:
                    features["ghost " + str(i) + " 7 away"] = 1
                    if cur_distance <= 5:
                        features["ghost " + str(i) + " 5 away"] = 1
                        if cur_distance <= 3:
                            features["ghost " + str(i) + " 3 away"] = 1
                            if cur_distance <= 2:
                                features["ghost " + str(i) + " 2 away"] = 1
                                if cur_distance <= 1:
                                    features["ghost " + str(i) + " 1 away"] = 1


# Reurns 1 if moving away and 0 otherwise
def directional(feature_pos, cur_pac_pos, next_pac_pos, state):
    cur_dis = len(BFS.BFS(cur_pac_pos, feature_pos, state))
    next_dist = len(BFS.BFS(next_pac_pos, feature_pos, state))
    if next_dist - cur_dis < 0:
        return 1
    else:
        return 0

    # Basic feature extractor. Deprecated
    # def getFeatures2(self, state, action):
    #     # extract the grid of food and wall locations and get the ghost locations
    #     food = state.getFood()
    #     walls = state.getWalls()
    #     ghosts = state.getGhostPositions()
    #
    #     features = util.Counter()
    #
    #     features["bias"] = 1.0
    #
    #     # compute the location of pacman after he takes the action
    #     x, y = state.getPacmanPosition()
    #     dx, dy = Actions.directionToVector(action)
    #     next_x, next_y = int(x + dx), int(y + dy)
    #
    #     # count the number of ghosts 1-step away
    #     features["#-of-ghosts-1-step-away"] = sum(
    #         (next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)
    #
    #     # if there is no danger of ghosts then add the food feature
    #     if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
    #         features["eats-food"] = 1.0
    #
    #     dist = closestFood((next_x, next_y), food, walls)
    #     if dist is not None:
    #         # make the distance a number less than one otherwise the update
    #         # will diverge wildly
    #         features["closest-food"] = float(dist) / (walls.width * walls.height)
    #     features.divideAll(10.0)
    #     return features

# class IdentityExtractor(FeatureExtractor):
#     def getFeatures(self, state, action):
#         feats = util.Counter()
#         feats[(state, action)] = 1.0
#         return feats

# class CoordinateExtractor(FeatureExtractor):
#     def getFeatures(self, state, action):
#         feats = util.Counter()
#         feats[state] = 1.0
#         feats['x=%d' % state[0]] = 1.0
#         feats['y=%d' % state[0]] = 1.0
#         feats['action=%s' % action] = 1.0
#         return feats

# def closestFood(pos, food, walls):
#     """
#     closestFood -- this is similar to the function that we have
#     worked on in the search project; here its all in one place
#     """
#     fringe = [(pos[0], pos[1], 0)]
#     expanded = set()
#     while fringe:
#         pos_x, pos_y, dist = fringe.pop(0)
#         if (pos_x, pos_y) in expanded:
#             continue
#         expanded.add((pos_x, pos_y))
#         # if we find a food at this location then exit
#         if food[pos_x][pos_y]:
#             return dist
#         # otherwise spread out from the location to its neighbours
#         nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
#         for nbr_x, nbr_y in nbrs:
#             fringe.append((nbr_x, nbr_y, dist + 1))
#     # no food found
#     return None


