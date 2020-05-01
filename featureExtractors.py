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
        walls = state.getWalls()

        features = util.Counter()
        features["bias"] = 1.0

        arena_size = walls.height * walls.width
        pacman = state.generateSuccessor(0, action).getPacmanPosition()

        # Finds closest scared and closest non-scared ghost
        for i in range(len(factors["ghost_locs"])):
            # Ghost is scared
            if factors["scared"][i] > 0:
                cur_distance = len(BFS.BFS(pacman, factors["ghost_locs"][i], state))
                # Scared ghost values
                if cur_distance <= 7:
                    features["scared-ghost-7-away"] += 1
                    if cur_distance <= 5:
                        features["scared-ghost-5-away"] += 1
                        if cur_distance <= 3:
                            features["scared-ghost-3-away"] += 1
                            if cur_distance <= 2:
                                features["scared-ghost-2-away"] += 1
                                if cur_distance <= 1:
                                    features["can-eat-scared-ghost"] += 1
                                    if cur_distance <= 0:
                                        features["eating-scared-ghost"] += 1

            # Ghost is not scared
            # Need to determine if ghost is facing/is a threat to pacman
            else:
                # Find all potential moves for ghost
                legal_ghost_actions = state.getLegalActions(i + 1)
                legal_ghost_moves = []
                for action in legal_ghost_actions:
                    next_state = state.generateSuccessor(i + 1, action)
                    legal_ghost_moves.append(next_state.getGhostPosition(i+1))

                possible_actions = Actions.getLegalNeighbors(state.getGhostPosition(i+1), state.getWalls())

                # Find all non-potential moves for a ghost
                illegal_moves = []
                for possible_action in possible_actions:
                    if possible_action not in legal_ghost_moves:
                        illegal_moves.append(possible_action)

                # Runs BFS without the spots behind the current ghosts (ghosts can't go backward)
                cur_distance = len(BFS.BFS(pacman, factors["ghost_locs"][i], state, illegal_moves))

                # Ghost values
                if cur_distance <= 7:
                    features["ghost-7-away"] += 1
                    if cur_distance <= 5:
                        features["ghost-5-away"] += 1
                        if cur_distance <= 3:
                            features["ghost-3-away"] += 1
                            if cur_distance <= 2:
                                print "HIT"
                                features["ghost-2-away"] += 1
                                if cur_distance <= 1:
                                    features["ghost-1-away"] += 1

        # Capsule values: distances sorted
        capsules = []
        for i in range(len(factors["capsule_locs"])):
            capsules.append(
                [float(len(BFS.BFS(pacman, factors["capsule_locs"][i], state))), factors["capsule_locs"][i]])
        capsules.sort()
        for i in range(len(capsules)):
            if capsules[i][0] == 0:
                features["eating capsule"] = 1
            else:
                features["capsule " + str(i) + " dist"] = capsules[i][0] / arena_size

        # Food groups: Finds 3 closest food groups
        food_groups = BFS.coinGroup3s((int(pacman[0]), int(pacman[1])), state)
        food_groups.sort()

        # Records distance away and if big or small
        for i in range(len(food_groups)):
            features["food group " + str(i) + " dist"] = \
                float(food_groups[i][0]) / (arena_size + (i + 1) * 20)

            # Big or small
            if food_groups[i][1] < 5:
                features["food group " + str(i) + " size"] = 1
            else:
                features["food group " + str(i) + " size"] = 0
            if food_groups[i][0] == 0:
                features["eating"] = 1
        features.divideAll(10.0)
        return features

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