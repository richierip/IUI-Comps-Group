# Created by Blake and Adam

import game
import BFS
import util

ghosts = ["red", "pink"]


# Generates the other game states possible from current position
def genAltGameStates(gameState, nextMove):
    moves = gameState.getLegalActions(0)
    alt_games = []
    for move in moves:
        if move != nextMove:
            alt_games.append(gameState.generateSuccessor(0, move))
    return alt_games


# Gathers important data from game in dict to be used in heursitic generation
# Pacman loc (x,y)
# Ghost locations [(x,y), ...]
# Food {(x,y): set(food), } where (x,y) is the nearest coin to pacman in that group
# num_food x
# Capsules (power pellets) [(x,y), ...]
# Scared ghosts [num, ...] where num = scared timer
def gatherFactors(state):
    factors = {}
    factors["pacman_loc"] = state.getPacmanPosition()
    factors["ghost_locs"] = state.getGhostPositions()
    factors["scared"] = [ghostState.scaredTimer for ghostState in state.getGhostStates()]
    factors["num_food"] = state.getNumFood()
    factors["capsule_locs"] = state.getCapsules()
    return factors


# Calculates differences in food group states
# diff: (dist, direction, size)
def foodGroupDiff(food, cur_pac, next_pac, state):
    diff = []
    for foodkey in food.keys():
        cur_dist = len(BFS.BFS(cur_pac, foodkey, state)) - 1
        next_dist = len(BFS.BFS(next_pac, foodkey, state))
        if next_dist - cur_dist > 0:
            diff.append((next_dist, 1, len(food[foodkey])))
        else:
            diff.append((next_dist, -1, len(food[foodkey])))
    return diff


# Calculates differences between pacman and objective such as ghosts
# Returns [(absolute distance, direction), ...]
def distanceDiff(cur_state, next_state, obj_loc, ghosts=False, scared_list=None):
    diff = []
    cur_pac = cur_state.getPacmanPosition()
    next_pac = next_state.getPacmanPosition()

    # Creates tuple for each object w/ absolute distance and direction
    for i in range(len(obj_loc)):
        cur_loc = (int(obj_loc[i][0]), int(obj_loc[i][1]))
        # Non-scared ghost
        if ghosts and scared_list[i][0] == 1:
            # Find all potential moves for ghost
            legal_ghost_actions = cur_state.getLegalActions(i + 1)
            legal_ghost_moves = []
            for action in legal_ghost_actions:
                next_state = cur_state.generateSuccessor(i + 1, action)
                legal_ghost_moves.append(next_state.getGhostPosition(i + 1))

            possible_actions = game.Actions.getLegalNeighbors(cur_state.getGhostPosition(i + 1), cur_state.getWalls())

            # Find all places the ghost cannot move
            illegal_moves = []
            for possible_action in possible_actions:
                if possible_action not in legal_ghost_moves:
                    illegal_moves.append(possible_action)

            path = BFS.BFS(cur_pac, cur_loc, cur_state, illegal_moves)

            # Finds current distance. Check for edge case where ghost is in house
            if path is []:
                cur_dist = len(BFS.BFS(cur_pac, cur_loc, cur_state))
                next_dist = len(BFS.BFS(next_pac, cur_loc, cur_state))

            # Normal case. Illegal moves excluded
            else:
                cur_dist = len(path)
                next_dist = len(BFS.BFS(next_pac, cur_loc, cur_state, illegal_moves))

        else:
            cur_dist = len(BFS.BFS(cur_pac, cur_loc, cur_state))
            next_dist = len(BFS.BFS(next_pac, cur_loc, next_state))

        if next_dist - cur_dist >= 0:
            diff.append((next_dist, 1))
        else:
            diff.append((next_dist, -1))
    return diff


# Finds the difference for all ghost timers
# [(timer=-1/no timer=1, total timer), ...]
def scaredDiff(cur_timer, next_timer):
    diff = []

    # Creates tuple for each ghost w/ absolute time and time difference
    for i in range(len(cur_timer)):
        if next_timer[i] > 0:
            diff.append((-1, next_timer[i]))
        else:
            diff.append((1, next_timer[i]))
    return diff


# Takes in two games states and returns calculated relvant differences between the two
# Distance difference from closest food group [(absolute disance, difference, size), ...]
# Distance difference for each ghost [(absolute distance, difference), ...]
# Scared mode timer difference [(total timer, -1 = timer and 1 = no timer), ...]
# Food amount x
# Distance from capsules [(absolute distance, difference), ...]
def compare(cur_state, next_state):
    diffs = {}
    cur_factors = gatherFactors(cur_state)
    next_factors = gatherFactors(next_state)

    # diffs["pac_loc"] = next_state.getPacmanPosition()
    diffs['scared'] = scaredDiff(next_factors["scared"], cur_factors["scared"])
    diffs['ghosts'] = distanceDiff(cur_state, next_state, cur_factors["ghost_locs"], True, diffs["scared"])
    diffs["food_groups"] = foodGroupDiff(BFS.coinGrouping(next_state.getPacmanPosition(), cur_state), \
                                         cur_state.getPacmanPosition(), next_state.getPacmanPosition(), next_state)
    diffs['food'] = next_factors["num_food"]
    diffs['capsules'] = distanceDiff(cur_state, next_state, cur_factors["capsule_locs"])
    return diffs


# Takes in a dictionary of factors. Returns list of tuples
# [(weights, distance, towards=-1/away=1, type, size), ...]
# A large weight means something good ex: moving away from a ghost, towards a scared ghost, etc.
# A negative weight implies something bad
def weight(factors):
    # [(weight, explanation, direction), ...]
    weights = []

    # Weight Ghosts
    for i in range(len(factors["ghosts"])):
        # 8/(max(1, distance_of_ghost)*movement_towards_or_away*scared_ghost
        cur_weight = 8 / float((max(1, factors["ghosts"][i][0]) * factors["ghosts"][i][1] * factors["scared"][i][0]))
        # direction *-1 bc it is good to move away from ghosts
        if factors["scared"][i][0] == 1:
            weights.append((cur_weight,
                            factors["ghosts"][i][0] - 2,
                            factors["ghosts"][i][1],
                            ghosts[i] + " ghost"))
        else:
            weights.append((cur_weight,
                            factors["ghosts"][i][0] - 2,
                            factors["ghosts"][i][1],
                            "scared " + ghosts[i] + " ghost"))

    # Weight Food Groups
    # Food tuple has extra value for size of food
    for food in factors["food_groups"]:
        # 5/(distance away + to center) + 5/total food
        cur_weight = ((5/float(max(food[0] + min(food[2] / 2, 6), 1))) + 5/float(max(factors["food"], 1))) * food[1] * -1
        weights.append((cur_weight, food[0], food[1], "food group", food[2]))

    # Weight Capsules
    for capsule in factors["capsules"]:
        # 6/(distance*towards_away*-1) -1 bc towards shrinks distance but good
        cur_weight = 6 / float((max(capsule[0], 1) * capsule[1] * -1))
        weights.append((cur_weight, capsule[0], capsule[1], "capsule"))
    return weights


# Generates explanation from given a good and bad factor
def genExplanation(good, bad=None):
    explanation = ""
    if good[2] == 1:
        explanation += "Moving away from "
    else:
        explanation += "Moving towards "

    if "food" in good[3]:
        explanation += good[3] + " with " + str(good[4]) + " pieces"
    else:
        explanation += good[3] + " which is " + str(good[1]) + " moves away"

    # A threat was detected
    if bad is not None and bad[0] < -1:
        explanation += " even though moving "
        if bad[2] == 1:
            explanation += "away from "
        else:
            explanation += "towards "

        if "food" in bad[3]:
            explanation += bad[3] + " with " + str(bad[4]) + " pieces"
        else:
            explanation += bad[3] + " which is " + str(bad[1]) + " moves away"

    # print "DECISION"
    # for factor in factors:
    #     try:
    #         print "Weight: " + str(factor[0]) + ", Reason: " + str(factor[1]) + "Distance: " + str(factor[3])
    #     except:
    #         print "Weight: " + str(factor[0]) + ", Reason: " + str(factor[1])
    # print explanation
    return explanation


# Returns an explanation if Pac Man is not moving
def genNotMovingExplanation(factors):
    ghosts = []
    for factor in factors:
        if "ghost" in factor[3] and factor[0] > 1:
            ghosts.append(factor)
    if len(ghosts) > 0:
        return "Not moving because of " + ghosts[0][3] + " which is " + str(ghosts[0][1]) + " moves away"
    else:
        return "No immediate benefit or threat: Standing still"


# Explanation when no benefit detected for a move
def genNoBenefitExplanation(factors):
    # for factor in factors:
    #     try:
    #         print "Weight: " + str(factor[0]) + ", Reason: " + str(factor[1]) + "Distance: " + str(factor[3])
    #     except:
    #         print "Weight: " + str(factor[0]) + ", Reason: " + str(factor[1])
    # Finds nearest food group and says moving towards it
    food = []
    for factor in factors:
        # Food and moving towards or eating food
        if "food" in factor[3] and (factor[2] == 1 or factor[1] == 0):
            food.append(factor)
    # Gets most relevant food group that Pac Man is moving towards
    try:
        nearest = max(food)
    except:
        # If there is no logical reason for moving that direction (no food group or benefit)
        return "No benefit or threat detected"
    # Returns moving towards a food group
    if nearest[1] == 0:
        return "No immediate benefit or threat: Eating " + nearest[3] + " with " + str(nearest[4]) + " pieces"
    else:
        return "No immediate benefit or threat: Moving towards " + nearest[3] + " with " + str(
            nearest[4]) + " pieces"


# Main function to be called. Gets heuristics and generates explanation
# Returns string of explanation
# nextMove = East, West, etc.
def newExplanation(cur_state, nextMove):
    # Next state chosen by game (chosen by AI)
    next_state = cur_state.generateSuccessor(0, nextMove)

    # Find biggest differences between two states
    factors = compare(cur_state, next_state)
    weighted_factors = weight(factors)

    # Generate explanation
    if bool(next_state.getPacmanPosition() == cur_state.getPacmanPosition()):
        # Not moving at all
        return genNotMovingExplanation(weighted_factors)
    else:
        good = max(weighted_factors)
        bad = min(weighted_factors)

        if good[0] < 1:
            # No immediate benefit detected
            return genNoBenefitExplanation(weighted_factors)
        else:
            return genExplanation(good, bad)


# Determines if we generate a new explanation
# Needs to check reverse or if there are more than two legal moves
def threshold(gameState, nextGameState):
    old_direction = gameState.getPacmanState().getDirection()
    new_direction = nextGameState.getPacmanState().getDirection()
    # Checks if at an intersection with more than one choice
    if len(gameState.getLegalActions(0)) > 3:
        return True

    # Checks if player reverses.
    if old_direction == game.Actions.reverseDirection(new_direction) and old_direction != 'Stop':
        return True
    return False


# Used in neural network. Generates ghost distances, capsule distances, closest 3 foods and size.
# [ghost dist, ghost scared dist, scared timer, capsule, food groups]
def neuralDistances(state, action):
    factors = gatherFactors(state.generateSuccessor(0, action))

    features = util.Counter()
    pacman = factors["pacman_loc"]
    for i in range(len(factors["ghost_locs"])):
        if factors["scared"][i] > 0:
            features["ghost " + str(i) + " scared"] = len(BFS.BFS(pacman, factors["ghost_locs"][i], state))
            features["ghost " + str(i) + " timer"] = factors["scared"][i][1]
        else:
            features["ghost " + str(i)] = len(BFS.BFS(pacman, factors["ghost_locs"][i], state))

    for i in range(len(factors["capsule_locs"])):
        features["capsule" + str(i)] = len(BFS.BFS(pacman, factors["capsule_locs"][i], state))

    food_groups = BFS.coinGroup3s((int(pacman[0]), int(pacman[1])), state)
    while len(food_groups) < 3:
        food_groups.append((0, 0))

    for i in range(3):
        features["food group " + str(i) + " dist"] = food_groups[i][0]
        features["food group " + str(i) + " size"] = food_groups[i][1]
    return features


#
# 	comparison = compare(gameState, lastGameState)
# 	if comparison is over threshold:
# 		altGameStates = generateAlterateGameStates(gameState)
# 		return genExplanation(gameState, altGameStates)
#
#
#
#
# # a mock-up of what we might want our heuristic to look like
# def heuristic(gameState):
# 	pac_v = gameState.pacman.getLocation() # (x,y,v)
# 	ghosts_v = gameState.ghosts.getLocations() #[(x,y,v), (x,y,v)...]
# 	powerPellet_loc = gameState.powerPellets.getLocations() # [] if none, else ^
# 	coin_loc = gameState.coins.getLocations() # ^
#
# 	coinGroups = coinGrouping(pac_v, coin_loc)
#
# 	return {"numCoins":numCoins, "coinGroups":coinGroups, "ghostDanger":ghostDanger, ...}
#
#
# # mathematically figures out where the biggest differences are
# def compare(gameStateUsed, gameStateAlt):
# 	heuristicUsed = heuristic(gameStateUsed)
# 	heuristicAlt = heuristic(gameStateAlt)
#
# 	# somehow weight each thing to determine what's important,
# 	# assume used gamestate is somehow better
#
# 	return largestDifferenceInAlt
#
#
# # generates explanation from compare
# def genExplanation(gameStateUsed, altGameStates):
#
#
#
# 	comparisons = {}
# 	for key in altGameStates:
# 		comparisons[key] = compare(gameStateUsed, altGameStates[key])
#
# 	print(key + " direction was a bad direction because " + reason)
#
#
#
#
#
#
# # Helper functions
#
# def distance(point1, point2):
# 	# going to actually need to find a path bc it's not always a simple L
#
# def coinGrouping(pac_v, coin_loc, powerPellet_loc):
#
# 	unsearched coins = coin_loc + powerPellet_loc
#
# 	while unsearched coins:
# 	if coin next to existing group:
# 		add coin to group
# 	else:
# 		make new group with just it - note this doesn't work if it's not sorted x O x
#
#
# 	pseudoGroup = [[(x,y)...], [(x,y)...]...]
# 	for group in pseudoGroup:
# 		find coin closest to pac man location using distance func
#
# 	return [(numCoins, closest), (numCoins, closest)...]
#
# def generateAlternateGameStates(gameState):
# 	# play the game 1 state in the future, with all possible moves
#
# 	return altGameStates # {"left": gameStateLeft, "right":gameStateRight, "up": None, ...}
#
