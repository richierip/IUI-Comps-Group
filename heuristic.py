# Created by Blake and Adam

import game
import BFS
import util


# Generates the other game states possible from current position
def genAltGameStates(gameState, nextMove):
    moves = gameState.getLegalActions(0)
    alt_games = []
    for move in moves:
        if move != nextMove:
            alt_games.append(gameState.generateSuccessor(0, move))
    return alt_games


# Used in neural network. Generates ghost distances, capsule distances, closest 3 foods and size.
# [ghost dist, ghost scared dist, scared timer, capsule, food groups]
def neuralDistances(state, action):
    factors = gatherFactors(state.generateSuccessor(0, action))

    features = util.Counter()
    pacman = factors["pacman_loc"]
    for i in range(len(factors["ghost_locs"])):
        if factors["scared"][i] > 0:
            features["ghost " + str(i)] = 0
            features["ghost " + str(i) + " scared"] = len(BFS.BFS(pacman, factors["ghost_locs"][i], state))
            features["ghost " + str(i) + " timer"] = factors["scared"][i]
        else:
            features["ghost " + str(i)] = len(BFS.BFS(pacman, factors["ghost_locs"][i], state))
            features["ghost " + str(i) + " scared"] = 0
            features["ghost " + str(i) + " timer"] = 0

    for i in range(len(factors["capsule_locs"])):
        features["capsule" + str(i)] = BFS.BFS(pacman, factors["capsule_locs"][i], state)

    food_groups = BFS.coinGroup3s((int(pacman[0]), int(pacman[1])), state)
    while len(food_groups) < 3:
        food_groups.append((0,0))
    
    for i in range(3):
        features["food group " + str(i) + " dist"] = food_groups[i][0]
        features["food group " + str(i) + " size"] = food_groups[i][1]
    return features


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
        cur_dist = len(BFS.BFS(cur_pac, foodkey, state))
        next_dist = len(BFS.BFS(next_pac, foodkey, state))
        if next_dist - cur_dist > 0:
            diff.append((next_dist, 1, len(food[foodkey])))
        else:
            diff.append((next_dist, -1, len(food[foodkey])))
    return diff


# Calculates differences between pacman and objective such as ghosts
# Returns [(absolute distance, direction), ...]
def distanceDiff(cur_state, next_state, obj_loc):
    diff = []
    cur_pac = cur_state.getPacmanPosition()
    next_pac = next_state.getPacmanPosition()

    # Creates tuple for each object w/ absolute distance and direction
    for obj in obj_loc:
        cur_dist = len(BFS.BFS(cur_pac, (int(obj[0]), int(obj[1])), cur_state))
        next_dist = len(BFS.BFS(next_pac, (int(obj[0]), int(obj[1])), next_state))
        if next_dist - cur_dist >= 0:
            diff.append((next_dist, 1))
        else:
            diff.append((next_dist, -1))
    return diff


# Finds the difference for all ghost timers
# [(total timer, -1 = timer and 1 = no timer), ...]
def scaredDiff(cur_timer, next_timer):
    diff = []

    # Creates tuple for each ghost w/ absolute time and time difference
    for i in range(len(cur_timer)):
        if next_timer[i] > 0:
            diff.append((next_timer[i], -1))
        else:
            diff.append((next_timer[i], 1))
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

    diffs['ghosts'] = distanceDiff(cur_state, next_state, cur_factors["ghost_locs"])
    diffs['scared'] = scaredDiff(next_factors["scared"], cur_factors["scared"])
    diffs["food_groups"] = foodGroupDiff(BFS.coinGrouping(next_state.getPacmanPosition(), next_state), \
                                         cur_state.getPacmanPosition(), next_state.getPacmanPosition(), next_state)
    diffs['food'] = next_factors["num_food"]
    diffs['capsules'] = distanceDiff(cur_state, next_state, cur_factors["capsule_locs"])
    return diffs


# Takes in a dictionary of factors. Returns list of tuples [(weights, explanations, towards/away), ...]
# A large weight means something good ex: moving away from a ghost, towards a scared ghost, etc.
# A negative weight implies something bad
def weight(factors):
    weights = []

    # Weight Ghosts
    for i in range(len(factors["ghosts"])):
        # 7/(max(1, distance_of_ghost)*movement_towards_or_away*scared_ghost
        cur_weight = 12 / float((max(1, factors["ghosts"][i][0]) * factors["ghosts"][i][1] * factors["scared"][i][1]))
        # direction *-1 bc it is good to move away from ghosts
        weights.append((cur_weight, "Ghost " + str(i) + " which is " + str(factors["ghosts"][i][0]) + " moves away", \
                        factors["ghosts"][i][1]))

    # Weight Food Groups
    for food in factors["food_groups"]:
        # 80/(distance*towards_away*-1) -1 bc towards shrinks distance but good
        cur_weight = 80 / float(max(food[0], 1) * food[1] * -1 * factors["food"])
        weights.append((cur_weight, "food group with " + str(food[2]) + " pieces", food[1]))

    # Weight Capsules
    for capsule in factors["capsules"]:
        # 6/(distance*towards_away*-1) -1 bc towards shrinks distance but good
        cur_weight = 6 / float((capsule[0] * capsule[1] * -1))
        weights.append((cur_weight, "capsule " + str(capsule[0]) + " moves away", capsule[1]))
    return weights


# Generates explanation from given factors
def genExplanation(factors):
    explanation = ""
    good = max(factors)
    bad = min(factors)

    # No explanation is good
    if good[0] < 1:
        return "No immediate threats or benefits. Collecting coins..."

    if good[2] == 1:
        explanation += "Moving away from " + good[1]
    else:
        explanation += "Moving towards " + good[1]
    if bad[0] < -1:
        explanation += " even though "
        if bad[1] == 1:
            explanation += "moving away from " + bad[1]
        else:
            explanation += "moving towards " + bad[1]
    print explanation
    return explanation


# Main function to be called. Gets heuristics and generates explanation
# Returns string of explanation
# nextMove = East, West, etc.
def newExplanation(cur_state, nextMove):
    # Next state chosen by game (chosen by AI)
    next_state = cur_state.generateSuccessor(0, nextMove)

    # Find biggest differences between two states
    factors = compare(cur_state, next_state)
    weighted_factors = weight(factors)

    # Returns generated explanation
    return genExplanation(weighted_factors)


# Determines if we generate a new explanation
# Needs to check reverse or if there are more than two legal moves
def threshold(gameState, nextGameState):
    # TODO 3 because of stop. If we remove stop, needs to change. Just check at end
    # Checks if at an intersection with more than one choice
    if len(gameState.getLegalActions(0)) > 3:
        return True

    # Checks if player reverses.
    old_direction = gameState.getPacmanState().getDirection()
    new_direction = nextGameState.getPacmanState().getDirection()
    if (bool(old_direction == 'Stop') != bool(new_direction == 'Stop')) or \
            (old_direction == game.Actions.reverseDirection(new_direction) and old_direction != 'Stop'):
        return True
    return False

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
