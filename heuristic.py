# Created by Blake and Adam

import game
import BFS

'''
Files we think we don't need:
PacmanAgents.py - examples of basic AIs. 
	Might be useful to adapt for our ghosts, but not necessary in its own right

multiagentTestClasses.py
	this is just minimax? shit worked with it all commented

multiAgents.py is also not necessary

remove whole directory test_cases, looks like it's for grading

remove grading.py

remove VERSION

remove autograder.py

remove projectParams.py

remove testParser.py

Shit we definitely need but don't want to touch:

graphicsDisplay.py

graphicsUtils.py

textDisplay.py


keyboardAgents.py - not hard but no reason to fuck with it

layout.py

shit we need but might want to clean:


util.py

ghostAgents.py we will need to modify to have good ghosts / ghosts with personality

pacman.py

game.py
	These control the basic game



'''

# copied from multiAgents.py
'''
def evaluationFunction(self, currentGameState, action):
		"""
		Design a better evaluation function here.

		The evaluation function takes in the current and proposed successor
		GameStates (pacman.py) and returns a number, where higher numbers are better.

		The code below extracts some useful information from the state, like the
		remaining food (newFood) and Pacman position after moving (newPos).
		newScaredTimes holds the number of moves that each ghost will remain
		scared because of Pacman having eaten a power pellet.

		Print out these variables to see what you're getting, then combine them
		to create a masterful evaluation function.
		"""
		# Useful information you can extract from a GameState (pacman.py)
		successorGameState = currentGameState.generatePacmanSuccessor(action)
		newPos = successorGameState.getPacmanPosition()
		newFood = successorGameState.getFood()
		newGhostStates = successorGameState.getGhostStates()
		newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

		"*** YOUR CODE HERE ***"
		return successorGameState.getScore()
'''

'''
Approaches / things we want to implement


What info do we want

1. a list of distances to each ghost (with directions?)

2. pac man's location and direction

3. power pellet T/F, locations/distance

4. coin locations, distance to coins?
	- grouping?

What things do we want explanations to say?

1. ghost proximity threshold-- nearest ghost is too close, go away from it

2. Going towards coin group

3. Going towards power pellet

4. Eating ghost with power pellet




'''


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
    # TODO uncomment
    # factors["food_groups"] = BFS.coinGrouping(state.getPacmanPosition(), state)
    factors["num_food"] = state.getNumFood()
    factors["capsule_locs"] = state.getCapsules()
    return factors


# TODO i'm not sure how this comes in from BFS
# Calculates differences in food group states
def foodGroupDiff(cur_food, next_food):
    diff = []
    index = 0
    # Ate entire coin group
    if len(cur_food) != len(next_food):
        index = 1

    for i in range(len(cur_food)):
        diff.append(("DISTANCE AWAY", "DIRECTION 1 = away, -1 = towards"
                                      "", "SIZE OF GROUPING"))
    return diff


# Calculates differences between pacman and objective such as ghosts
# Returns [(absolute distance, direction), ...]
def distanceDiff(cur_state, next_state, obj_loc):
    diff = []
    cur_pac = cur_state.getPacmanPosition()
    next_pac = next_state.getPacmanPosition()

    # Creates tuple for each object w/ absolute distance and direction
    for obj in obj_loc:
        cur_dist = len(BFS.BFS(cur_pac, obj, cur_state))
        next_dist = len(BFS.BFS(next_pac, obj, next_state))
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
# Food amount difference x
# Distance from capsules [(absolute distance, difference), ...]
def compare(cur_state, next_state):
    diffs = {}
    cur_factors = gatherFactors(cur_state)
    next_factors = gatherFactors(next_state)

    diffs['ghosts'] = distanceDiff(cur_state, next_state, cur_factors["ghost_locs"])
    diffs['scared'] = scaredDiff(next_factors["scared"], cur_factors["scared"])
    # TODO uncomment
    # diffs["food_groups"] = foodGroupDiff(cur_factors["food_groups"], next_factors["food_groups"])
    diffs['food'] = next_factors["num_food"] - cur_factors["num_food"]
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
    # for food in factors["food_groups"]:
    # 	# 5/(distance*towards_away*-1) -1 bc towards shrinks distance but good
    #     cur_weight = 5 / float((food[0] * food[1] * -1))
    #     weights.append((cur_weight, "food group with " + str(food[2]) + " pieces", food[1]))

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
