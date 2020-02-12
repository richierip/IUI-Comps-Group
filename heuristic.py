# Created by Blake and Adam

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

# compares the cur gameState to the last one. If they are judged significantly different,
# we want to run alternate states
def threshold(gameState, lastGameState):
	comparison = compare(gameState, lastGameState)
	if comparison is over threshold:
		altGameStates = generateAlterateGameStates(gameState)
		return genExplanation(gameState, altGameStates)




# a mock-up of what we might want our heuristic to look like
def heuristic(gameState):
	pac_v = gameState.pacman.getLocation() # (x,y,v)
	ghosts_v = gameState.ghosts.getLocations() #[(x,y,v), (x,y,v)...]
	powerPellet_loc = gameState.powerPellets.getLocations() # [] if none, else ^
	coin_loc = gameState.coins.getLocations() # ^

	coinGroups = coinGrouping(pac_v, coin_loc)

	return {"numCoins":numCoins, "coinGroups":coinGroups, "ghostDanger":ghostDanger, ...}


# mathematically figures out where the biggest differences are
def compare(gameStateUsed, gameStateAlt):
	heuristicUsed = heuristic(gameStateUsed)
	heuristicAlt = heuristic(gameStateAlt)

	# somehow weight each thing to determine what's important,
	# assume used gamestate is somehow better

	return largestDifferenceInAlt
	

# generates explanation from compare
def genExplanation(gameStateUsed, altGameStates):



	comparisons = {}
	for key in altGameStates:
		comparisons[key] = compare(gameStateUsed, altGameStates[key])

	print(key + " direction was a bad direction because " + reason)
		





# Helper functions

def distance(point1, point2):
	# going to actually need to find a path bc it's not always a simple L
	
def coinGrouping(pac_v, coin_loc, powerPellet_loc):
	
	unsearched coins = coin_loc + powerPellet_loc

	while unsearched coins:
	if coin next to existing group:
		add coin to group
	else:
		make new group with just it - note this doesn't work if it's not sorted x O x


	pseudoGroup = [[(x,y)...], [(x,y)...]...]
	for group in pseudoGroup:
		find coin closest to pac man location using distance func

	return [(numCoins, closest), (numCoins, closest)...]

def generateAlternateGameStates(gameState):
	# play the game 1 state in the future, with all possible moves

	return altGameStates # {"left": gameStateLeft, "right":gameStateRight, "up": None, ...}



