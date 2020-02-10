# Created by Blake and Adam


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

3. food? location/distance

4. power pellet T/F, locations/distance

5. coin locations, distance to coins?
	- grouping?

What things do we want explanations to say?

1. ghost proximity threshold-- nearest ghost is too close, go away from it

2. Going towards coin group

3. Going towards power pellet / food

4. Eating ghost with power pellet




'''