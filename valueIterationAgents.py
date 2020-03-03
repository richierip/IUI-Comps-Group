# valueIterationAgents.py
# -----------------------
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

import sys

import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        """
        Creates a dictionary in which we store the maximum value option from each
        other option.  It is a bit like a linked list, in that for each state, we
        have the preferred state to visit.
        """
        nextValues = util.Counter()
        for stage in range(0, self.iterations):
            for state in mdp.getStates():
                neighbors = []
                if self.mdp.isTerminal(state):
                    neighbors.append(0)
                for action in mdp.getPossibleActions(state):
                    neighbors.append(self.computeQValueFromValues(state, action))

                nextValues[state] = max(neighbors)

            self.values = nextValues.copy()



    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    """
    The selected state has several other states it can immediately access
    For each state, we multiply the probability of getting to the next state
    by the quantity of a) reward for getting to that state, and b) the discount
    multiplied by the current value.  Finally, we add up the sum of all the neighbors
    to get the Q value.
    """
    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        neighbors = self.mdp.getTransitionStatesAndProbs(state, action)
        neighborArray = []
        for neighbor in neighbors:
            #recall neighbor[0] is the state
            reward = self.mdp.getReward(state, action, neighbor[0])
            #recall neighbor[1] is the probability of getting to the next state
            neighborArray.append(neighbor[1] * (reward + self.discount * self.values[neighbor[0]]))
        return sum(neighborArray)

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        actions = self.mdp.getPossibleActions(state)
        if not actions:
            return None

        bestAction = None
        bestQValue = None
        for action in actions:
            qValue = self.computeQValueFromValues(state, action)
            if bestAction == None:
                bestAction = action
                bestQValue = qValue
            if qValue > bestQValue:
                bestAction = action
                bestQValue = qValue
        return bestAction


    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
