# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        #Initializes q values with dictionary intitialized to 0s.
        self.qValues = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        if not self.qValues.has_key((state,action)):
            self.qValues[(state, action)]= 0.0

        return self.qValues[(state, action)]


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        actions = self.getLegalActions(state)
        if not actions:
            return 0.0
        qValues = []
        for action in actions:
            qValue = self.getQValue(state, action)
            qValues.append(qValue)
        return max(qValues)

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """

        actions = self.getLegalActions(state)
        if not actions:
            return None
        qActions = util.Counter()
        for action in actions:
            qActions[action] = self.getQValue(state, action)
        return qActions.argMax()



    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        if legalActions:
            if util.flipCoin(self.epsilon):
                action = random.choice(legalActions)
            else:
                action = self.computeActionFromQValues(state)
        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        estimate = reward + self.discount * self.computeValueFromQValues(nextState)
        self.qValues[(state, action)] = (1 - self.alpha) * self.getQValue(state, action) + self.alpha * estimate

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)

    def save(self, weights, fileName):
        with open(fileName, 'w') as file:
            for key, value in weights.items():
                file.write(key + ":" + str(value) + "\n")

    def loadWeights(self, fileName):
        loadedWeights = util.Counter()
        with open(fileName) as file:
            for line in file:
                if line != "\n":
                    (key, value) = line.split(":")
                    loadedWeights[key] = float(value[:-2])

        return loadedWeights


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='SimpleExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()
        # Automatically loads weights if any were previously saved, otherwise initializes empty.
        #self.weights = self.loadWeights("weightData.txt")
        #self.decisionWeights = self.loadWeights("decisionsWeights.txt")

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        return self.weights * self.featExtractor.getFeatures(state, action)

    def getInputWeightCombinations(self, state, action):
        wKeys = self.weights.keys()
        combinations = []
        features = self.featExtractor.getFeatures(state, action)

        for k, v in features.items():
            if k in wKeys and k != "bias":
                inputWeightCombo = v * self.weights[k]
                combinations.append((k, inputWeightCombo))

        return combinations

    def updateDecisionWeights(self, rating, combinations):
        #Do something if no options given
        if rating is 0 or None:
            pass
        #Do something if all options were bad (None of the above)
        if rating == "4" or int(rating) > len(combinations):
            pass

        bestIndex = int(rating) - 1
        for i in range(len(combinations)):
            featureKey = combinations[i][0]
            if i == bestIndex:
                #self.decisionWeights[featureKey] += 1 # TODO Update a valuable learning algorithm
                pass
            else:
                pass
                #self.decisionWeights[featureKey] += -1



    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
           First, compute the difference between the two states, in which we
           add the reward for going to the next state to the discount factor
           multiplied by the Value of going to the next state, subtracting the
           q value of the current state.
           Then, calculate your weights vector, which should be the current weights
           plus the learning factor multiplied by the difference in the two states
           and the feature vector.
        """
        difference = reward + self.discount * self.computeValueFromQValues(nextState) - self.getQValue(state, action)
        features = self.featExtractor.getFeatures(state, action)
        for featureKey in features:
            self.weights[featureKey] += self.alpha * difference * features[featureKey]


    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"

            # self.save(self.weights)
            # print(self.weights*self.featExtractor.getFeatures)
            # print(type(self.weights), len(self.weights))
            # print("----------------------------")
            # print(self.loadWeights())
            # print(type(self.loadWeights()), len(self.loadWeights()))

