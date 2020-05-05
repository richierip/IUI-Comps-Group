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
import operator
import random, util, math
import heuristic
import re


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

        # Initializes q values with dictionary intitialized to 0s.
        self.qValues = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        if not self.qValues.has_key((state, action)):
            self.qValues[(state, action)] = 0.0

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

    # Saves given weights to a file with fileName
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

    def __init__(self, epsilon=0.05, gamma=0.8, alpha=0.2, numTraining=0, **args):
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
        action = QLearningAgent.getAction(self, state)
        self.doAction(state, action)
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
        # self.weights = util.Counter()

        # Automatically loads weights if any were previously saved, otherwise initializes empty.
        try:
            self.weights = self.loadWeights("weightData.txt")
        except:
            self.weights = util.Counter()

        # Loads explanation weights
        try:
            self.decisionWeights = self.loadWeights("decisionWeights.txt")
        # If none found uses loaded weights for movement
        except:
            self.decisionWeights = self.weights.copy()

    # Returns weights for movement
    def getWeights(self):
        return self.weights

    # Returns weights for decisions
    def getDecisionWeights(self):
        return self.decisionWeights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        return self.weights * self.featExtractor.getFeatures(state, action)

    # Returns input weight combinations for explanation generation
    def getInputWeightCombinations(self, state, action):
        wKeys = self.weights.keys()
        combinations = []
        features = self.featExtractor.getFeatures(state, action)

        for k, v in features.items():
            if k in wKeys and k != "bias":
                inputWeightCombo = v * self.weights[k]
                combinations.append((k, inputWeightCombo))

        return combinations

    # Updates weights for decision NN
    def updateDecisionWeights(self, state, action, rating, combinations):
        features = self.featExtractor.getFeatures(state, action)
        # Do something if no options given
        if rating is 0 or None:
            return

        # Do something if all options were bad (None of the above)
        if rating == "4" or int(rating) > len(combinations):
            for i in range(3):
                # Top 3 should be negative with high exploration rate
                featureKey = combinations[i][0]
                self.decisionWeights[featureKey] += self.alpha * -1 * features[featureKey]
            return

        bestIndex = int(rating) - 1
        for i in range(len(combinations)):
            featureKey = combinations[i][0]
            # If best option
            if i == bestIndex:
                self.decisionWeights[featureKey] += self.alpha * 1 * features[featureKey]
            # If one of top 3 choices but not best
            elif i < 3:
                self.decisionWeights[featureKey] += self.alpha * -1 * features[featureKey]

        print(self.decisionWeights)

    # Takes in state, action, and number of features needed
    # Returns list of highest weight*input combinations in readable form
    # [(interpretable explanation, original key), ...]
    def generateFeatureExplanation(self, state, action, num_factors=1):
        features = self.featExtractor.getFeatures(state, action)

        # Find input weight combinations
        combinations = util.Counter()
        for key, value in features.items():
            if key not in ["bias"]:
                combinations[key] = value * self.weights[key]

        # Create and add in newly generated ghost weights
        explainatory_combinations = combineGhostValues(combinations, features)
        explainatory_combinations = sorted(explainatory_combinations.items(), key=operator.itemgetter(1))
        explainatory_combinations.reverse()

        # Find biggest differences between current and next state
        next_state = state.generateSuccessor(0, action)
        differences = heuristic.compare(state, next_state)

        # Generate explanations for most important features and append tuple (explainable explanation, original key)
        explanations = []
        for i in range(num_factors):
            explanations.append(
                [interpret(explainatory_combinations[i][0], differences, state),
                 explainatory_combinations[i][0]])

        return explanations

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
        self.save(self.decisionWeights, "decisionWeights.txt")
        PacmanQAgent.final(self, state)

        # If training is finished
        if self.episodesSoFar == self.numTraining:
            # Save weights for movement
            self.save(self.weights, "weightData.txt")

            # print self.weights
            # print(type(self.weights), len(self.weights))
            # print("----------------------------")
            print "Done"


# Combines distance ghost weights into one weight for each ghost
# Returns original combinations but ghost values have been replaced
def combineGhostValues(combinations, features):
    # Get new values for each ghost
    ghosts = util.Counter()
    for key, value in combinations.items():
        if 'ghost' in key and features[key] > 0:
            if "scared" in key:
                for i in range(int(value*10)):
                    ghosts["scared-ghost-num-" + str(i)] += value / (features[key]*10)
            else:
                for i in range(int(value*10)):
                    ghosts["ghost-num-" + str(i)] += value / (features[key]*10)

    # Remove all ghost weights
    for key, weight in combinations.items():
        if "ghost" in key:
            del combinations[key]

    # Add new ghost weights
    for key, value in ghosts.items():
        combinations[key] = value

    return combinations


# Takes in a factor and retuns an interpretable sentence about it
def interpret(cur_factor, factors, state):
    # Get number from factor. Bc everything is sorted by distance, factor num corresponds with factors
    num = re.search(r'\d+', cur_factor)
    if num is None:
        if cur_factor is "eating":
            return "No important factor. Eating..."
        else:
            return "UNKNOWN INPUT: " + str(cur_factor)
    else:
        num = int(num.group())
        # Ghosts
        if "ghost" in cur_factor:

            # Sort scared and non scared
            ghosts = []
            scared_ghosts = []
            for i in range(len(factors["ghosts"])):
                if factors["scared"][i][1] == 1:
                    ghosts.append(factors["ghosts"][i])
                else:
                    scared_ghosts.append(factors["ghosts"][i])

            # Scared ghosts
            if "scared" in cur_factor:
                scared_ghosts.sort()
                if scared_ghosts[num][1] == -1:
                    return "Moving towards scared ghost which is " + str(scared_ghosts[num][0] + " moves away.")
                else:
                    return "Moving away from scared ghost which is " + str(scared_ghosts[num][0] + " moves away.")

            # Non-scared ghosts
            else:
                ghosts.sort()
                if ghosts[num][1] == -1:
                    return "Moving towards ghost which is " + str(ghosts[num][0] + " moves away.")
                else:
                    return "Moving away from ghost which is " + str(ghosts[num][0] + " moves away.")

        # Food
        elif "food" in cur_factor:
            food_groups = BFS.coinGroup3s((int(factors["pac_loc"][0]), int(factors["pac_loc"][1])), state)
            food_groups.sort()
            if "size" in cur_factor:
                return "SIZE of food group with " + str(food_groups[num][1]) + " pieces " + \
                       str(food_groups[num][0]) + " moves away."
            else:
                return "DISTANCE of food group with " + str(food_groups[num][1]) + " pieces " + \
                       str(food_groups[num][0]) + " moves away."

        # Capsule
        elif "capsule" in cur_factor:
            capsules = factors["capsules"]
            capsules.sort()
            if capsules[num][1] == -1:
                return "Moving towards capsule which is " + str(capsules[num][0]) + " moves away."
            else:
                return "Moving away from capsule which is " + str(capsules[num][0]) + " moves away."

        else:
            return "UNKNOWN INPUT: " + str(cur_factor)
