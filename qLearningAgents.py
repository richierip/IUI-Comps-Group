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
import time


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

    def saveDecisionWeights(self, weights, training_rounds, fileName):
        print(weights)
        rawWeights = open(fileName, 'w')
        rawWeights.write("training rounds:" + str(training_rounds) + "\n")
        for category, miniDict in weights.items():
            rawWeights.write(category + "\n")
            for key, subValue in miniDict.items():
                rawWeights.write(key + ":" + str(subValue) + "\n")
            rawWeights.write("---------------\n")

    def loadDecisionWeights(self, fileName):
        loadedDecisionWeights = {"ghost 0": util.Counter(),
                                 "ghost 1": util.Counter(),
                                 "capsule": util.Counter(),
                                 "food group small": util.Counter(),
                                 "closest food group": util.Counter()}
        rawWeights = open(fileName)
        newWeights = True
        curDict = ""
        rounds = 0
        for line in rawWeights:
            if line != "\n":
                if "training rounds" in line:
                    rounds = int(line.split(":")[1])
                elif newWeights:
                    curDict = line.strip()
                    newWeights = False
                elif line.strip() != "---------------":
                    (key, value) = line.split(":")
                    loadedDecisionWeights[curDict][key] = float(value[:-2])
                else:
                    newWeights = True
        return loadedDecisionWeights, rounds


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
        if numTraining == 0:
            args['epsilon'] = 0
            args['gamma'] = 0
            args['alpha'] = 0
        else:
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
            self.weights = self.loadWeights("QLearningWeightData.txt")
            print "Weights Loaded Successfully"
        except:
            self.weights = util.Counter()

        # Loads explanation weights
        try:
            self.decisionWeights, self.training_rounds = self.loadDecisionWeights("QLearningDecisionWeights.txt")
            print("Decision weights successfully loaded\n")
        # # If none found uses loaded weights for movement
        except:
            self.decisionWeights = {"ghost 0": util.Counter(),
                                    "ghost 1": util.Counter(),
                                    "capsule": util.Counter(),
                                    "food group small": util.Counter(),
                                    "closest food group": util.Counter()}
            self.training_rounds = 0

    def getTrainingRounds(self):
        return self.training_rounds

    def updateExplanationRounds(self):
        self.training_rounds += 1

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

    # Calulates input*weight combinations for each dictionary in a dictionary
    # Used for explanations
    # Returns [(key, value),...] in sorted order from highest value to lowest
    def getOutputQValues(self, state, action):
        combinations = []
        features = self.featExtractor.getFeaturesExplanations(state, action)

        # Dot product function to multiply v and features as both are Counter() instances.
        for k, v in self.decisionWeights.items():
            qval = v * features
            combinations.append((k, qval))

        combinations = sorted(combinations, key=lambda x: x[1], reverse=True)
        return combinations

    # Updates weights for explanation NN
    def updateDecisionWeights(self, state, action, ratings, combinations):
        mults = [1, 20, 10, 0, -15, -25]
        if ratings[0] == "0":
            pass

        features = self.featExtractor.getFeaturesExplanations(state, action)
        for i in range(len(ratings)):
            explanationKey = combinations[i][0]
            t = max(0, -(self.getTrainingRounds() / 2000) ** 3 + 1)
            reward = mults[int(ratings[i])]
            for featurekey in features:
                self.decisionWeights[explanationKey][featurekey] += reward * features[featurekey] * t

    @staticmethod
    # Takes in state and action and key for key factor
    # Returns most likely explanation
    def generateFeatureExplanation(good_key, state, action, num_factors=1):
        next_state = state.generateSuccessor(0, action)
        moving = not bool(next_state.getPacmanPosition() == state.getPacmanPosition())
        good = interpretKey(good_key, state, action)

        if not moving and "ghost" in good_key:
            return "Not moving because of " + str(good[3]) + " which is " + str(good[1]) + " moves away"
        elif not moving:
            return "NOT moving for unknown reason"
        else:
            return heuristic.genExplanation(good)

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

        # If training is finished
        if self.episodesSoFar == self.numTraining:
            # Save weights for movement
            # self.save(self.weights, "QLearningWeightData.txt")

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
                for i in range(int(value * 10)):
                    ghosts["scared-ghost-num-" + str(i)] += value / (features[key] * 10)
            else:
                for i in range(int(value * 10)):
                    ghosts["ghost-num-" + str(i)] += value / (features[key] * 10)

    # Remove all ghost weights
    for key, weight in combinations.items():
        if "ghost" in key:
            del combinations[key]

    # Add new ghost weights
    for key, value in ghosts.items():
        combinations[key] = value

    return combinations


# Takes in a key and makes a best guess about object it is referring to
def interpretKey(key, state, action):
    next_state = state.generateSuccessor(0, action)

    if "ghost" in key:
        num = int(re.search(r'\d', key).group())
        cur_ghost_position = state.getGhostPositions()[num]
        timer = state.getGhostState(num + 1).getScaredTimer()

        # Scared or not scared
        # (arbitrary weight, distance, towards=-1/away=1, type)
        if timer == 0:
            # heuristic.distanceDiff(state, next state, list w/ current pos, ghost=True, [(scared, timer)])[num]
            cur_ghost_info = heuristic.distanceDiff(state,
                                                    next_state,
                                                    [cur_ghost_position, cur_ghost_position],
                                                    True,
                                                    [(1, 0), (1, 0)])[num]

            # If ghosts are in home, BFS sometimes doesn't find legal move bc ghosts can't usually go backwards
            # Treats ghost as scared
            if cur_ghost_info[0] == 0:
                cur_ghost_info = heuristic.distanceDiff(state,
                                                        next_state,
                                                        [cur_ghost_position, cur_ghost_position],
                                                        False)[num]

            cur_ghost = (1,
                         cur_ghost_info[0] - 1,
                         cur_ghost_info[1],
                         heuristic.ghosts[num] + " ghost")

        else:
            cur_ghost_info = heuristic.distanceDiff(state,
                                                    next_state,
                                                    [cur_ghost_position,cur_ghost_position],
                                                    True,
                                                    [(-1, 5), (-1, 5)])[num]
            cur_ghost = (1,
                         cur_ghost_info[0] - 1,
                         cur_ghost_info[1],
                         "scared " + heuristic.ghosts[num] + " ghost")

        return cur_ghost

    elif "food" in key:
        try:
            # Get all food groups
            food = BFS.coinGrouping(next_state.getPacmanPosition(), state)
            food_groups_info = heuristic.foodGroupDiff(food, state.getPacmanPosition(), next_state.getPacmanPosition(), state)
            food_groups_info.sort()

            # Remove ones PacMan is moving away from
            for info in food_groups_info:
                if info[1] != -1:
                    food_groups_info.remove(info)

            if "small" in key:
                cur_food_group_info = food_groups_info[0]
                # (arbitrary weight, distance, towards=-1, type, size)
                cur_food = (1, cur_food_group_info[0], cur_food_group_info[1], "small food group", cur_food_group_info[2])

            else:
                # Defaults to closest: (arbitrary weight, distance, towards=-1, type, size)
                cur_food = (1, food_groups_info[0][0], food_groups_info[0][1], "food group", food_groups_info[0][2])

                # Finds closest food group larger than 4
                for i in range(len(food_groups_info)):
                    if food_groups_info[i][2] >= 4:
                        # (arbitrary weight, distance, towards=-1, type, size)
                        cur_food = (1, food_groups_info[i][0], food_groups_info[i][1], "food group", food_groups_info[i][2])
                        break

            return cur_food

        except:
            cur_food = (1, 0, 1, "ERROR: no food in front of Pacman", 0)

        return cur_food

    elif "capsule" in key:
        try:
            cur_capsule_info = sorted(heuristic.distanceDiff(state, next_state, state.getCapsules()))[0]

            # (arbitrary weight, distance, towards=-1, type, size)
            cur_capsule = (1, cur_capsule_info[0], cur_capsule_info[1], "capsule")
        except:
            cur_capsule = (1, 0, 1, "no capsule", 0)

        return cur_capsule


# Takes in an input factor (i.e. ghost distnace) and retuns an interpretable sentence about it
def interpret(cur_factor, factors, state):
    print "HELLO"
    pass
    # Get number from factor. Bc everything is sorted by distance, factor num corresponds with factors
    try:
        num = int(re.search(r'\d+', cur_factor).group())
    except:
        num = None
    if num is None:
        if cur_factor is "eating":
            return "No important factor. Eating..."
        else:
            return "UNKNOWN INPUT: " + str(cur_factor)
    else:
        # Ghosts
        if "ghost" in cur_factor:

            # Sort scared and non scared
            ghosts = []
            scared_ghosts = []
            for i in range(len(factors["ghosts"])):
                if factors["scared"][i][0] == 1:
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
