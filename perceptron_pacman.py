# perceptron_pacman.py
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


# Perceptron implementation for apprenticeship learning
import util
from perceptron import PerceptronClassifier
from pacman import GameState

PRINT = True


class PerceptronClassifierPacman(PerceptronClassifier):
    def __init__(self, legalLabels, maxIterations):
        PerceptronClassifier.__init__(self, legalLabels, maxIterations)
        self.weights = util.Counter()

    def classify(self, data ):
        """
        Data contains a list of (datum, legal moves)
        
        Datum is a Counter representing the features of each GameState.
        legalMoves is a list of legal moves for that GameState.
        """
        guesses = []
        for datum, legalMoves in data:
            vectors = util.Counter()
            for l in legalMoves:
                vectors[l] = self.weights * datum[l] #changed from datum to datum[l]
            guesses.append(vectors.argMax())
        return guesses


    def train( self, trainingData, trainingLabels, validationData, validationLabels ):
        self.features = trainingData[0][0]['Stop'].keys() # could be useful later
        # DO NOT ZERO OUT YOUR WEIGHTS BEFORE STARTING TRAINING, OR
        # THE AUTOGRADER WILL LIKELY DEDUCT POINTS.

        for iteration in range(self.max_iterations):
            print "Starting iteration ", iteration, "..."
            for i in range(len(trainingData)):
                "*** YOUR CODE HERE ***"
                datum, legal_moves = trainingData[i]
                label = trainingLabels[i]
                score = util.Counter()
                for move in legal_moves:
                    score[move] = self.weights * datum[move] 
                    # The score for a particular move is calculated by
                    # multiplying the binary input by the initialized weight 
                a_prime = score.argMax() 
                # find the argument that produced the produced the highest score
                self.weights += datum[label]
                self.weights -= datum[a_prime]
                # adjust weights accordingly 

    # Returns input weight combinations for explanation generation

    ## Not yet utilized
    def PerceptronInputWeight(self, state, action):
        wKeys = self.weights.keys()
        combinations = []
        features = self.featExtractor.getFeatures(state, action)

        for k, v in features.items():
            if k in wKeys and k != "bias":
                inputWeightCombo = v * self.weights[k]
                combinations.append((k, inputWeightCombo))

        return combinations