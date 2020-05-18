# classificationAgents.py
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


# pacmanAgents.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from pacman import Directions
from game import Agent

import random
import game
import util
import pickle
import qLearningAgents

class DummyOptions:
    def __init__(self):
        self.data = "pacman"
        self.training = 25000
        self.test = 100
        self.odds = False
        self.weights = False


import perceptron_pacman
# so this is where the pacman.py call goes to, and it looks like it links from here
# into perceptron_pacman
class ClassifierAgent(Agent):
    def __init__(self, trainingData=None, validationData=None, classifierType="perceptron", agentToClone=None, numTraining=3):
        from dataClassifier import runClassifier, enhancedFeatureExtractorPacman
        legalLabels = ['Stop', 'West', 'East', 'North', 'South']
        
        didLoadFromFile = False
        self.classifierType = classifierType
        if(classifierType == "perceptron"):
            try:
                classifier = pickle.load(open("perceptron.pkl", "rb"))
                print("loaded perceptron from file")
                didLoadFromFile = True
            except:
                # here's the actual perceptron part
                print("perceptron.pkl not found")
                classifier = perceptron_pacman.PerceptronClassifierPacman(legalLabels,numTraining)
                
        self.classifier = classifier
        #looks like enhanced Feature extractor is in dataClassifier
        self.featureFunction = enhancedFeatureExtractorPacman 
        args = {'featureFunction': self.featureFunction,
                'classifier':self.classifier,
                'printImage':None,
                'trainingData':trainingData,
                'validationData':validationData,
                'agentToClone': agentToClone,
        }
        options = DummyOptions()
        options.classifier = classifierType
        runClassifier(args, options, didLoadFromFile=didLoadFromFile)

        if not didLoadFromFile:
            pickle.dump(classifier,open("perceptron.pkl","wb"))
            print("saving perceptron to file")

    def getAction(self, state):
        from dataClassifier import runClassifier, enhancedFeatureExtractorPacman
        
        features = self.featureFunction(state)
        action = self.classifier.classify([features])[0]

        if self.classifierType == "perceptron":
            pass
            #print("Features")
            #print(features)
            
            # print("#######")
            # print("0 is : ", features[0])
            #print("class weights")
            #print(self.classifier.getWeights())
            #for i in range(len(self.classifier.getWeights().keys())):
                #thisKey = self.classifier.getWeights().keys()[i]
                #if i ==1: continue
                #print("########## i is :", i)
                #print("key is : ", thisKey)

                #'''the next line triggers an indexoutofbounds error on line 453 in qlearning agents. It seems to call
                #getGhostPosition[7] or something like that, but there are only 2 ghosts so thats a mistake and it crashes. Not sure
                #what the lines ~ 450 are supposed to do but we should maybe just copy over the method and make the changes we need.'''
                #print(" interpretKeys returns : ", qLearningAgents.interpretKey(thisKey, state, action))
        

        return action

    def getExplanation(self, state):
        features = self.featureFunction(state)
        direction = self.classifier.classify([features])[0]
        print(features[direction])
        for key in features[direction]:
            pass
        return

def featureFunction(self, state):
    return self.featureFunction(state)

def scoreEvaluation(state):
    return state.getScore()
