from pacman import Directions
from game import Agent
import random
import game
import util

#from pacman import GameState
#import random, math, traceback, sys, os
import dataClassifier, samples

#### keras imports we will need ####

# import keras
# from keras.datasets import mnist
# from keras.models import Sequential
# from keras.layers import Dense, Dropout
# from keras.optimizers import RMSprop


class KerasAgent(game.Agent):
	def __init__(self):
		pass

	def getAction(self, state):
		legal = state.getLegalPacmanActions()
		#print(legal)


		current = state.getPacmanState().configuration.direction 
		# not necessary for making a random move, but could be useful in the future

		# turns out random moves are garbage so i'm gonna make it only make random
		# moves that are not stop if it's at an intersection
		legal.remove('Stop')
		if len(legal) <= 2 and current != 'Stop' and current in legal:
			return current

		randomLegalMove = legal[int(random.random() * len(legal))]

		return randomLegalMove

	# The agent will need data to train on. I think this would load it? Had a hard 
	# time trying to tell which dataset did what.
	def loadTrainingData(self, trainingSize=100, testSize=100):

		# load training data. rawTrainingData is a list of GameStates, while trainingLabels is a list 
		# of move directions (i.e. north, west etc.). Code from ClassificationTestClasses.py
		rootdata = 'pacmandata'
		rawTrainingData, trainingLabels = samples.loadPacmanData(rootdata + '/contest_training.pkl', trainingSize)
		rawValidationData, validationLabels = samples.loadPacmanData(rootdata + '/contest_validation.pkl', testSize)
		rawTestData, testLabels = samples.loadPacmanData(rootdata + '/contest_test.pkl', testSize)

		
		return (trainingLabels, validationLabels, testLabels, rawTrainingData, rawValidationData, rawTestData)

'''
Example keras code from Dave's AI class, modified / annotated. I imagine us doing something similar.
To make this work we will need to know about how the input dataset is stored in the 
array.

# Get training and testing sets
rootdata = 'pacmandata'
rawTrainingData, trainingLabels = samples.loadPacmanData(rootdata + '/contest_training.pkl', trainingSize)
rawValidationData, validationLabels = samples.loadPacmanData(rootdata + '/contest_validation.pkl', testSize)
rawTestData, testLabels = samples.loadPacmanData(rootdata + '/contest_test.pkl', testSize)

batch_size = 128
num_classes = 5 # Five for us because 4 cardinal directions, plus stop
epochs = 20

# These numbers are not right for us. Not sure what we would put here.
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# These would be different as well
x_train /= 255
x_test /= 255

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Dense(num_classes, activation='softmax', input_shape=(784,))) # input shape would be shape of our array?


# This next stuff would probably be the same

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
'''
