from pacman import Directions
from game import Agent
import random
import game
import util

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