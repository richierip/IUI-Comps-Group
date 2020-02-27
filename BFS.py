from util import Queue
from game import GameStateData
from game import Actions


def BFS(xy1, xy2, gameStateData):
	q = Queue()
	q.push(xy1)
	seen = {xy1: None}
	done = False
	while not q.isEmpty() and not done:
		nextPoint = q.pop()
		neighbors = Actions.getLegalNeighbors(nextPoint,gameStateData.layout.walls)
		for neighbor in neighbors:
			if neighbor not in seen:
				q.push(neighbor)
				seen[neighbor] = nextPoint
				if neighbor == xy2:
					done = True
					break
	if q.isEmpty():
		print("BFS error: could not reach point " + str(xy2) + " from " + str(xy1))
		return []
	cur = xy2
	path = []
	while cur is not None:
		path.insert(0, cur)
		cur = seen[cur]
	return path


def coinGrouping(xy, gameStateData):
	q = Queue()
	q.push(xy)
	seen = {xy: None}
	coinGroups = {}
	while not q.isEmpty():
		nextPoint = q.pop()
		neighbors = Actions.getLegalNeighbors(nextPoint,gameStateData.layout.walls)
		for neighbor in neighbors:
			if neighbor not in seen:
				q.push(neighbor)
				seen[neighbor] = nextPoint
				if [neighbor[0]][neighbor[1]]:
					# we have a coin at neighbor
					coinGroups[neighbor] = coinGroup(neighbor, gameStateData)
					for coin in coinGroups[neighbor]:
						seen[coin] = neighbor
	return coinGroups


					
	
# given a coin location, adds all coins connected to it to a set.
def coinGroup(xy, gameStateData):
	q = Queue()
	q.push(xy)
	seen = set((xy))
	while not q.isEmpty():
		nextPoint = q.pop()
		neighbors = Actions.getLegalNeighbors(nextPoint,gameStateData.layout.walls)
		for neighbor in neighbors:
			if neighbor not in seen:
				seen.add(nextPoint)
				if gameStateData.getFood()[neighbor[0]][neighbor[1]]:
					q.push(neighbor)
	return seen

	





