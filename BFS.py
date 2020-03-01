from util import Queue
import game


# TODO bfs fails with half distances when ghosts run away
def BFS(xy1, xy2, gameStateData):
	q = Queue()
	q.push(xy1)
	seen = {xy1: None}
	done = False
	while not q.isEmpty() and not done:
		nextPoint = q.pop()
		neighbors = game.Actions.getLegalNeighbors(nextPoint, gameStateData.getWalls())
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

# {(x,y): set((x1,y1),(x2,y2)...) , (x1,y1) :}
def coinGrouping(xy, gameStateData):
	food = gameStateData.getFood().asList()
	capsules = gameStateData.getCapsules()
	q = Queue()
	q.push(xy)
	seen = set()
	coinGroups = {}
	while not q.isEmpty():
		nextPoint = q.pop()
		neighbors = game.Actions.getLegalNeighbors(nextPoint, gameStateData.getWalls())
		for neighbor in neighbors:
			if neighbor not in seen:
				q.push(neighbor)
				seen.add(neighbor)

				if food[neighbor[0]][neighbor[1]] or neighbor in capsules:
					# we have a coin at neighbor
					coinGroups[neighbor] = coinGroup(neighbor, gameStateData)
					for coin in coinGroups[neighbor]:
						seen.add(coin)
	return coinGroups


					
	
# given a coin location, adds all coins connected to it to a set.
def coinGroup(xy, gameStateData):
	food = gameStateData.getFood().asList()
	capsules = gameStateData.getCapsules()
	q = Queue()
	q.push(xy)
	seen = set(xy)
	seenCoin = set(xy)
	while not q.isEmpty():
		nextPoint = q.pop()
		neighbors = game.Actions.getLegalNeighbors(nextPoint, gameStateData.getWalls())
		for neighbor in neighbors:
			if neighbor not in seen:
				seen.add(neighbor)
				if food[neighbor[0]][neighbor[1]] or neighbor in capsules:
					seenCoin.add(neighbor)
					q.push(neighbor)
	return seen

	





