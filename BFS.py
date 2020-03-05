from util import Queue
import game


# TODO bfs fails with half distances when ghosts run away
def BFS(xy1, xy2, gameStateData):
	q = Queue()
	q.push(xy1)
	seen = {xy1: None}
	done = False
	if xy1 == xy2:
		return []
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
	seencoins = set()
	coinGroups = {}
	while not q.isEmpty():
		nextPoint = q.pop()
		if (nextPoint in food or nextPoint in capsules) and (nextPoint not in seencoins):
			# we have a coin at nextPoint
			coinGroups[nextPoint] = coinGroup(nextPoint, gameStateData)
			for coin in coinGroups[nextPoint]:
				seencoins.add(coin)

		seen.add(nextPoint)
		neighbors = game.Actions.getLegalNeighbors(nextPoint, gameStateData.getWalls())

		for neighbor in neighbors:
			if neighbor not in seen:
				q.push(neighbor)
				
	''' # testing purposes
	print("printing groups")
	testset = set()
	for coins in coinGroups.keys():
		for coin in coinGroups[coins]:
			if coin in testset:
				raise(Error("found coin in 2 groups"))
		print("closest: " + str(coins) + ", stuff: " + str(len(coinGroups[coins])))
	print("done printing groups")
	'''
	return coinGroups


					
	
# given a coin location, adds all coins connected to it to a set.
def coinGroup(xy, gameStateData):
	food = gameStateData.getFood().asList()
	capsules = gameStateData.getCapsules()
	q = Queue()
	q.push(xy)
	seen = set()
	seen.add(xy)
	seenCoin = set()
	seenCoin.add(xy)
	while not q.isEmpty():
		nextPoint = q.pop()
		neighbors = game.Actions.getLegalNeighbors(nextPoint, gameStateData.getWalls())
		for neighbor in neighbors:
			if neighbor not in seen:
				seen.add(neighbor)
				if neighbor in food or neighbor in capsules:
					seenCoin.add(neighbor)
					q.push(neighbor)
	return seenCoin

# returns the closest 3 coinGroups
# format: [(distance , numCoins), (distance1 : numCoins1)]
# if less than 3, returns only first 2 or 1.
def coinGroup3s(xy, gameStateData):
	food = gameStateData.getFood().asList()
	capsules = gameStateData.getCapsules()
	q = Queue()
	q.push(xy)
	seen = set()
	seencoins = set()
	coinGroups = {}
	while not q.isEmpty() and len(coinGroups) < 3:
		nextPoint = q.pop()
		if (nextPoint in food or nextPoint in capsules) and (nextPoint not in seencoins):
			# we have a coin at nextPoint
			coinGroups[nextPoint] = coinGroup(nextPoint, gameStateData)
			for coin in coinGroups[nextPoint]:
				seencoins.add(coin)

		seen.add(nextPoint)
		neighbors = game.Actions.getLegalNeighbors(nextPoint, gameStateData.getWalls())

		for neighbor in neighbors:
			if neighbor not in seen:
				q.push(neighbor)
	rlist = []
	for key in coinGroups:
		rlist.append((len(BFS(xy,key, gameStateData)), len(coinGroups[key])))
	
	return rdict





