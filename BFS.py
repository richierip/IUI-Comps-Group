from util import Queue
from game import GameStateData
from game import Actions


def BFS(xy1, xy2, gameStateData):
	q = Queue()
	q.push(xy1)
	seen = {xy1: None}
	while not q.isEmpty():
		nextPoint = q.pop()
		neighbors = Actions.getLegalNeighbors(nextPoint,gameStateData.layout.walls)
		for neighbor in neighbors:
			if neighbor not in seen:
				q.push(neighbor)
				seen[neighbor] = nextPoint
				if neighbor == xy2:
					break
	if xy2 not in q:
		print("BFS error: could not reach point " + str(xy2) + " from " + str(xy1))
		return []
	cur = xy2
	path = []
	while cur is not None:
		path.insert(0, cur)
		cur = seen[cur]
	return path






