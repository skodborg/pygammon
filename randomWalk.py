class RandomWalk:
	def __init__(self):
		# states = [T_0, A, B, C, D, E, T_1]
		self.state = 3
		self.moves = 0

	def move_right(self):
		if self.state == 6:
			return False
		self.state += 1
		self.moves += 1

	def move_left(self):
		if self.state == 0:
			return False
		self.state -= 1
		self.moves += 1

	def done(self):
		reward = 1 if self.state == 6 else 0
		done = True if self.state in [0, 6] else False
		return done, reward, self.state, self.moves

	def getState(self):
		return self.state

	def state_space(self):
		return list(range(7))

	def moves(self):
		return self.moves


def main():
	rw = RandomWalk()
	print(rw.done())
	rw.move_right()
	print(rw.done())
	rw.move_right()
	print(rw.done())
	rw.move_right()
	print(rw.done())

	print('-' * 10)
	
	rw = RandomWalk()
	print(rw.done())
	rw.move_left()
	print(rw.done())
	rw.move_left()
	print(rw.done())
	rw.move_left()
	print(rw.done())

if __name__ == '__main__':
	main()
