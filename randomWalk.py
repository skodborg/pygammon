import random as rnd
import math

class RandomWalk:

	def __init__(self, state_seed=None):
		self.MOVE_LEFT = 0
		self.MOVE_RIGHT = 1

		self.NO_OF_STATES = 7
		self.LAST_STATE = self.NO_OF_STATES - 1
		self.FIRST_STATE = 0
		# states = [T_0, A, B, C, D, E, T_1]
		self.state = math.floor(self.NO_OF_STATES / 2)
		if state_seed:
			self.state = state_seed
		self.moves = 0

	def move_right(self):
		if self.state == self.LAST_STATE:
			return False
		self.state += 1
		self.moves += 1

	def move_left(self):
		if self.state == 0:
			return False
		self.state -= 1
		self.moves += 1

	def done(self):
		reward = 1 if self.state == self.LAST_STATE else 0
		done = True if self.state in [self.FIRST_STATE, self.LAST_STATE] else False
		return done, [[reward]], self.state, self.moves

	def getState(self):
		return self.state

	def state_space(self):
		return list(range(self.NO_OF_STATES))

	def move_space(self):
		return [self.MOVE_LEFT, self.MOVE_RIGHT]

	def move_sample(self):
		return rnd.choice([self.MOVE_LEFT, self.MOVE_RIGHT])

	def move(self, direction):
		if direction == self.MOVE_LEFT:
			self.move_left()
		elif direction == self.MOVE_RIGHT:
			self.move_right()
		else:
			raise ValueError('Unrecognized move')
		return self.done()

	def nn_state_repr(self):
		state_repr = [0 for _ in range(self.NO_OF_STATES)]
		state_repr[self.state] = 1
		return [state_repr]


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
