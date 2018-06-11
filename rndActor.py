import random as rnd

class RandomActor:
    def act(self, gamestate, valid_moves):
        return rnd.choice(valid_moves)
