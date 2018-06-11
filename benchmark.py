import randomWalk as rw
import rndActor

def benchmark(rounds=1000):
    actor = rndActor.RandomActor()

    sum_moves = 0
    sum_reward = 0
    for i in range(rounds):
        print(i)
        walk = rw.RandomWalk()
        done, _, _, _ = walk.done()
        while not done:
            done, reward, _, moves = walk.move(actor.act(walk.state_space(), walk.move_space()))
        sum_moves += moves
        sum_reward += reward

    print('avg moves: %f' % (sum_moves/1000))
    print('avg reward: %f' % (sum_reward/1000))


def main():
    benchmark()

if __name__ == '__main__':
    main()