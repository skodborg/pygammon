import game
import sys

class interactiveActor:
    def __init__(self):
        pass

    def act(self, gamestate, valid_moves):
        game.print_game_state(gamestate)
        # print("\033c")  # clear screen, no scrollback
        while(True):
            cmd = input()
            try:
              if cmd == 'q':
                sys.exit()
              else:
                from_pos, to_pos = cmd.split(',')

                if not from_pos in ['bar']:
                  from_pos = int(from_pos)
                if not to_pos in ['end']:
                  to_pos = int(to_pos)
                if not (from_pos, to_pos) in valid_moves:
                    print('move %i,%i is not valid, try again' % (from_pos, to_pos))
                    print('valid moves are %s' % str(valid_moves))
                    continue
                return (from_pos, to_pos)
            except:
                sys.exit(1)
