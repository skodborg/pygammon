import random as rnd

board = []
players = {-1:'black', 1:'white'}
player_in_turn = None

def roll_dices(k=2):
  eyes = []
  for _ in range(0,k):
    eyes.append(rnd.randint(1,6))
  return eyes

def initialize_game(starting_player=None):
  global board, player_in_turn
  # 24 board positions, 4x6 triangles
  board = [0 for _ in range(24)]
  # white is positive, black is negative, int is no. of pieces at a given
  # board position, indexed clockwise starting in lower left corner
  board[0] = 2
  board[5] = -5
  board[7] = -3
  board[11] = 5
  board[12] = -5
  board[16] = 3
  board[18] = 5
  board[23] = -2

  # starting player chosen randomly
  if starting_player is None:
    player_in_turn = list(players)[rnd.randint(0,1)]
  else:
    player_in_turn = starting_player

def move_piece(from_pos, to_pos):
  # mutates board state; return true if valid move was performed, false if not
  pieces = board[from_pos-1]
  if pieces == 0:
    print("There are no pieces to move at position %i" % from_pos)
    return False
  else:
    if from_pos == to_pos:
      print("Cannot move piece to the same position")
      return False

    sign = -1 if pieces < 0 else 1
    if player_in_turn is not players[sign]:
      print("Attempted to move opponents' pieces")
      return False
    if player_in_turn == players[1] and from_pos > to_pos or \
       player_in_turn == players[-1] and from_pos < to_pos:
       # TODO: FIX FOR BLACK
      print("Cannot move %s piece backwards" % player_in_turn)
      return False

    board[from_pos-1] -= sign
    board[to_pos-1] += sign

  return True
  
def print_game_state():
  # black: x  white: o
  str_board = " 13  14  15  16  17  18         19  20  21  22  23  24\n|"
  for r in range(5):
    for i, k in enumerate(board[12:]):
      if i == 6:
        str_board += "      |"
      if abs(k) > r:
        str_board += " x |" if k < 0 else " o |"
      else:
        str_board += "   |"
    str_board += "\n|" if r < 4 else "\n"
  str_board += "--------------------------------------------------------\n|"
  for r in range(4,-1,-1):
    for i, k in reversed(list(enumerate(board[:12]))):
      if i == 5:
        str_board += "      |"
      if abs(k) > r:
        str_board += " x |" if k < 0 else " o |"
      else:
        str_board += "   |"
    str_board += "\n|" if r > 0 else "\n"
  str_board += " 12  11  10   9   8   7          6   5   4   3   2   1  "
  print(str_board)
  print("\nPlayer currently in turn: %s" % player_in_turn)

def main():
  initialize_game('white')
  # move_piece(12,14)
  print_game_state()

if __name__ == '__main__':
  main()