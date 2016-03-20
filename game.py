import random as rnd

BAR = 'bar'
WHITE_END = 25
BLACK_END = 0

board = []
bar = []
players = {-1:'black', 1:'white'}
player_in_turn = None
curr_roll = None

# returns ({die1: uses}, {die2: uses})
def roll_dice(fixed_roll=None):
  if fixed_roll is not None:
    return fixed_roll
  eyes = []
  for _ in range(2):
    eyes.append(rnd.randint(1,6))
  # each die is usable twice if they match
  uses = 1 if eyes[0] != eyes[1] else 2
  return tuple([{eye: uses} for eye in eyes])

def initialize_game(starting_player=None, init_roll=None, init_board=None):
  global board, player_in_turn, curr_roll, bar
  # 24 board positions, 4x6 triangles
  board = [0 for _ in range(24)]
  # white is positive, black is negative, int is no. of pieces at a given
  # board position, indexed clockwise starting in lower left corner
  board[0]  =  2
  board[5]  = -5
  board[7]  = -3
  board[11] =  5
  board[12] = -5
  board[16] =  3
  board[18] =  5
  board[23] = -2

  # initializing the bar with zero of each piece in it (white is listed first)
  bar = [0, 0]

  # starting player and initial dice roll chosen randomly
  curr_roll = roll_dice()
  # the two dice rolls cannot have matching eyes, keep rolling until different
  while (list(curr_roll[0].keys())[0] is list(curr_roll[1].keys())[0]):
    curr_roll = roll_dice()

  player_in_turn = list(players)[rnd.randint(0,1)]
  if starting_player is not None:
    player_in_turn = 1 if starting_player is 'white' else -1
  if init_roll is not None:
    curr_roll = init_roll
  if init_board is not None:
    board = init_board
  
def initialize_testgame(starting_player=None, init_roll=None):
  global board, bar
  
  initialize_game(starting_player=starting_player, init_roll=init_roll)

  # 24 board positions, 4x6 triangles
  board = [0 for _ in range(24)]
  # white is positive, black is negative, int is no. of pieces at a given
  # board position, indexed clockwise starting in lower left corner
  board[21] =  1
  board[22] =  0

  # initializing the bar with zero of each piece in it (white is listed first)
  bar = [0, 0]


# returns list of (from,to)-tuples of valid moves for white
def white_valid_moves(dice_rolls):

  # locations of all white pieces to select from
  wht_locations = []
  for pos, pieces in enumerate(board):
    if pieces > 0: wht_locations.append(pos+1)

  # result list, filled with tuples of (from,to)-positions of valid moves
  wht_possible_moves = []

  # handle pieces on the bar
  if bar[0] > 0:
    for roll in dice_rolls:
      eyes = list(roll.keys())[0]
      if board[eyes - 1] >= -1:
        wht_possible_moves.append((BAR, eyes))
    return list(set(wht_possible_moves))

  # handle pieces leaving the board in end-game 
  # (no piece positioned earlier than 19th edge, i.e. white end-zone)
  if wht_locations and min(wht_locations) >= 19:
    for roll in dice_rolls:
      eyes = list(roll.keys())[0]
      if board[24-eyes] > 0:
        wht_possible_moves.append((25-eyes, WHITE_END))
    if not wht_possible_moves:
      # not possible to take any pieces off the board, even though they're
      # all in home region. Take backmost piece and move as much as possible
      # (capped at 25 if we have more eyes than we need to take piece off board)
      for roll in dice_rolls:
        eyes = list(roll.keys())[0]
        target = min(min(wht_locations) + eyes, 25)
        wht_possible_moves.append((min(wht_locations), target))

    # remove duplicates and return as list of tuples
    return list(set(wht_possible_moves))

  # handle pieces when bar is empty and not all pieces are home yet
  for loc in wht_locations:
    for roll in dice_rolls:
      eyes = list(roll.keys())[0]
      uses = roll[eyes]

      # if piece at 'loc' is moved 'eyes', and resulting loc is within board
      if len(board) > loc+eyes-1:

        # if the target is unoccupied or only occupied by white pieces already
        # or occupied by a single opponent (allow 'hitting' the opponent piece)
        if -1 <= board[loc+eyes-1]:
          wht_possible_moves.append((loc, loc+eyes))

  return list(set(wht_possible_moves))


# mutates board state
def move_piece(from_pos, to_pos):
  global player_in_turn, curr_roll
  # check if move is valid
  # - if so, perform it, deduct from dice usages
  #   - if no dices left for use, swap player in turn
  # - check if a winner is found

  # TODO: handle double rolls
  # TODO: handle 'spending' rolls (i.e. when hitting 2-5, you can move one piece 2, another 5)
  # TODO:  - you can also move the same piece, first 5 then 2 or the other way around
  # TODO: handle 'hitting' an opponent
  # TODO: handle pulling pieces off the board for winning

  # TODO: handle moving pieces from the bar (from_pos = 'wb'/'bb')

  pieces = board[from_pos-1]
  # if pieces == 0:
  #   report_error("There are no pieces to move at position %i" % from_pos)
  #   return False
  # else:
  #   if from_pos == to_pos:
  #     report_error("Cannot move piece to the same position")
  #     return False

  sign = -1 if pieces < 0 else 1
  #   if player_in_turn is not players[sign]:
  #     report_error("Attempted to move opponents' pieces")
  #     return False
  #   if player_in_turn == players[1] and from_pos > to_pos or \
  #      player_in_turn == players[-1] and from_pos < to_pos:
  #     report_error("Cannot move %s piece backwards" % player_in_turn)
  #     return False

  # TODO: if no more moves (dice usages all 0) swap player and roll again

  board[from_pos-1] -= sign
  board[to_pos-1] += sign

  player_in_turn = -sign
  curr_roll = roll_dice()

  return True

def report_error(msg):
  print('ERROR: ' + msg)
  
def print_game_state():
  # black: x  white: o
  str_board = " 13  14  15  16  17  18         19  20  21  22  23  24\n|"
  for r in range(5):
    for i, k in enumerate(board[12:]):
      if i == 6:
        # if the bar has black pieces on it, paint them now, else paint nothing
        if bar[1] > 0 and r == 4:
          str_board += "  +%i  |" % bar[1]
        else:
          str_board += "      |"
      if r == 4 and (k > 5 or k < -5):
        str_board += "+%i |" % (abs(k)-5 + 1)
      elif abs(k) > r:
        str_board += " x |" if k < 0 else " o |"
      else:
        str_board += "   |"
    str_board += "\n|" if r < 4 else "\n"
  str_board += "--------------------------------------------------------\n|"
  for r in range(4,-1,-1):
    for i, k in reversed(list(enumerate(board[:12]))):
      if i == 5:
        # if the bar has white pieces on it, paint them now, else paint nothing
        if bar[0] > 0 and r == 4:
          str_board += "  +%i  |" % bar[0]
        else:
          str_board += "      |"
      if r == 0 and (k > 5 or k < -5):
        str_board += "+%i |" % (abs(k)-5 + 1)
      elif abs(k) > r:
        str_board += " x |" if k < 0 else " o |"
      else:
        str_board += "   |"
    str_board += "\n|" if r > 0 else "\n"
  str_board += " 12  11  10   9   8   7          6   5   4   3   2   1  "
  print(str_board)
  print("\nPlayer currently in turn: %s" % players[player_in_turn])
  print("Remaining roll: " + str(curr_roll))

def read_eval_loop():
  global curr_roll

  while True:
    cmd = input()
    if cmd == "q":
      break
    elif cmd == "r":
      curr_roll = roll_dice()
      print(curr_roll)
    else:
      from_pos, to_pos = cmd.split(',')
      move_piece(int(from_pos), int(to_pos))
      print_game_state()

def main():
  init_roll = roll_dice(fixed_roll=({3:1},{6:1}))
  initialize_game(starting_player='white', init_roll=init_roll)
  # initialize_testgame('white', init_roll)
  print_game_state()
  print(white_valid_moves(init_roll))
  read_eval_loop()


if __name__ == '__main__':
  main()