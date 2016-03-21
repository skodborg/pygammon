import random as rnd

BAR = 'bar'
WHITE_END = 25
BLACK_END = 0

board = []
bar = []
players = {-1:'black', 1:'white'}
player_in_turn = None
curr_roll = None

def sign_matches(x, y):
  return x > 0 and y > 0 or x < 0 and y < 0

def roll_dice(fixed_roll=None):
  """ returns tuple ({die1: uses}, {die2: uses}) """
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
  # 24+2 board positions, 4x6 triangles and two positions where pieces are
  # considered taken off the board. These are 0 for black, and 25 for white
  board = [0 for _ in range(26)]
  # white is positive, black is negative, int is no. of pieces at a given
  # board position, indexed clockwise starting in lower left corner
  board[1]  =  2
  board[6]  = -5
  board[8]  = -3
  board[12] =  5
  board[13] = -5
  board[17] =  3
  board[19] =  5
  board[24] = -2

  # initializing the bar with zero of each piece in it (white is listed first)
  bar = [0, 0]

  # starting player and initial dice roll chosen randomly
  curr_roll = roll_dice()
  # the two dice rolls cannot have matching eyes, keep rolling until different
  while (list(curr_roll[0].keys())[0] is list(curr_roll[1].keys())[0]):
    curr_roll = roll_dice()

  player_in_turn = list(players.values())[rnd.randint(0,1)]
  if starting_player is not None:
    # player_in_turn = 1 if starting_player is 'white' else -1
    player_in_turn = starting_player
  if init_roll is not None:
    curr_roll = init_roll
  if init_board is not None:
    board = init_board
  
def initialize_testgame():
  global board, bar, curr_roll, player_in_turn
  
  initialize_game()

  board = [0 for _ in range(26)]
  board[3] =  -5
  board[2] =  -5
  board[1] =   2
  player_in_turn = 'black'
  curr_roll = ({6:1}, {5:1})
  
  bar = [0, 0]


def white_valid_moves(dice_rolls):
  """ returns list of (from,to)-tuples of valid moves for white """
  usable_rolls = []
  for roll in dice_rolls:
    if list(roll.values())[0] > 0:
      usable_rolls.append(roll)

  # locations of all white pieces to select from
  wht_locations = []
  for pos, pieces in enumerate(board):
    if pieces > 0: wht_locations.append(pos)

  # result list, filled with tuples of (from,to)-positions of valid moves
  wht_possible_moves = []

  # handle pieces on the bar
  if bar[0] > 0:
    for roll in usable_rolls:
      eyes = list(roll.keys())[0]
      if board[eyes] >= -1:
        wht_possible_moves.append((BAR, eyes))

    # remove duplicates and return as list of tuples
    return list(set(wht_possible_moves))

  # handle pieces leaving the board in end-game 
  # (no piece positioned earlier than 19th edge, i.e. white home-zone)
  if wht_locations and min(wht_locations) >= 19:
    for roll in usable_rolls:
      eyes = list(roll.keys())[0]
      if board[25-eyes] > 0:
        wht_possible_moves.append((25-eyes, WHITE_END))
    if not wht_possible_moves:
      # not possible to take any pieces off the board, even though they're
      # all in home region. Take backmost piece and move as much as possible
      # (capped at 25 if we have more eyes than we need to take piece off board)
      for roll in usable_rolls:
        eyes = list(roll.keys())[0]
        target = min(min(wht_locations) + eyes, 25)
        wht_possible_moves.append((min(wht_locations), target))

    # remove duplicates and return as list of tuples
    return list(set(wht_possible_moves))

  # handle pieces when bar is empty and not all pieces are home yet
  for loc in wht_locations:
    for roll in usable_rolls:
      eyes = list(roll.keys())[0]
      uses = roll[eyes]

      # if piece at 'loc' is moved 'eyes', and resulting loc is within board
      last_board_pos = 24
      if last_board_pos >= loc+eyes-1:

        # if the target is unoccupied or only occupied by white pieces already
        # or occupied by a single opponent (allow 'hitting' the opponent piece)
        if -1 <= board[loc+eyes]:
          wht_possible_moves.append((loc, loc+eyes))
  
  # remove duplicates and return as list of tuples
  return list(set(wht_possible_moves))

def black_valid_moves(dice_rolls):
  """ returns list of (from,to)-tuples of valid moves for black """
  usable_rolls = []
  for roll in dice_rolls:
    if list(roll.values())[0] > 0:
      usable_rolls.append(roll)

  # locations of all white pieces to select from
  blk_locations = []
  for pos, pieces in enumerate(board):
    if pieces < 0: blk_locations.append(pos)

  # result list, filled with tuples of (from,to)-positions of valid moves
  blk_possible_moves = []

  # handle pieces on the bar
  if bar[1] > 0:
    for roll in usable_rolls:
      eyes = list(roll.keys())[0]
      target = 25-eyes
      if board[target] <= 1:
        blk_possible_moves.append((BAR, target))

    # remove duplicates and return as list of tuples
    return list(set(blk_possible_moves))

  # handle pieces leaving the board in end-game 
  # (no piece positioned earlier than 6th edge, i.e. black home-zone)
  if blk_locations and max(blk_locations) <= 6:
    for roll in usable_rolls:
      eyes = list(roll.keys())[0]
      if board[eyes] < 0:
        blk_possible_moves.append((eyes, BLACK_END))
    if not blk_possible_moves:
      # not possible to take any pieces off the board, even though they're
      # all in home region. Take backmost piece and move as much as possible
      # (capped at 0 if we have more eyes than we need to take piece off board)
      for roll in usable_rolls:
        eyes = list(roll.keys())[0]
        target = max(max(blk_locations) - eyes, 0)
        blk_possible_moves.append((max(blk_locations), target))

    # remove duplicates and return as list of tuples
    return list(set(blk_possible_moves))

  # handle pieces when bar is empty and not all pieces are home yet
  for loc in blk_locations:
    for roll in usable_rolls:
      eyes = list(roll.keys())[0]
      uses = roll[eyes]

      # if piece at 'loc' is moved 'eyes', and resulting loc is within board
      first_board_pos = 1
      if first_board_pos <= loc-eyes:

        # if the target is unoccupied or only occupied by black pieces already
        # or occupied by a single opponent (allow 'hitting' the opponent piece)
        if 1 >= board[loc-eyes]:
          blk_possible_moves.append((loc, loc-eyes))
  
  # remove duplicates and return as list of tuples
  return list(set(blk_possible_moves))

def end_current_turn():
  global player_in_turn, curr_roll, players
  sign = 1 if player_in_turn == 'white' else -1
  player_in_turn = players[-sign]
  curr_roll = roll_dice()

# mutates board state
def move_piece(from_pos, to_pos):
  global player_in_turn, curr_roll, board, bar

  # TODO: handle pulling pieces off the board for winning
  # TODO: handle proper skipping of turns when no move is available
  # TODO: handle finding a winner
  # TODO: refactor?

  try:
    from_pos = int(from_pos)
  except ValueError:
    pass

  try:
    to_pos = int(to_pos)
  except ValueError:
    pass

  if from_pos == 'bar':
    pieces = bar[0] if player_in_turn is 'white' else -bar[1]
  else:
    pieces = board[from_pos]
  
  if pieces == 0:
    report_error("There are no pieces to move at position " + str(from_pos))
    return False
  if from_pos == to_pos:
    report_error("Cannot move piece to the same position")
    return False

  sign = -1 if pieces < 0 else 1
  if player_in_turn is not players[sign]:
    report_error("Attempted to move opponents' pieces")
    return False

  # there is a piece at the from_pos, it belongs to the player_in_turn, and
  # the move is not trivial; check if valid
  move = (from_pos, to_pos)
  valid_moves = []
  if player_in_turn == 'white':
    valid_moves = white_valid_moves(curr_roll)
  else:
    valid_moves = black_valid_moves(curr_roll)

  if move not in valid_moves:
    report_error("Move is not valid: " + str(move))
    return False

  # TODO: if no more moves (dice usages all 0) swap player and roll again
  curr_roll_eyes = [list(roll.keys())[0] for roll in curr_roll]
  # curr_roll_uses = [list(roll.values())[0] for roll in curr_roll]

  if from_pos == 'bar':
    eyes_used = to_pos if to_pos < 10 else 25 - to_pos
  else:
    eyes_used = abs(to_pos - from_pos)

  found_dice = False
  for i, eyes in enumerate(curr_roll_eyes):
    if eyes == eyes_used:
      if not curr_roll[i][eyes] == 0:
        # found a dice; decrement its uses
        curr_roll[i][eyes] -= 1
        found_dice = True
        break
  if not found_dice and to_pos in [0, 25]:
    # in case pieces are leaving the board, look for one with eyes larger
    # than what is needed for the piece to leave the board
    for i, eyes in enumerate(curr_roll_eyes):
      if eyes >= eyes_used:
        if not curr_roll[i][eyes] == 0:
          # found a dice; decrement its uses
          curr_roll[i][eyes] -= 1
          found_dice = True
          break
    
  if not found_dice:
    # found no unused dice left to enable the move; report error and return
    report_error("Trying to reuse die roll %i already spent" % eyes)
    return False

  # move is valid, perform it
  if from_pos == 'bar':
    if sign > 0:
      bar[0] -= 1
    else:
      bar[1] -= 1
  else:
    board[from_pos] -= sign
  
  if board[to_pos] is not 0 and not sign_matches(board[to_pos], sign):
    # an opponent's piece has been hit; handle placement on the bar
    if board[to_pos] > 0 and sign > 0 or board[to_pos] < 0 and sign < 0:
      # moving piece from bar to already friendly occupied position
      board[to_pos] += sign
    else:
      # moving piece from bar to position and hitting an opponent here
      board[to_pos] += sign * 2
    if sign > 0:
      bar[1] += 1  # white is in turn; black has been hit
    else:
      bar[0] += 1  # black is in turn; white has been hit
  else:
    board[to_pos] += sign

  # end_current_turn()
  return True

def report_error(msg):
  print('ERROR: ' + msg)
  
def print_game_state():
  # black: x  white: o
  str_board = "\n\n 13  14  15  16  17  18         19  20  21  22  23  24\n|"
  for r in range(5):
    for i, k in enumerate(board[13:25]):
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
    for i, k in reversed(list(enumerate(board[1:13]))):
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
  print("\nPlayer currently in turn: %s" % player_in_turn)
  print("Remaining roll: " + str(curr_roll))
  if player_in_turn == 'white':
    print(white_valid_moves(curr_roll))
  else:
    print(black_valid_moves(curr_roll))

def read_eval_loop():
  global curr_roll

  while True:
    cmd = input()
    if cmd == "q":
      break
    elif cmd == "r":
      curr_roll = roll_dice()
      print(curr_roll)
    elif cmd == 'skip':
      end_current_turn()
      print_game_state()
    else:
      try:
        from_pos, to_pos = cmd.split(',')

        if not from_pos in ['bar']:
          int(from_pos)
        if not to_pos in ['end']:
          int(to_pos)

        move_piece(from_pos, to_pos)

        curr_roll_uses = [list(roll.values())[0] for roll in curr_roll]
        if max(curr_roll_uses) == 0:
          end_current_turn()

        print_game_state()
      except ValueError:
        report_error("Unknown command: %s" % cmd)


def main():
  # initialize_game()
  initialize_testgame()
  print_game_state()
  read_eval_loop()


if __name__ == '__main__':
  main()