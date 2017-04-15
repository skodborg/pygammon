import random as rnd
import copy

BAR = 'bar'
WHITE_END = 25
BLACK_END = 0

# board = []
# bar = []
players = {-1:'black', 1:'white'}
# player_in_turn = None
# curr_roll = None
winner = None

# global_gamestate = (board, bar, player_in_turn, curr_roll)
global_gamestate = None

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
  global global_gamestate
  # global board, player_in_turn, curr_roll, bar, global_gamestate
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

  global_gamestate = (board, bar, player_in_turn, curr_roll)

  
def initialize_testgame():
  # global board, bar, curr_roll, player_in_turn, winner, global_gamestate
  global global_gamestate
  
  initialize_game()

  board = [0 for _ in range(26)]
  board[1]  =  2
  board[6]  = -5
  board[8]  = -3
  board[12] =  5
  board[13] = -5
  board[17] =  3
  board[19] =  5
  board[24] = -2
  player_in_turn = 'white'
  curr_roll = ({2:1}, {6:1})
  
  bar = [0, 0]
  # winner = 'white'

  global_gamestate = (board, bar, player_in_turn, curr_roll)
  print(global_gamestate)

def white_valid_moves(gamestate):
  board, bar, player_in_turn, curr_roll = gamestate

  """ returns list of (from,to)-tuples of valid moves for white """
  usable_rolls = []
  for roll in curr_roll:
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
      if last_board_pos >= loc+eyes:

        # if the target is unoccupied or only occupied by white pieces already
        # or occupied by a single opponent (allow 'hitting' the opponent piece)
        if -1 <= board[loc+eyes]:
          wht_possible_moves.append((loc, loc+eyes))
  
  # remove duplicates and return as list of tuples
  return list(set(wht_possible_moves))

def black_valid_moves(gamestate):
  board, bar, player_in_turn, curr_roll = gamestate

  """ returns list of (from,to)-tuples of valid moves for black """
  usable_rolls = []
  for roll in curr_roll:
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
        if 1 >= board[loc-eyes]: #and loc - eyes > BLACK_END:
          blk_possible_moves.append((loc, loc-eyes))
  
  # remove duplicates and return as list of tuples
  return list(set(blk_possible_moves))

def end_current_turn(gamestate):
  # global player_in_turn, curr_roll, players
  global players
  board, bar, player_in_turn, curr_roll = gamestate
  sign = 1 if player_in_turn == 'white' else -1
  player_in_turn = players[-sign]
  curr_roll = roll_dice()
  gamestate = board, bar, player_in_turn, curr_roll
  return gamestate

# mutates board state
def move_piece(from_pos, to_pos, gamestate=global_gamestate):

  board, bar, player_in_turn, curr_roll = gamestate
  # TODO: FIX
  # if not (player_in_turn and curr_roll and board and bar):
    # global player_in_turn, curr_roll, board, bar
  # global player_in_turn, curr_roll, board, bar

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
    return None
  if from_pos == to_pos:
    report_error("Cannot move piece to the same position")
    return None

  sign = -1 if pieces < 0 else 1
  if player_in_turn is not players[sign]:
    report_error("Attempted to move opponents' pieces")
    return None

  # there is a piece at the from_pos, it belongs to the player_in_turn, and
  # the move is not trivial; check if valid
  move = (from_pos, to_pos)
  valid_moves = []
  if player_in_turn == 'white':
    valid_moves = white_valid_moves(gamestate)
  else:
    valid_moves = black_valid_moves(gamestate)

  if move not in valid_moves:
    report_error("Move is not valid: " + str(move))
    return None

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
    return None

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

  curr_roll_uses = [list(roll.values())[0] for roll in curr_roll]
    # print(curr_roll_uses)
  if max(curr_roll_uses) == 0:
    gamestate = end_current_turn((board, bar, player_in_turn, curr_roll))
    board, bar, player_in_turn, curr_roll = gamestate

  find_winner(gamestate)


  # end_current_turn()
  resulting_gamestate = (board, bar, player_in_turn, curr_roll)
  return resulting_gamestate

def report_error(msg):
  print('ERROR: ' + msg)
  
def print_game_state(gamestate):
  board, bar, player_in_turn, curr_roll = gamestate
  global winner
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
  if winner:
    print("\n%s wins!" % winner.capitalize())
    return
  print("\nPlayer currently in turn: %s" % player_in_turn)
  print("Remaining roll: " + str(curr_roll))
  if player_in_turn == 'white':
    print(white_valid_moves(gamestate))
  else:
    print(black_valid_moves(gamestate))

def find_winner(gamestate):
  """ sets global variable 'winner' if a player has won, i.e. has successfully
      taken all his pieces off of the board """
  global winner
  board = gamestate[0]

  black_has_won = True
  for pieces in board[1:24+1]:
    if pieces < 0: 
      black_has_won = False
      break
  if black_has_won:
    winner = 'black'

  white_has_won = True
  for pieces in board[1:24+1]:
    if pieces > 0: 
      white_has_won = False
      break
  if white_has_won:
    winner = 'white'  

def read_eval_loop(gamestate):
  # global curr_roll
  _, _, _, curr_roll = gamestate
  # TODO: SOMETHING'S FUCKED UP WITH THE GAMESTATE,
  #       END_CURRENT_TURN UPDATES AND RETURNS A DIFFERENT OBJECT
  while True:
    cmd = input()
    if cmd == "q":
      break
    elif cmd == "r":
      curr_roll = roll_dice()
      gamestate[3] = curr_roll
      print(curr_roll)
    elif cmd == 'skip':
      gamestate = end_current_turn(gamestate)
      print_game_state(gamestate)
    else:
      try:
        from_pos, to_pos = cmd.split(',')

        if not from_pos in ['bar']:
          int(from_pos)
        if not to_pos in ['end']:
          int(to_pos)

        print('before: %s' % str(curr_roll))
        gamestate = move_piece(from_pos, to_pos, gamestate)
        print('after: %s' % str(curr_roll))
        # global_gamestate = gamestate

        
        print_game_state(gamestate)
      except ValueError:
        report_error("Unknown command: %s" % cmd)

def nn_game_representation(gamestate):
    ''' returns a list with 198 elements. First 96 elements corresponds to
        white players pieces on the board, 4 elements to represent each of 
        the 24 positions on the board. The following 3 elements are the
        number of pieces white has on the bar, has removed from the board,
        and lastly a value indicating if it is white players turn.
        These values are followed by same values, but for black player.'''
    board, bar, player_in_turn, curr_roll = gamestate

    # values: n/2 for n pieces on the bar of each color
    bar_neurons = [0.0, 0.0]
    bar_neurons[0] = bar[0] / 2
    bar_neurons[1] = bar[1] / 2

    # values: n/15 for n pieces taken off the board of each color
    p_removed_neurons = [0, 0]
    p_removed_neurons[0] = board[25] / 15 # white
    p_removed_neurons[1] = board[0] / 15  # black

    # values: 0/1 whether white/blue is in turn or not
    pl_turn_neurons = [1.0, 0.0] if player_in_turn == 'white' else [0.0, 1.0]

    # 24 positions, 4 neurons to represent a position, 2 colors = 192 neurons
    # iterating the 192 neurons in the neural net used to represent the
    # black and white pieces on each of the 24 edges on the board
    neurons_per_pos = 4
    white_board_neurons = [0.0] * 24 * 4
    black_board_neurons = [0.0] * 24 * 4
    for pos in range(1, len(board) - 1):
        this_pos = board[pos]
        if this_pos == 0:
            # no pieces to update board representation with at this_pos
            continue
        elif this_pos > 0:
            # white
            neuron_pos = (pos - 1) * neurons_per_pos
            if this_pos > 0:
                white_board_neurons[neuron_pos] = 1.0
            if this_pos > 1:
                white_board_neurons[neuron_pos + 1] = 1.0
            if this_pos > 2:
                white_board_neurons[neuron_pos + 2] = 1.0
            if this_pos > 3:
                neuron_val = (this_pos - 3) / 2
                white_board_neurons[neuron_pos + 3] = neuron_val
        else:
            # black
            neuron_pos = (pos - 1) * neurons_per_pos
            if this_pos < 0:
                black_board_neurons[neuron_pos] = 1.0
            if this_pos < -1:
                black_board_neurons[neuron_pos + 1] = 1.0
            if this_pos < -2:
                black_board_neurons[neuron_pos + 2] = 1.0
            if this_pos < -3:
                neuron_val = (abs(this_pos) - 3) / 2
                black_board_neurons[neuron_pos + 3] = neuron_val
    board_neurons = white_board_neurons + black_board_neurons
    
    return board_neurons + bar_neurons + p_removed_neurons + pl_turn_neurons
    
def decide_move(gamestate):
    board, bar, player_in_turn, curr_roll = gamestate
    def eval_single_move(gamestate, move):
        # perform move
        from_pos, to_pos = move
        post_move_gamestate = move_piece(from_pos, to_pos, gamestate)
        print_game_state(post_move_gamestate)
        # eval board afterwards (nn)
        # return evaluation
        pass

    # should then, for each of the ~20 possible moves with the roll
    # throw the resulting afterstate through the neural network to 
    # estimate the value of a given board. Has to do this for ALL combinations
    # of rolls I guess?


    def play_out_game(gamestate):
      global winner
      curr_gamestate = gamestate
      while not winner:
        valid_moves = []
        if curr_gamestate[2] == 'white':
          valid_moves = white_valid_moves(curr_gamestate)
        else:
          valid_moves = black_valid_moves(curr_gamestate)

        if not valid_moves:
          print('NO VALID MOVES')
          curr_gamestate = end_current_turn(curr_gamestate)
          continue

        to_positions = [x[1] for x in valid_moves]
        from_positions = [x[0] for x in valid_moves]
        if 25 in to_positions:
          ok = True
          for from_pos in from_positions:
            if from_pos < 19:
              ok = False
          assert ok
        if 0 in to_positions:
          ok = True
          for from_pos in from_positions:
            if from_pos > 6:
              ok = False
          assert ok

        print('valid moves: %s' % str(valid_moves))
      # perform random move
        random_move = valid_moves[rnd.randint(0, len(valid_moves)-1)]
        print('moving from %s to %s' % (str(random_move[0]), str(random_move[1])))
        curr_gamestate = move_piece(random_move[0], random_move[1], curr_gamestate)
        print_game_state(curr_gamestate)

      print('a winner has been found in the following gamestate')
      print_game_state(curr_gamestate)


        # for move in valid_moves:
        #   gamestate_copy = copy.deepcopy(global_gamestate)
        #   from_pos = move[0]
        #   to_pos = move[1]

        #   curr_gamestate = move_piece(from_pos, to_pos, gamestate_copy)
          # print_game_state(returned_gamestate)
          # print(nn_game_representation(returned_gamestate))

      # evaluate returned_gamestate

    # print('\n\n ORIGINAL GAMESTATE')
    # print_game_state(global_gamestate)
    # should return the moves corresponding to the most likely to win
    
    play_out_game(global_gamestate)



def main():
  # initialize_game()
  initialize_testgame()

  print_game_state(global_gamestate)
  # read_eval_loop(global_gamestate)
  

  decide_move(global_gamestate)


if __name__ == '__main__':
  main()