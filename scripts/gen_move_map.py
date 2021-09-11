def is_valid_square(x, y):
  if x >= 0 and x < 8 and y >= 0 and y < 8:
    return True
  else:
    return False

def gen_move_map(sq, move_count, imap):
  fx, fy = sq // 8, sq % 8
  print("MOVE_MAP_{} = {{".format(sq))
  # N
  for i in range(1,8):
    tx, ty = fx - i, fy
    if is_valid_square(tx, ty):
      print("  ({}, 0): {},".format(i * -1, move_count))
      imap.append((move_count, sq, "({}, 0)".format(i * -1)))
      move_count += 1
  # NE
  for i in range(1,8):
    tx, ty = fx - i, fy + i
    if is_valid_square(tx, ty):
      print("  ({}, {}): {},".format(i * -1, i, move_count))
      imap.append((move_count, sq, "({}, {})".format(i * -1, i)))
      move_count += 1
  # E
  for i in range(1,8):
    tx, ty = fx, fy + i
    if is_valid_square(tx, ty):
      print("  ( 0, {}): {},".format(i, move_count))
      imap.append((move_count, sq, "( 0, {})".format(i)))
      move_count += 1
  # SE
  for i in range(1,8):
    tx, ty = fx + i, fy + i
    if is_valid_square(tx, ty):
      print("  ( {}, {}): {},".format(i, i, move_count))
      imap.append((move_count, sq, "( {}, {})".format(i, i)))
      move_count += 1
  # S
  for i in range(1,8):
    tx, ty = fx + i, fy
    if is_valid_square(tx, ty):
      print("  ( {}, 0): {},".format(i, move_count))
      imap.append((move_count, sq, "( {}, 0)".format(i)))
      move_count += 1
  # SW
  for i in range(1,8):
    tx, ty = fx + i, fy - i
    if is_valid_square(tx, ty):
      print("  ( {},{}): {},".format(i, i * -1, move_count))
      imap.append((move_count, sq, "( {},{})".format(i, i * -1)))
      move_count += 1
  # W
  for i in range(1,8):
    tx, ty = fx, fy - i
    if is_valid_square(tx, ty):
      print("  ( 0,{}): {},".format(i * -1, move_count))
      imap.append((move_count, sq, "( 0,{})".format(i * -1)))
      move_count += 1
  # NW
  for i in range(1,8):
    tx, ty = fx - i, fy - i
    if is_valid_square(tx, ty):
      print("  ({},{}): {},".format(i * -1, i * -1, move_count))
      imap.append((move_count, sq, "({},{})".format(i * -1, i * -1)))
      move_count += 1
  # Knight Moves
  tx, ty = fx - 2, fy + 1
  if is_valid_square(tx, ty):
    print("  (-2, 1): {},".format(move_count))
    imap.append((move_count, sq, "(-2, 1)"))
    move_count += 1
  tx, ty = fx - 1, fy + 2
  if is_valid_square(tx, ty):
    print("  (-1, 2): {},".format(move_count))
    imap.append((move_count, sq, "(-1, 2)"))
    move_count += 1
  tx, ty = fx + 1, fy + 2
  if is_valid_square(tx, ty):
    print("  ( 1, 2): {},".format(move_count))
    imap.append((move_count, sq, "( 1, 2)"))
    move_count += 1
  tx, ty = fx + 2, fy + 1
  if is_valid_square(tx, ty):
    print("  ( 2, 1): {},".format(move_count))
    imap.append((move_count, sq, "( 2, 1)"))
    move_count += 1
  tx, ty = fx + 2, fy - 1
  if is_valid_square(tx, ty):
    print("  ( 2,-1): {},".format(move_count))
    imap.append((move_count, sq, "( 2,-1)"))
    move_count += 1
  tx, ty = fx + 1, fy - 2
  if is_valid_square(tx, ty):
    print("  ( 1,-2): {},".format(move_count))
    imap.append((move_count, sq, "( 1,-2)"))
    move_count += 1
  tx, ty = fx - 1, fy - 2
  if is_valid_square(tx, ty):
    print("  (-1,-2): {},".format(move_count))
    imap.append((move_count, sq, "(-1,-2)"))
    move_count += 1
  tx, ty = fx - 2, fy - 1
  if is_valid_square(tx, ty):
    print("  (-2,-1): {},".format(move_count))
    imap.append((move_count, sq, "(-2,-1)"))
    move_count += 1
  print("}")
  print("")
  return (move_count, imap)

def gen_move_map_map(imap):
  print("MOVE_MAP_MAP = {")
  for i in range(64):
    print("  {}: MOVE_MAP_{},".format(i,i))
  print("}")
  print("")
  print("INV_MOVE_MAP_MAP = {")
  for key, fsq, val in imap:
    print("  {}: ({}, {}),".format(key, fsq, val))
  print("  1792: (64, ( 0, 0)),")
  print("}")

def gen_none_move(move_count):
  print("NONE_MOVE_IDX = {}".format(move_count))
  print("")

move_count = 0
imap = []
for i in range(0, 64):
  move_count, imap = gen_move_map(i, move_count, imap)
gen_none_move(move_count)
gen_move_map_map(imap)
