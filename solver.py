from viability_check import check_viable_numbers


class Box:

    def __init__(self, xcord, ycord):
        self.xcord = xcord
        self.ycord = ycord
        self.number = None
        self.viable_numbers = set()
        self.block = False

    def __str__(self):
        return self.number

    def __repr__(self):
        return str(int(self.number))

    def change_number(self, x, board):
        if x is None:
            self.block = False
            self.number = x
        else:
            if x in check_viable_numbers(self, board):
                self.block = True
                self.number = x


def create_board_from_numpy(numpy_board):
    board = create_sudoku_board()
    for row in range(numpy_board.shape[0]):
        for col in range(numpy_board.shape[1]):
            if numpy_board[row, col] != 0:
                board[row][col].change_number(numpy_board[row, col], board)
    return board


def create_sudoku_board(rows=9):
    board = []
    for _ in range(rows):
        board.append([])
    for x in range(rows):
        for y in range(rows):
            board[x].append(Box(x, y))
    return board


def solve(numpy_board, rows=9):
    board = create_board_from_numpy(numpy_board)
    run = True
    x, y = determine_first(board, rows)
    while run:
        # check result
        if y == -1:
            break
        elif y == rows:
            break
        box = board[x][y]
        if box.number is None:
            box.viable_numbers = check_viable_numbers(box, board, rows)
        if len(box.viable_numbers) == 0:
            x, y = go_backwards(x, y, board)
            continue
        else:
            if box.number is None:
                box.number = get_first_from_set(box.viable_numbers)
                x, y = go_forward(x, y, board)
                continue
            else:
                box.viable_numbers.remove(box.number)
                box.number = None
                if len(box.viable_numbers) == 0:
                    x, y = go_backwards(x, y, board)
                    continue
                else:
                    box.number = get_first_from_set(box.viable_numbers)
                    x, y = go_forward(x, y, board)
                    continue
    return board


def go_forward(x, y, board):
    if x == 8:
        new_x = 0
        new_y = y + 1
    else:
        new_x = x + 1
        new_y = y
    try:
        if not board[new_x][new_y].block:
            return new_x, new_y
        else:
            return go_forward(new_x, new_y, board)
    except Exception:
        return 0, 9


def go_backwards(x, y, board):
    if x == 0:
        new_x = 8
        new_y = y - 1
    else:
        new_x = x - 1
        new_y = y
    if not board[new_x][new_y].block:
        return new_x, new_y
    else:
        return go_backwards(new_x, new_y, board)


def get_first_from_set(s):
    for e in s:
        break
    return e


def determine_first(board, rows):
    for y in range(rows):
        for x in range(rows):
            if not board[x][y].block:
                return x, y
    return rows - 1, rows - 1
