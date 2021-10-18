digits = {1, 2, 3, 4, 5, 6, 7, 8, 9}


def check_viable_numbers(box, board, rows=9):
    A = get_row_numbers(box, board, rows)
    B = get_col_numbers(box, board, rows)
    C = get_square_numbers(box, board)
    return digits.difference(A).difference(B).difference(C)


def get_row_numbers(box, board, rows):
    x = box.xcord
    y = box.ycord
    nmbrs = set()
    for i in range(rows):
        if i != x:
            if board[i][y] is not None:
                nmbrs.add(board[i][y].number)
    return nmbrs


def get_col_numbers(box, board, rows):
    x = box.xcord
    y = box.ycord
    nmbrs = set()
    for i in range(rows):
        if i != y:
            if board[x][i] is not None:
                nmbrs.add(board[x][i].number)
    return nmbrs


def get_square_numbers(box, board):
    x = box.xcord
    y = box.ycord
    return get_numbers_from_square(x // 3, y // 3, board, box)


def get_numbers_from_square(squarex, squarey, board, box):
    nmbrs = set()
    for x in range(squarex * 3, squarex * 3 + 3, 1):
        for y in range(squarey * 3, squarey * 3 + 3, 1):
            nmbrs.add(board[x][y].number)
    if box.number is not None:
        nmbrs.remove(box.number)
    return nmbrs
