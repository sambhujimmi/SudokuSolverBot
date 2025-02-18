import numpy as np

# sudoku size
n = 9
# box size
m = 3

def check(sudoku, row, col, val):
    # row and col
    for i in range(n):
        if sudoku[row][i] == val or sudoku[i][col] == val:
            return False
    # box
    boxrow = (row//m)*m
    boxcol = (col//m)*m
    for i in range(boxrow, boxrow+m):
        for j in range(boxcol, boxcol+m):
            if sudoku[i][j] == val:
                return False
    return True

def possibles(sudoku, i, j):
    possible_nums = []
    for v in range(1, n+1):
        if check(sudoku, i, j, v):
            possible_nums.append(v)
    return possible_nums

def solver(sudoku):
    empty = {}
    for i in range(8, -1, -1):
        for j in range(8, -1, -1):
            if sudoku[i][j] == 0:
                possible_nums = possibles(sudoku, i, j)
                if len(possible_nums) < 1:
                    print("\nImpossible to solve\n")
                elif len(possible_nums) == 1:
                    sudoku[i][j] = possible_nums[0]
                else:
                    empty[(i, j)] = possible_nums
    # backtracking
    pos = [x for x in empty]
    
    i = 0
    while i < len(pos):
        row = pos[i][0]
        col = pos[i][1]
        val = sudoku[row][col]
        possible_nums = empty[pos[i]]
        if val == 0:
            num_index = 0
        elif val == possible_nums[-1]:
            sudoku[row][col] = 0
            i -= 1
            continue
        else:
            num_index = possible_nums.index(val)+1
        
        for j in range(num_index, len(possible_nums)):
            num = possible_nums[j]
            if check(sudoku, row, col, num):
                sudoku[row][col] = num
                i += 1
                break
            elif j == len(possible_nums)-1:
                sudoku[row][col] = 0
                i -= 1
    return np.array(sudoku)