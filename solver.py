import numpy as np

def is_valid(board, row, col, num):
    """
    Check if it's strictly valid to place 'num' at board[row][col].
    Rules of Sudoku:
    1. The number must not exist in the current row.
    2. The number must not exist in the current column.
    3. The number must not exist in the current 3x3 sub-grid.
    """
    # 1. Check the row
    for i in range(9):
        if board[row][i] == num:
            return False

    # 2. Check the column
    for i in range(9):
        if board[i][col] == num:
            return False

    # 3. Check the 3x3 sub-grid (box)
    start_row = (row // 3) * 3
    start_col = (col // 3) * 3
    for i in range(3):
        for j in range(3):
            if board[start_row + i][start_col + j] == num:
                return False

    # If it passes all checks, it's a valid placement (for now)
    return True

def solve_sudoku(board):
    """
    Main backtracking algorithm to solve the Sudoku puzzle.
    It modifies the 'board' matrix in-place.
    Returns True if a solution is found, False otherwise.
    """
    # Iterate through the entire board to find an empty cell (represented by 0)
    for row in range(9):
        for col in range(9):
            if board[row][col] == 0:
                # Found an empty cell. Try numbers 1 through 9.
                for num in range(1, 10):
                    if is_valid(board, row, col, num):
                        # Make a tentative guess
                        board[row][col] = num

                        # Recursively attempt to solve the rest of the board
                        if solve_sudoku(board):
                            return True # Solution found!

                        # If solve_sudoku returns False, our guess was wrong.
                        # UNDO the guess (this is the BACKTRACKING step)
                        board[row][col] = 0

                # If all numbers 1-9 fail in this empty cell, the puzzle is unsolvable 
                # from the current configuration. Trigger a backtrack.
                return False
                
    # If there are no empty cells left, the puzzle is solved!
    return True

def print_board(board, title="SUDOKU BOARD"):
    """
    Utility function to print the board in a clean, readable format.
    """
    print(f"\n--- {title} ---")
    for i in range(9):
        if i % 3 == 0 and i != 0:
            print("- - - - - - - - - - - - - ")

        for j in range(9):
            if j % 3 == 0 and j != 0:
                print(" | ", end="")

            if j == 8:
                print(board[i][j])
            else:
                print(str(board[i][j]) + " ", end="")
    print("-" * 25 + "\n")

if __name__ == "__main__":
    # Example board
    sample_board = [
        [0, 2, 0, 0, 0, 0, 0, 3, 0],
        [0, 0, 1, 7, 6, 0, 0, 0, 9],
        [0, 0, 0, 0, 0, 0, 5, 7, 0],
        [0, 0, 6, 5, 0, 0, 0, 0, 0],
        [0, 1, 8, 0, 3, 6, 0, 0, 0],
        [7, 0, 3, 0, 0, 0, 9, 6, 0],
        [0, 6, 2, 0, 0, 0, 7, 0, 3],
        [0, 8, 0, 0, 0, 3, 2, 0, 0],
        [0, 0, 0, 2, 9, 0, 6, 1, 5]
    ]

    # Convert to numpy array if it isn't already
    puzzle = np.array(sample_board)

    print_board(puzzle, "UNSOLVED PUZZLE")

    print("[*] Algorithm is thinking...")
    
    # Run the solver
    if solve_sudoku(puzzle):
        print("[+] SUCCESS! Puzzle solved.")
        print_board(puzzle, "SOLVED PUZZLE")
    else:
        print("[-] FAILED. No solution exists for this puzzle.")