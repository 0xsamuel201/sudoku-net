import cv2
import onnxruntime as ort
import time

from extract_puzzle import extract_board, extract_and_recognize, print_board
from solver import solve_sudoku

def draw_solution_on_board(warped_img, original_board, solved_board):
    """
    Draws the Sudoku solution (numbers that were originally empty) onto the 
    warped image in a distinct color.
    """
    h, w = warped_img.shape
    cell_h = h // 9
    cell_w = w // 9
    
    # Convert grayscale warped image to BGR color image to draw colored numbers
    result_img = cv2.cvtColor(warped_img, cv2.COLOR_GRAY2BGR)
    
    # Define font properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0  # Adjust based on cell size/resolution
    font_color = (0, 0, 255)  # Red color (B, G, R) for the solution numbers
    font_thickness = 2
    
    print("[*] Drawing solution numbers on the image...")
    # Iterate through each cell
    for i in range(9):
        for j in range(9):
            # Only draw numbers for cells that were originally empty (represented by 0)
            if original_board[i][j] == 0:
                num_str = str(solved_board[i][j])
                
                # Calculate the position to center the number in the cell
                # 1. Get size of the number to be drawn
                (num_w, num_h), baseline = cv2.getTextSize(num_str, font, font_scale, font_thickness)
                
                # 2. Calculate coordinates of the number's origin (bottom-left) to center it
                #    Top-left corner of the cell is (j * cell_w, i * cell_h)
                #    Center of the cell is roughly ((j + 0.5) * cell_w, (i + 0.5) * cell_h)
                #    Adjust number origin slightly to center number text visually
                num_x = int((j + 0.5) * cell_w - num_w / 2)
                num_y = int((i + 0.5) * cell_h + num_h / 2)
                
                # Draw the number text on the image
                cv2.putText(result_img, num_str, (num_x, num_y), 
                            font, font_scale, font_color, font_thickness, cv2.LINE_AA)
                
    return result_img

def main():
    # File paths
    IMAGE_INPUT = "./images/mobile_game_sample.jpg" 
    WARPED_IMAGE_TEMP = "./images/warped_mobile_game_board.jpg"
    ONNX_MODEL = "./weights/sudoku-net.onnx"
    RESULT_IMAGE = "./output/result_sudoku.jpg" # Final output filename

    try:
        print("=== SUDOKU SOLVER WITH IMAGE OUTPUT ===")
        # STEP 1: Detect & Extract the Board ---
        _, _, warped_gray_img, _ = extract_board(IMAGE_INPUT)
        
        # Save warped image temporarily (recognizer function needs a filename)
        cv2.imwrite(WARPED_IMAGE_TEMP, warped_gray_img)
        print("[+] Board detected and warped.")

        # STEP 2: Recognize Digits ---
        # The recognizer handles full recognition, including ONNX session setup
        # Note: We pass the weight file just so the model object is initialized properly inside
        # although we actually use the ONNX model for inference. The previous function 
        # extract_and_recognize used this approach. I will assume that script structure here.
        recognized_board = extract_and_recognize(WARPED_IMAGE_TEMP, ONNX_MODEL)
        
        # Create a deep copy for the solution function (which modifies in-place)
        solved_board_copy = recognized_board.copy()
        
        # Print recognized board to terminal
        print_board(recognized_board)

        # STEP 3: Solve Sudoku ---
        print("[*] Solving the Sudoku puzzle...")
        start_time_solve = time.perf_counter()
        if solve_sudoku(solved_board_copy):
            end_time_solve = time.perf_counter()
            print(f"[+] Puzzle solved in {end_time_solve - start_time_solve:.4f} seconds.")
            print_board(solved_board_copy)
        else:
            print("[-] FAILED. No solution exists for this puzzle.")
            return

        # STEP 4: Draw Solution & Save Image ---
        # Call the drawing function with grayscale warped image, original board, and solution
        annotated_result_img = draw_solution_on_board(warped_gray_img, recognized_board, solved_board_copy)
        
        # Save the result image
        cv2.imwrite(RESULT_IMAGE, annotated_result_img)
        print(f"[+] Solution drawn successfully! Result saved to: {RESULT_IMAGE}")

    except Exception as e:
        print(f"[-] Error: {e}")

if __name__ == "__main__":
    main()