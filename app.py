import streamlit as st
import cv2
import numpy as np
import time
from PIL import Image

# Import functions from your custom pipeline
from extract_puzzle import extract_board, extract_and_recognize
from solver import solve_sudoku

ONNX_MODEL = "./weights/sudoku-net.onnx"

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

# Configure the Streamlit page
st.set_page_config(page_title="AI Sudoku Solver", page_icon="🧩", layout="wide")

st.title("🧩 Real-time Sudoku Solver")
st.markdown("""
This app combines **Computer Vision (OpenCV)**, a highly-optimized **MiniMobileNet (ONNX)** Deep Learning model, and a recursive **Backtracking** algorithm to solve real-world Sudoku puzzles from images in the blink of an eye!
""")

# File uploader widget
uploaded_file = st.file_uploader("Upload your Sudoku image here...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Split the screen into 2 columns for a side-by-side view
    col1, col2 = st.columns(2)
    
    # 1. Save the uploaded file temporarily to disk for OpenCV to read
    temp_input_path = "./output/temp_upload.jpg"
    with open(temp_input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    with col1:
        st.subheader("📷 Original Image")
        # Use PIL to display the image smoothly on the web interface
        st.image(Image.open(uploaded_file), use_container_width=True)
        
    # The main action button
    if st.button("🚀 Solve This Board!", type="primary"):
        with st.spinner('AI is scanning and algorithms are crunching...'):
            try:
                # STEP 1: DETECT & WARP BOARD
                _, _, warped_gray, _ = extract_board(temp_input_path)
                cv2.imwrite("./output/temp_warped.jpg", warped_gray)
                
                # STEP 2: DIGIT RECOGNITION (ONNX)
                original_board = extract_and_recognize("./output/temp_warped.jpg", ONNX_MODEL)
                
                # STEP 3: SOLVE SUDOKU
                solved_board = original_board.copy()
                start_time = time.time()
                is_solved = solve_sudoku(solved_board)
                end_time = time.time()
                
                if is_solved:
                    # STEP 4: RENDER SOLUTION
                    result_bgr = draw_solution_on_board(warped_gray, original_board, solved_board)
                    # Convert color space from BGR (OpenCV) to RGB (Streamlit)
                    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
                    
                    with col2:
                        st.subheader("✨ Result")
                        st.image(result_rgb, use_container_width=True)
                        
                    st.success(f"🎉 Successfully solved! Algorithm processing time: {end_time - start_time:.4f} seconds")
                    
                    # Optional: Show the raw matrix for the nerds
                    with st.expander("View Raw Number Matrix"):
                        st.write(solved_board)
                else:
                    st.error("No luck! Couldn't find a valid solution for this board. It might be invalid or misread.")
                    
            except Exception as e:
                st.error(f"An error occurred during processing: {e}")