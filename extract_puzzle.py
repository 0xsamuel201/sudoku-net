import cv2
import numpy as np
import matplotlib.pyplot as plt
import onnxruntime as ort

def order_points(pts):
    """
    Helper function to order points in the following order: Top-Left, Top-Right, Bottom-Right, Bottom-Left.
    This is crucial for the perspective transform to work correctly.
    """
    rect = np.zeros((4, 2), dtype="float32")
    
    # calculate the sum of the points (x + y): the top-left point will have the smallest sum, whereas the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # calculate the difference of the points (y - x): the top-right point will have the smallest difference, whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

def extract_board(image_path):
    """
    Function takes an image path, detects the Sudoku board, and returns the original image, thresholded image, warped (top-down) image, and the contour of the board.
    """
    # 1. read the image & preprocess
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found at: {image_path}")
        
    original = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # blur and Noise reduction
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # adaptive thresholding: helps to highlight the edges even under uneven lighting conditions
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # 2. find contours and detect the largest 4-corner contour
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # sort contours by area and keep the largest one (which should be the Sudoku board)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    board_contour = None

    for c in contours:
        # calculate the perimeter of the contour
        peri = cv2.arcLength(c, True)
        # approximate the contour to a polygon and check if it has 4 points (indicating a quadrilateral)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        
        # if polygon has 4 points, we can assume it's the Sudoku board
        if len(approx) == 4:
            board_contour = approx
            break

    if board_contour is None:
        raise Exception("cannot found Sudoku board contour")

    # 3. perspective transform to get a top-down view of the board
    # get the 4 points of the board and order them
    pts = board_contour.reshape(4, 2)
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width and height of the new image based on the distances between the points
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # define the destination points for the perspective transform to get a top-down view of the board
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    # calculate the perspective transform matrix and apply it to get the warped image
    M = cv2.getPerspectiveTransform(rect, dst)
    # use gray image for the warped output since we will use it for digit recognition model later
    warped = cv2.warpPerspective(gray, M, (maxWidth, maxHeight))

    return original, thresh, warped, board_contour

def visualize_board_extract_results(original, thresh, warped, board_contour):
    """
    function to visualize the results: original image with contour, thresholded image, and warped top-down view of the board.
    """
    # draw the contour on the original image to visualize the detected board
    img_with_contour = original.copy()
    cv2.drawContours(img_with_contour, [board_contour], -1, (0, 0, 255), 3)

    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title("1. Original & Contour")
    plt.imshow(cv2.cvtColor(img_with_contour, cv2.COLOR_BGR2RGB))
    
    plt.subplot(1, 3, 2)
    plt.title("2. Adaptive Threshold")
    plt.imshow(thresh, cmap='gray')
    
    plt.subplot(1, 3, 3)
    plt.title("3. Warped Top-Down View")
    plt.imshow(warped, cmap='gray')
    
    plt.tight_layout()
    plt.show()

def preprocess_cell(cell_img):
    """
    preprocess advanced: robust noise filtering and center the digit in a MNIST-style 28x28 canvas
    """
    h, w = cell_img.shape
    
    # 1. Dynamic Cropping
    # cut 15% margin from each side to remove grid lines and focus on the central area where the digit is likely to be
    crop_margin_x = int(w * 0.15)
    crop_margin_y = int(h * 0.15)
    cell_cropped = cell_img[crop_margin_y:h-crop_margin_y, crop_margin_x:w-crop_margin_x]
    
    # 2. find contours to reduce noise and focus on the actual digit strokes
    contours, _ = cv2.findContours(cell_cropped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # if no contours are found, it means there is no digit in this cell, so we can return None to indicate an empty cell
    if not contours:
        return None
        
    # get the largest contour by area, which should correspond to the digit strokes, while smaller contours are likely to be noise or grid line remnants
    largest_contour = max(contours, key=cv2.contourArea)
    
    # if the area of the largest contour is too small, we can assume it's just noise and not a valid digit, so we return None to indicate an empty cell
    if cv2.contourArea(largest_contour) < 30: 
        return None

    # 3. center align the digit in a 28x28 canvas, which is the standard input size for MNIST-trained models. 
    # This involves cropping the digit tightly and then resizing it while maintaining aspect ratio, and finally placing it in the center of a 28x28 black canvas.
    # get the rectangle bounding box of the largest contour to crop the digit tightly
    x, y, w_box, h_box = cv2.boundingRect(largest_contour)
    
    # if the bounding box is too narrow or too short, just skip
    if w_box < 5 or h_box < 10:
        return None
        
    digit_roi = cell_cropped[y:y+h_box, x:x+w_box]

    # init a black canvas of size 28x28 to place the digit in the center
    mnist_canvas = np.zeros((28, 28), dtype=np.uint8)
    
    # max size of the digit should be 20x20 to fit in the 28x28 canvas with some margin
    if h_box > w_box:
        new_h = 20
        new_w = max(1, int(w_box * (20.0 / h_box)))
    else:
        new_w = 20
        new_h = max(1, int(h_box * (20.0 / w_box)))
        
    resized_digit = cv2.resize(digit_roi, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # calculate the starting x and y coordinates to place the resized digit in the center of the 28x28 canvas
    start_x = (28 - new_w) // 2
    start_y = (28 - new_h) // 2
    mnist_canvas[start_y:start_y+new_h, start_x:start_x+new_w] = resized_digit

    # 4. normalize before feeding to classifier model
    img_float = mnist_canvas.astype(np.float32) / 255.0
    img_normalized = (img_float - 0.1307) / 0.3081
    tensor_input = img_normalized.reshape(1, 1, 28, 28)
    
    return tensor_input

def extract_and_recognize(warped_img_path, onnx_model_path):
    """
    extract 81 cells from the warped image and recognize digits using the ONNX model
    """
    print("[*] loading warped Sudoku image...")
    # Read the image in grayscale mode since the model expects single channel input
    img = cv2.imread(warped_img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("cannot find warped image at: " + warped_img_path)

    # binarize and invert the image so that digits are white on black background, which matches the MNIST training data
    _, thresh = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # get width and height of the warped image to calculate cell size
    height, width = thresh.shape
    cell_h = height // 9
    cell_w = width // 9

    print("[*] loading ONNX MiniMobileNet...")
    ort_session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
    input_name = ort_session.get_inputs()[0].name

    # initialize an empty 9x9 board to store the recognized digits
    board = np.zeros((9, 9), dtype=int)

    print("[*] starting digit recognition...")
    for i in range(9):
        for j in range(9):
            # get cell [i, j] from the thresholded image
            cell = thresh[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
            
            tensor_input = preprocess_cell(cell)
            
            if tensor_input is not None:
                # onnx inference
                output = ort_session.run(None, {input_name: tensor_input})[0]
                
                # get highest probability digit
                predicted_digit = np.argmax(output, axis=1)[0]
                board[i][j] = predicted_digit

    return board

def print_board(board):
    """
    print extracted sudoku board to console
    """
    print("\n" + "="*25)
    print("   SUDOKU (9x9) matrix   ")
    print("="*25)
    for i in range(9):
        if i % 3 == 0 and i != 0:
            print("-" * 21)
        
        row_str = ""
        for j in range(9):
            if j % 3 == 0 and j != 0:
                row_str += "| "
            
            val = board[i][j]
            row_str += str(val) + " " if val != 0 else ". "
        
        print(row_str)
    print("="*25 + "\n")

if __name__ == "__main__":
    # prepare a test image path (replace this with your own image path)
    TEST_IMAGE = "./images/mobile_game_sample.jpg" 
    WARPED_IMAGE = "./images/warped_mobile_game_board.jpg"
    ONNX_MODEL = "./output/sudoku-net.onnx"
    
    try:
        print("[*] processing...")
        original, thresh, warped, contour = extract_board(TEST_IMAGE)
        print("[+] Processing completed successfully! Displaying results...")
        visualize_board_extract_results(original, thresh, warped, contour)
        
        # save the warped image for the next step (digit recognition)
        cv2.imwrite(WARPED_IMAGE, warped)
        print(f"[+] Saved warped board image as '{WARPED_IMAGE}'")

        sudoku_matrix = extract_and_recognize(WARPED_IMAGE, ONNX_MODEL)
        print_board(sudoku_matrix)
        
    except Exception as e:
        print(f"[-] Error: {e}")