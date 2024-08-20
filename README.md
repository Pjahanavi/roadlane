# roadlane detection
import cv2
import numpy as np

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines):
    img = np.copy(img)
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(blank_image, (x1, y1), (x2, y2), (255, 0, 0), thickness=10)

    img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
    return img

def process_image(image):
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply Gaussian Blur
    blur_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    
    # Edge detection using Canny
    canny_image = cv2.Canny(blur_image, 50, 150)
    
    # Define region of interest
    height = image.shape[0]
    polygons = np.array([
        [(200, height), (1100, height), (550, 250)]
    ])
    cropped_image = region_of_interest(canny_image, polygons)
    
    # Hough Transform to detect lines
    lines = cv2.HoughLinesP(cropped_image, 
                            rho=2, 
                            theta=np.pi/180, 
                            threshold=100, 
                            lines=np.array([]), 
                            minLineLength=40, 
                            maxLineGap=5)
    
    # Draw the lines on the original image
    image_with_lines = draw_lines(image, lines)
    
    return image_with_lines

# Load the video feed or a test image
cap = cv2.VideoCapture('test_video.mp4')

while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break
    
    processed_frame = process_image(frame)
    
    # Display the result
    cv2.imshow("Lane Detection", processed_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
