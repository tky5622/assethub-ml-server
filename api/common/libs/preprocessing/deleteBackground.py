import cv2
import numpy as np

# Function to detect person and create binary mask
def detect_person(image_path):
    # Load the image
    image = cv2.imread(image_path)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply a person detection algorithm (e.g., HOG-based person detector)
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    boxes, _ = hog.detectMultiScale(gray, winStride=(4, 4), padding=(8, 8), scale=1.05)
    
    # Create a binary mask for the person
    mask = np.zeros_like(gray)
    for (x, y, w, h) in boxes:
        cv2.rectangle(mask, (x, y), (x + w, y + h), (255), -1)
    
    # Create a 3-channel mask
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    # Combine the mask with the original image
    result = cv2.bitwise_and(image, mask_3ch)
    
    # Save the result with the person isolated
    result_path = image_path[:-4] + '_isolated.png'
    cv2.imwrite(result_path, result)

if __name__ == "__main__":
    image_path = 'images/sample_images/sample1.jpeg'
    detect_person(image_path)
