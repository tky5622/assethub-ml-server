import cv2

def extract_face(image_path):
    # Load the Haar cascade file for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Read the input image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        # Assuming only one face is present, extract the bounding box coordinates
        (x, y, w, h) = faces[0]

        # Extract the face ROI (Region of Interest)
        face_roi = image[y:y+h, x:x+w]

        # Save the extracted face as another image
        cv2.imwrite('extracted_face.jpg', face_roi)
        print("Face saved as extracted_face.jpg")

    else:
        print("No faces found in the image")

# Specify the path to the input image
input_image_path = 'transparent_images/sample3_transparent.png'

# Extract and save the face from the input image
extract_face(input_image_path)
