import os
import cv2
import numpy as np

# Define the path to your dataset folder
dataset_folder = r"C:\Users\Muhammad Zeeshan\Documents\Python chilla with baba\open cv\Model_Iteration"  # Update this with the actual path to your dataset

# Define the input image dimensions
img_width, img_height = 512, 512  # Update to desired size

# Define the path to save preprocessed images
preprocessed_folder = r"C:\Users\Muhammad Zeeshan\Documents\Python chilla with baba\open cv\Preprocessed_Images3"  # Update this with the path where you want to save preprocessed images

# Create the directory for saving preprocessed images
if not os.path.exists(preprocessed_folder):
    os.makedirs(preprocessed_folder)

# Function to apply Gabor filter
def apply_gabor_filter(image, ksize=31, sigma=4.0, theta=np.pi / 4, lambd=10.0, gamma=0.5):
    gabor_kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, 0, ktype=cv2.CV_32F)
    filtered_img = cv2.filter2D(image, cv2.CV_8UC3, gabor_kernel)
    return filtered_img

# Loop through each class folder
for class_names in os.listdir(dataset_folder):
    class_path = os.path.join(dataset_folder, class_names)
    preprocessed_class_path = os.path.join(preprocessed_folder, class_names)
    
    # Create the directory for saving preprocessed images for this class
    if not os.path.exists(preprocessed_class_path):
        os.makedirs(preprocessed_class_path)
    
    # Get the list of image files in the class folder
    images = os.listdir(class_path)
    
    # Loop through each image
    for image_name in images:
        image_path = os.path.join(class_path, image_name)
        
        # Load and preprocess the image
        image = cv2.imread(image_path)
        image = cv2.resize(image, (img_width, img_height))
        
        # Extract green channel
        green_channel = image[:, :, 1]  # Green channel
        
        # Apply Gabor filter
        gabor_img = apply_gabor_filter(green_channel)
        
        # Convert the preprocessed image from BGR to HSV
        hsv_image = cv2.cvtColor(gabor_img, cv2.COLOR_BGR2HSV)
        
        # Save the preprocessed image
        preprocessed_image_path = os.path.join(preprocessed_class_path, image_name)
        cv2.imwrite(preprocessed_image_path, hsv_image)
        
        # Display the image (optional)
        cv2.imshow("Preprocessed Image", hsv_image)
        cv2.waitKey(0)  # Wait for a key press to proceed to the next image
        cv2.destroyAllWindows()

print("Preprocessing complete.")
