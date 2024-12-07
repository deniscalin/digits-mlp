import cv2
import PIL
from PIL import Image, ImageFilter
import PIL.ImageOps as ops
from pillow_heif import register_heif_opener
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt


# Register HEIF opener to work with the HEIC format
register_heif_opener()


def process_image(path, target_size=(28, 28)):
    # Load the image
    img = cv2.imread(path)
    if img is None:
        print(f"Error: Could not read image at {path}")
        return None
    
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h, w = gray_img.shape
    aspect_ratio = float(w) / h
    print(f"INSIDE PROCESSING. height: {h}, width: {w}, aspect ratio: {aspect_ratio}")

    # Calculate new dimensions while keeping aspect ratio
    if aspect_ratio > 1: # wider than tall
        new_w = target_size[1]
        new_h = int(new_w / aspect_ratio)
    else: # taller than wide or square
        new_h = target_size[0]
        new_w = int(new_h * aspect_ratio)

    # Resize the image
    resized_img = cv2.resize(gray_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create a blank canvas for padding
    canvas = np.zeros(target_size, dtype=np.uint8)

    # Calculate padding to center resized image
    pad_h = (target_size[0] - new_h) // 2
    pad_w = (target_size[1] - new_w) // 2

    # Place the image onto the canvas with padding
    canvas[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized_img

    # Apply Gaussian Blur
    blurred_img = cv2.GaussianBlur(canvas, (5, 5), 0)

    # Apply thresholding for better contrast
    _, thresh_img = cv2.threshold(blurred_img, 127, 255, cv2.THRESH_BINARY_INV)

    cv2.imshow("Thresholded image: ", thresh_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



# Helper functions
def load_and_grayscale(image_path: str):
    img = Image.open(image_path).convert('L')
    return img


def resize_image(img, size=(28, 28)):
    img.thumbnail(size=size, resample=Image.Resampling.LANCZOS)
    img = img.resize(size, Image.Resampling.LANCZOS)
    return img


def normalize_and_threshold(img):
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    # print("PRINT FROM UTILS.N_AND_T. img_array: ", img_array)
    img_array = np.where(img_array < 0.2, 0, 255)
    img_array = np.asarray(img_array, dtype=np.uint8)
    return img_array


def invert_image(img):
    return ops.invert(img)


def flatten_tensor(img_tensor):
    return img_tensor.reshape((784))


def add_blur(img):
    return img.filter(ImageFilter.GaussianBlur(radius=0.70))

def array_to_img(img):
    return Image.fromarray(img)


def img_to_array(img):
    return np.asarray(img, dtype=np.float32)


# Load a sample x_train vector and normalize
x_train_0 = torch.load('test_images/x_train_0.pt')
x_train_0 = x_train_0 / 255.0

def prepare_image(path):
    # Load the local test image and process
    img = load_and_grayscale(path)
    img = resize_image(img)
    # img.show("Image after resizing")
    # input = input()
    # if input == 'q':
    #     import sys; sys.exit(0)
    img = invert_image(img)
    img = normalize_and_threshold(img)
    img = array_to_img(img)
    img = add_blur(img)
    img = img_to_array(img)
    

    img_tensor = torch.tensor(img, dtype=torch.float)
    print("Original shape: ", img_tensor.shape)
    img_tensor = flatten_tensor(img_tensor)
    print("After reshaping: ", img_tensor.shape)
    # torch.save(img_tensor, 'test_images/img.pt')
    return img_tensor


# Commenting out for testing
def display_images(img1: torch.tensor, img2 = None, img3 = None, img4 = None):
    """Display the images
    
    Args: 
        img1 | the first image pixels wrapped in a torch tensor
        img2 | the second image (optional)
    """
    # Create subplots
    fig, axs = plt.subplots(2, 2)

    # Show the image(s)
    axs[0, 0].imshow(img1.reshape((28, 28)), cmap='Greys', interpolation='none')
    if img2 != None:
        axs[0, 1].imshow(img2.reshape((28, 28)), cmap='Greys', interpolation='none')
    if img3 != None:
        axs[1, 0].imshow(img3.reshape((28, 28)), cmap='Greys', interpolation='none')
    if img4 != None:
        axs[1, 1].imshow(img4.reshape((28, 28)), cmap='Greys', interpolation='none')  
    plt.show()
    input = input()
    if input == 'q':
        import sys; sys.exit(0)






# img = load_and_convert('test_images/IMG_4296.HEIC')
# img = resize_image(img)
# img.show()
# input = input()
# if input == 'q':
#     import sys; sys.exit(0)