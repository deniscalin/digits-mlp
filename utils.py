import PIL
from PIL import Image, ImageFilter
import PIL.ImageOps as ops
import cv2
import imutils
from pillow_heif import register_heif_opener
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt


# Register HEIF opener to work with the HEIC format
register_heif_opener()


# Helper functions
def load_and_grayscale(image_path: str):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def threshold(img):
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return thresh


def find_contours(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print("Hierarchy: ", hierarchy)
    img_inverted_back = cv2.bitwise_not(img)
    if contours:
        # Combine contours into one
        comb_cont = np.vstack(contours)

        # Get bounding box
        x, y, w, h = cv2.boundingRect(comb_cont)

        # Calculate center of bounding box
        box_center_x = x + w // 2
        box_center_y = y + h // 2

        # Calculate center of image
        img_center_x = img_inverted_back.shape[1] // 2
        img_center_y = img_inverted_back.shape[0] // 2

        # Calculate translation amounts
        t_x = img_center_x - box_center_x
        t_y = img_center_y - box_center_y

        # Create translation matrix
        trans_matrix = np.float32([[1, 0, t_x], [0, 1, t_y]])

        # Get image dimensions
        img_height, img_width = img_inverted_back.shape[:2]

        # Apply translations to the img
        cent_img = cv2.warpAffine(img_inverted_back, trans_matrix, (img_width, img_height))

        # Draw bounding box on original and centered images
        cv2.rectangle(img_inverted_back, (x, y), (x + w, y + h), (0, 0, 0), 2)
        cv2.rectangle(cent_img, (x + t_x, y + t_y), (x + t_x + w, y + t_y + h), (0, 0, 0), 2)
        # cv2.drawContours(img_contours, contours, -1, (0, 0, 0), 1)
        cv2.imshow("Original img ", img_inverted_back)
        cv2.imshow("Centered img ", cent_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def center_image(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(contours)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        print("Box: ({x}, {y}, {w}, {h})")
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.imshow("The img with bounding boxes: ", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def resize_image(img, size=(28, 28)):
    img = cv2.resize(img, size)
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
    return img.filter(ImageFilter.GaussianBlur(radius=0.75))


def array_to_img(img):
    return Image.fromarray(img)


def img_to_array(img):
    return np.asarray(img, dtype=np.float32)


# Load a sample x_train vector and normalize
x_train_0 = torch.load('test_images/x_train_0.pt')
x_train_0 = x_train_0 / 255.0


def process_image(path):
    img = load_and_grayscale(path)
    img = threshold(img)
    img = cv2.bitwise_not(img)
    resized_image = resize_image(img)
    cv2.namedWindow("Resized Image: ")
    cv2.resizeWindow("Resized Image: ", 500, 500)
    cv2.imshow("Resized Image: ", resized_image)
    # cv2.imshow("Original Image: ", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def prepare_image(path):
    # Load the local test image and process
    img = load_and_grayscale(path)
    center_image(img)
    img = resize_image(img)
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


def display_images(img1: torch.tensor, img2 = None):
    """Display the images
    
    Args: 
        img1 | the first image pixels wrapped in a torch tensor
        img2 | the second image (optional)
    """
    # Create subplots
    fig, axs = plt.subplots(2, 2)

    # Show the image(s)
    axs[0, 0].imshow(img1.reshape((28, 28)), cmap='Greys', interpolation='none')
    if img2:
        axs[0, 1].imshow(img2, cmap='Greys', interpolation='none')
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