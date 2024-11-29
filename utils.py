import PIL
from PIL import Image
import PIL.ImageOps as ops
from pillow_heif import register_heif_opener
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt


# Register HEIF opener to work with the HEIC format
register_heif_opener()

# Helper functions
def load_and_grayscale(image_path: str):
    img = Image.open(image_path).convert('L')
    return img


def resize_image(img, size=(28, 28)):
    img = img.resize(size, Image.LANCZOS)
    return img


def normalize_and_threshold(img):
    img_array = np.asarray(img, dtype=np.float32)
    img_array = img_array / 255.0
    img_array = np.where(img_array < 0.48, 0, 255)
    return img_array


def invert_image(img):
    return ops.invert(img)


def flatten_tensor(img_tensor):
    return img_tensor.reshape((784))


# Load a sample x_train vector and normalize
x_train_0 = torch.load('test_images/x_train_0.pt')
x_train_0 = x_train_0 / 255.0

# Load the local test image and process
img = load_and_grayscale('test_images/IMG_4296.HEIC')
img = resize_image(img)
img = invert_image(img)
img = normalize_and_threshold(img)

img_tensor = torch.tensor(img, dtype=torch.float)
print("Original shape: ", img_tensor.shape)
img_tensor = flatten_tensor(img_tensor)
print("After reshaping: ", img_tensor.shape)
torch.save(img_tensor, 'test_images/img.pt')

# Create subplots
fig, axs = plt.subplots(2, 2)

# Show the images
axs[0, 0].imshow(x_train_0.reshape((28, 28)), cmap='Greys', interpolation='none')
axs[0, 1].imshow(img, cmap='Greys', interpolation='none')
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