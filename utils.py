import numpy as np
from PIL import Image
import torch 

def data_augmentation(image, mode):
    if mode == 0:
        # original
        return image
    elif mode == 1:
        # flip up and down
        return np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        return np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        image = np.rot90(image)
        return np.flipud(image)
    elif mode == 4:
        # rotate 180 degree
        return np.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        image = np.rot90(image, k=2)
        return np.flipud(image)
    elif mode == 6:
        # rotate 270 degree
        return np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        image = np.rot90(image, k=3)
        return np.flipud(image)

def load_images(file):
    im = Image.open(file)
    return np.array(im, dtype="float32") / 255.0


def save_images(filepath, result_1, result_2=None):
    # Detach tensors and convert to numpy if they are still PyTorch tensors
    if isinstance(result_1, torch.Tensor):
        result_1 = result_1.detach().cpu().numpy()
    if isinstance(result_2, torch.Tensor):
        result_2 = result_2.detach().cpu().numpy()

    # Squeeze dimensions to remove extra single dimensions
    result_1 = np.squeeze(result_1)

    if result_2 is not None:
        result_2 = np.squeeze(result_2)
        # Ensure both arrays have the same number of dimensions
        if result_1.ndim == 2:
            result_1 = np.expand_dims(result_1, axis=-1)  # Grayscale image
        if result_2.ndim == 2:
            result_2 = np.expand_dims(result_2, axis=-1)  # Grayscale image

        # Concatenate result_1 and result_2 if result_2 is provided
        cat_image = np.concatenate([result_1, result_2], axis=1)
    else:
        # Use only result_1 if result_2 is None
        cat_image = result_1

    # Ensure the image is in a valid format for PIL (Height x Width x Channels)
    if cat_image.ndim == 2:  # If it's still grayscale
        cat_image = np.expand_dims(cat_image, axis=-1)  # Add a channel dimension

    # Convert to uint8, clip values between 0 and 255, and save as PNG
    cat_image = np.clip(cat_image * 255.0, 0, 255.0).astype('uint8')

    # Check if the image has a single channel or 3 channels
    if cat_image.shape[-1] == 1:  # Grayscale
        cat_image = np.squeeze(cat_image, axis=-1)  # Remove the channel dimension for saving

    # Convert numpy array to image and save
    im = Image.fromarray(cat_image)
    im.save(filepath, 'png')
