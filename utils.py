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
    if result_2 is not None and isinstance(result_2, torch.Tensor):
        result_2 = result_2.detach().cpu().numpy()

    # Squeeze to remove extra single dimensions, such as batch size 1
    result_1 = np.squeeze(result_1)
    if result_2 is not None:
        result_2 = np.squeeze(result_2)

    # Transpose tensors from (channels, height, width) to (height, width, channels)
    if result_1.ndim == 3 and result_1.shape[0] in [1, 3]:  # (channels, height, width)
        result_1 = np.transpose(result_1, (1, 2, 0))
    if result_2 is not None and result_2.ndim == 3 and result_2.shape[0] in [1, 3]:
        result_2 = np.transpose(result_2, (1, 2, 0))

    # If result_2 exists, concatenate along the width (axis=1)
    if result_2 is not None:
        cat_image = np.concatenate([result_1, result_2], axis=1)
    else:
        cat_image = result_1

    # If grayscale (single channel), add a channel dimension for PIL compatibility
    if cat_image.ndim == 2:  # For pure grayscale images
        cat_image = np.expand_dims(cat_image, axis=-1)

    # Convert to uint8 by scaling if necessary
    cat_image = np.clip(cat_image * 255.0, 0, 255.0).astype(np.uint8)

    # Squeeze again if it's a grayscale image (shape: height, width, 1)
    if cat_image.shape[-1] == 1:
        cat_image = np.squeeze(cat_image, axis=-1)

    # Convert the NumPy array to a PIL Image and save it
    im = Image.fromarray(cat_image)
    im.save(filepath, 'png')

    print(f"Image saved to {filepath}")