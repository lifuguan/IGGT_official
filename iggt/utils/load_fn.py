# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from PIL import Image
from torchvision import transforms as TF


def load_and_preprocess_images(image_path_list, mode="crop", resize_target_size=None):
    """
    A quick start function to load and preprocess images for model input.

    Args:
        image_path_list (list): List of paths to image files.
        mode (str, optional): Preprocessing mode, either "crop", "pad", or "resize".
        resize_target_size (tuple, optional): A tuple of (width, height) to resize images to.
                                               This is required and used only when mode is "resize".

    Returns:
        torch.Tensor: Batched tensor of preprocessed images with shape (N, 3, H, W).

    Raises:
        ValueError: If the input list is empty, mode is invalid, or if mode is "resize"
                    and resize_target_size is not a valid tuple of (width, height).

    Notes:
        - When mode="crop": The function ensures width=518px while maintaining aspect ratio
          and height is center-cropped if larger than 518px.
        - When mode="pad": The function ensures the largest dimension is 518px while maintaining aspect ratio
          and the smaller dimension is padded to reach a square shape (518x518).
        - When mode="resize": The function resizes the image to the specified resize_target_size,
          ignoring the original aspect ratio.
    """
    # Check for empty list
    if not image_path_list:
        raise ValueError("At least 1 image is required")

    # Validate mode
    if mode not in ["crop", "pad", "resize"]:
        raise ValueError("Mode must be either 'crop', 'pad', or 'resize'")

    # Validate resize_target_size if mode is "resize"
    if mode == "resize":
        if resize_target_size is None:
            raise ValueError("resize_target_size must be provided as a (width, height) tuple when mode is 'resize'")
        if not (isinstance(resize_target_size, (tuple, list)) and len(resize_target_size) == 2):
            raise ValueError("resize_target_size must be a tuple or list of two integers: (width, height)")

    images = []
    shapes = set()
    to_tensor = TF.ToTensor()
    # This target_size is now only for "crop" and "pad" modes
    default_target_size = 518

    # Process all images and collect their shapes
    for image_path in image_path_list:
        img = Image.open(image_path)

        if img.mode == "RGBA":
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(background, img)

        img = img.convert("RGB")
        width, height = img.size

        if mode == "pad":
            if width >= height:
                new_width = default_target_size
                new_height = round(height * (new_width / width) / 14) * 14
            else:
                new_height = default_target_size
                new_width = round(width * (new_height / height) / 14) * 14
        elif mode == "resize":
            # Unpack the user-provided target size
            new_width, new_height = resize_target_size
        else:  # mode == "crop"
            new_width = default_target_size
            new_height = round(height * (new_width / width) / 14) * 14

        # Resize with new dimensions
        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        img = to_tensor(img)

        # Post-processing for crop and pad modes
        if mode == "crop" and new_height > default_target_size:
            start_y = (new_height - default_target_size) // 2
            img = img[:, start_y : start_y + default_target_size, :]
        elif mode == "pad":
            h_padding = default_target_size - img.shape[1]
            w_padding = default_target_size - img.shape[2]
            if h_padding > 0 or w_padding > 0:
                pad_top, pad_left = h_padding // 2, w_padding // 2
                pad_bottom, pad_right = h_padding - pad_top, w_padding - pad_left
                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )

        shapes.add((img.shape[1], img.shape[2]))
        images.append(img)

    # Pad images to the same size if their shapes differ after processing
    if len(shapes) > 1:
        print(f"Warning: Found images with different shapes after processing: {shapes}")
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)

        padded_images = []
        for img in images:
            h_padding = max_height - img.shape[1]
            w_padding = max_width - img.shape[2]
            if h_padding > 0 or w_padding > 0:
                pad_top, pad_left = h_padding // 2, w_padding // 2
                pad_bottom, pad_right = h_padding - pad_top, w_padding - pad_left
                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )
            padded_images.append(img)
        images = padded_images

    images = torch.stack(images)
    
    if len(image_path_list) == 1 and images.dim() == 3:
        images = images.unsqueeze(0)

    return images