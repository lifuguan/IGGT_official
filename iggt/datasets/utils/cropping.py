# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# croppping utilities
# --------------------------------------------------------
import PIL.Image
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2  # noqa
import numpy as np  # noqa
import math
from iggt.utils.geometry import colmap_to_opencv_intrinsics, opencv_to_colmap_intrinsics  # noqa
try:
    lanczos = PIL.Image.Resampling.LANCZOS
    bicubic = PIL.Image.Resampling.BICUBIC
except AttributeError:
    lanczos = PIL.Image.LANCZOS
    bicubic = PIL.Image.BICUBIC


class ImageList:
    """ Convenience class to aply the same operation to a whole set of images.
    """

    def __init__(self, images):
        if not isinstance(images, (tuple, list, set)):
            images = [images]
        self.images = []
        for image in images:
            if not isinstance(image, PIL.Image.Image):
                image = PIL.Image.fromarray(image)
            self.images.append(image)

    def __len__(self):
        return len(self.images)

    def to_pil(self):
        return tuple(self.images) if len(self.images) > 1 else self.images[0]

    @property
    def size(self):
        sizes = [im.size for im in self.images]
        assert all(sizes[0] == s for s in sizes)
        return sizes[0]

    def resize(self, *args, **kwargs):
        return ImageList(self._dispatch('resize', *args, **kwargs))

    def crop(self, *args, **kwargs):
        return ImageList(self._dispatch('crop', *args, **kwargs))

    def _dispatch(self, func, *args, **kwargs):
        return [getattr(im, func)(*args, **kwargs) for im in self.images]


def rescale_image_depthmap(image, depthmap, camera_intrinsics, output_resolution, force=True):
    """ Jointly rescale a (image, depthmap) 
        so that (out_width, out_height) >= output_res
    """
    image = ImageList(image)
    input_resolution = np.array(image.size)  # (W,H)
    output_resolution = np.array(output_resolution)
    if depthmap is not None:
        # can also use this with masks instead of depthmaps
        assert tuple(depthmap.shape[:2]) == image.size[::-1]

    # define output resolution
    assert output_resolution.shape == (2,)
    scale_final = max(output_resolution / image.size) + 1e-8
    if scale_final >= 1 and not force:  # image is already smaller than what is asked
        return (image.to_pil(), depthmap, camera_intrinsics)
    output_resolution = np.floor(input_resolution * scale_final).astype(int)

    # first rescale the image so that it contains the crop
    image = image.resize(tuple(output_resolution), resample=lanczos if scale_final < 1 else bicubic)
    if depthmap is not None:
        depthmap = cv2.resize(depthmap, output_resolution, fx=scale_final,
                              fy=scale_final, interpolation=cv2.INTER_NEAREST)

    # no offset here; simple rescaling
    camera_intrinsics = camera_matrix_of_crop(
        camera_intrinsics, input_resolution, output_resolution, scaling=scale_final)

    return image.to_pil(), depthmap, camera_intrinsics

def center_crop_image_depthmap(image, depthmap, camera_intrinsics, crop_scale):
    """
    Jointly center-crop an image and its depthmap, and adjust the camera intrinsics accordingly.

    Parameters:
    - image: PIL.Image or similar, the input image.
    - depthmap: np.ndarray, the corresponding depth map.
    - camera_intrinsics: np.ndarray, the 3x3 camera intrinsics matrix.
    - crop_scale: float between 0 and 1, the fraction of the image to keep.

    Returns:
    - cropped_image: PIL.Image, the center-cropped image.
    - cropped_depthmap: np.ndarray, the center-cropped depth map.
    - adjusted_intrinsics: np.ndarray, the adjusted camera intrinsics matrix.
    """
    # Ensure crop_scale is valid
    assert 0 < crop_scale <= 1, "crop_scale must be between 0 and 1"

    # Convert image to ImageList for consistent processing
    image = ImageList(image)
    input_resolution = np.array(image.size)  # (width, height)
    if depthmap is not None:
        # Ensure depthmap matches the image size
        assert depthmap.shape[:2] == tuple(image.size[::-1]), "Depthmap size must match image size"

    # Compute output resolution after cropping
    output_resolution = np.floor(input_resolution * crop_scale).astype(int)
    # get the correct crop_scale
    crop_scale = output_resolution / input_resolution

    # Compute margins (amount to crop from each side)
    margins = input_resolution - output_resolution
    offset = margins / 2  # Since we are center cropping

    # Calculate the crop bounding box
    l, t = offset.astype(int)
    r = l + output_resolution[0]
    b = t + output_resolution[1]
    crop_bbox = (l, t, r, b)

    # Crop the image and depthmap
    image = image.crop(crop_bbox)
    if depthmap is not None:
        depthmap = depthmap[t:b, l:r]

    # Adjust the camera intrinsics
    adjusted_intrinsics = camera_intrinsics.copy()

    # Adjust focal lengths (fx, fy)                         # no need to adjust focal lengths for cropping
    # adjusted_intrinsics[0, 0] /= crop_scale[0]  # fx
    # adjusted_intrinsics[1, 1] /= crop_scale[1]  # fy

    # Adjust principal point (cx, cy)
    adjusted_intrinsics[0, 2] -= l  # cx
    adjusted_intrinsics[1, 2] -= t  # cy

    return image.to_pil(), depthmap, adjusted_intrinsics



def camera_matrix_of_crop(input_camera_matrix, input_resolution, output_resolution, scaling=1, offset_factor=0.5, offset=None):
    # Margins to offset the origin
    margins = np.asarray(input_resolution) * scaling - output_resolution
    assert np.all(margins >= 0.0)
    if offset is None:
        offset = offset_factor * margins

    # Generate new camera parameters
    output_camera_matrix_colmap = opencv_to_colmap_intrinsics(input_camera_matrix)
    output_camera_matrix_colmap[:2, :] *= scaling
    output_camera_matrix_colmap[:2, 2] -= offset
    output_camera_matrix = colmap_to_opencv_intrinsics(output_camera_matrix_colmap)

    return output_camera_matrix


def crop_image_depthmap(image, depthmap, camera_intrinsics, crop_bbox, mask=None):
    """
    Return a crop of the input view.
    """
    image = ImageList(image)
    l, t, r, b = crop_bbox

    image = image.crop((l, t, r, b))
    depthmap = depthmap[t:b, l:r]
    if mask is not None:
        mask = mask[t:b, l:r]
    camera_intrinsics = camera_intrinsics.copy()
    camera_intrinsics[0, 2] -= l
    camera_intrinsics[1, 2] -= t

    return image.to_pil(), depthmap, camera_intrinsics, mask


def bbox_from_intrinsics_in_out(input_camera_matrix, output_camera_matrix, output_resolution):
    out_width, out_height = output_resolution
    l, t = np.int32(np.round(input_camera_matrix[:2, 2] - output_camera_matrix[:2, 2]))
    crop_bbox = (l, t, l + out_width, t + out_height)
    return crop_bbox


def resize_image_depth_and_intrinsic(
    image,
    depth_map,
    intrinsic,
    target_shape,
    original_size,
    track=None,
    pixel_center=True,
    safe_bound=4,
    rescale_aug=True,
):
    """
    Resizes the given image and depth map (if provided) to slightly larger than `target_shape`,
    updating the intrinsic matrix (and track array if present). Optionally uses random rescaling
    to create some additional margin (based on `rescale_aug`).

    Steps:
      1. Compute a scaling factor so that the resized result is at least `target_shape + safe_bound`.
      2. Apply an optional triangular random factor if `rescale_aug=True`.
      3. Resize the image with LANCZOS if downscaling, BICUBIC if upscaling.
      4. Resize the depth map with nearest-neighbor.
      5. Update the camera intrinsic and track coordinates (if any).

    Args:
        image (np.ndarray):
            Input image array (H, W, 3).
        depth_map (np.ndarray or None):
            Depth map array (H, W), or None if unavailable.
        intrinsic (np.ndarray):
            Camera intrinsic matrix (3x3).
        target_shape (np.ndarray or tuple[int, int]):
            Desired final shape (height, width).
        original_size (np.ndarray or tuple[int, int]):
            Original size of the image in (height, width).
        track (np.ndarray or None):
            Optional (N, 2) array of pixel coordinates. Will be scaled.
        pixel_center (bool):
            If True, accounts for 0.5 pixel center shift during resizing.
        safe_bound (int or float):
            Additional margin (in pixels) to add to target_shape before resizing.
        rescale_aug (bool):
            If True, randomly increase the `safe_bound` within a certain range to simulate augmentation.

    Returns:
        tuple:
            (resized_image, resized_depth_map, updated_intrinsic, updated_track)

            - resized_image (np.ndarray): The resized image.
            - resized_depth_map (np.ndarray or None): The resized depth map.
            - updated_intrinsic (np.ndarray): Camera intrinsic updated for new resolution.
            - updated_track (np.ndarray or None): Track array updated or None if not provided.

    Raises:
        AssertionError:
            If the shapes of the resized image and depth map do not match.
    """
    if rescale_aug:
        random_boundary = np.random.triangular(0, 0, 0.3)
        safe_bound = safe_bound + random_boundary * target_shape.max()

    resize_scales = (target_shape + safe_bound) / original_size
    max_resize_scale = np.max(resize_scales)
    intrinsic = np.copy(intrinsic)

    # Convert image to PIL for resizing
    # image = PIL.Image.fromarray(image)
    input_resolution = np.array(image.size)
    output_resolution = np.floor(input_resolution * max_resize_scale).astype(int)
    image = image.resize(tuple(output_resolution), resample=lanczos if max_resize_scale < 1 else bicubic)
    image = np.array(image)

    if depth_map is not None:
        depth_map = cv2.resize(
            depth_map,
            output_resolution,
            fx=max_resize_scale,
            fy=max_resize_scale,
            interpolation=cv2.INTER_NEAREST,
        )

    actual_size = np.array(image.shape[:2])
    actual_resize_scale = np.max(actual_size / original_size)

    if pixel_center:
        intrinsic[0, 2] = intrinsic[0, 2] + 0.5
        intrinsic[1, 2] = intrinsic[1, 2] + 0.5

    intrinsic[:2, :] = intrinsic[:2, :] * actual_resize_scale

    if track is not None:
        track = track * actual_resize_scale

    if pixel_center:
        intrinsic[0, 2] = intrinsic[0, 2] - 0.5
        intrinsic[1, 2] = intrinsic[1, 2] - 0.5

    assert image.shape[:2] == depth_map.shape[:2]
    return image, depth_map, intrinsic, track


def crop_image_depth_and_intrinsic_by_pp(
    image, depth_map, intrinsic, target_shape, track=None, filepath=None, strict=False
):
    """
    TODO: some names of width and height seem not consistent. Need to check.
    
    
    Crops the given image and depth map around the camera's principal point, as defined by `intrinsic`.
    Specifically:
      - Ensures that the crop is centered on (cx, cy).
      - Optionally pads the image (and depth map) if `strict=True` and the result is smaller than `target_shape`.
      - Shifts the camera intrinsic matrix (and `track` if provided) accordingly.

    Args:
        image (np.ndarray):
            Input image array of shape (H, W, 3).
        depth_map (np.ndarray or None):
            Depth map array of shape (H, W), or None if not available.
        intrinsic (np.ndarray):
            Camera intrinsic matrix (3x3). The principal point is assumed to be at (intrinsic[1,2], intrinsic[0,2]).
        target_shape (tuple[int, int]):
            Desired output shape.
        track (np.ndarray or None):
            Optional array of shape (N, 2). Interpreted as (x, y) pixel coordinates. Will be shifted after cropping.
        filepath (str or None):
            An optional file path for debug logging (only used if strict mode triggers warnings).
        strict (bool):
            If True, will zero-pad to ensure the exact target_shape even if the cropped region is smaller.

    Raises:
        AssertionError:
            If the input image is smaller than `target_shape`.
        ValueError:
            If the cropped image is larger than `target_shape` (in strict mode), which should not normally happen.

    Returns:
        tuple:
            (cropped_image, cropped_depth_map, updated_intrinsic, updated_track)

            - cropped_image (np.ndarray): Cropped (and optionally padded) image.
            - cropped_depth_map (np.ndarray or None): Cropped (and optionally padded) depth map.
            - updated_intrinsic (np.ndarray): Intrinsic matrix adjusted for the crop.
            - updated_track (np.ndarray or None): Track array adjusted for the crop, or None if track was not provided.
    """
    original_size = np.array(image.shape)
    intrinsic = np.copy(intrinsic)

    if original_size[0] < target_shape[0]:
        error_message = (
            f"Width check failed: original width {original_size[0]} "
            f"is less than target width {target_shape[0]}."
        )
        print(error_message)
        raise AssertionError(error_message)

    if original_size[1] < target_shape[1]:
        error_message = (
            f"Height check failed: original height {original_size[1]} "
            f"is less than target height {target_shape[1]}."
        )
        print(error_message)
        raise AssertionError(error_message)

    # Identify principal point (cx, cy) from intrinsic
    cx = (intrinsic[1, 2])
    cy = (intrinsic[0, 2])

    # Compute how far we can crop in each direction
    if strict:
        half_x = min((target_shape[0] / 2), cx)
        half_y = min((target_shape[1] / 2), cy)
    else:
        half_x = min((target_shape[0] / 2), cx, original_size[0] - cx)
        half_y = min((target_shape[1] / 2), cy, original_size[1] - cy)

    # Compute starting indices
    start_x = math.floor(cx) - math.floor(half_x)
    start_y = math.floor(cy) - math.floor(half_y)

    assert start_x >= 0
    assert start_y >= 0

    # Compute ending indices
    if strict:
        end_x = start_x + target_shape[0]
        end_y = start_y + target_shape[1]
    else:
        end_x = start_x + 2 * math.floor(half_x)
        end_y = start_y + 2 * math.floor(half_y)

    # Perform the crop
    image = image[start_x:end_x, start_y:end_y, :]
    if depth_map is not None:
        depth_map = depth_map[start_x:end_x, start_y:end_y]

    # Shift the principal point in the intrinsic
    intrinsic[1, 2] = intrinsic[1, 2] - start_x
    intrinsic[0, 2] = intrinsic[0, 2] - start_y

    # Adjust track if provided
    if track is not None:
        track[:, 1] = track[:, 1] - start_x
        track[:, 0] = track[:, 0] - start_y

    # If strict, zero-pad if the new shape is smaller than target_shape
    if strict:
        if (image.shape[:2] != target_shape):
            print(f"{filepath} does not meet the target shape")
            current_h, current_w = image.shape[:2]
            target_h, target_w = target_shape[0], target_shape[1]
            pad_h = target_h - current_h
            pad_w = target_w - current_w
            if pad_h < 0 or pad_w < 0:
                raise ValueError(
                    f"The cropped image is bigger than the target shape: "
                    f"cropped=({current_h},{current_w}), "
                    f"target=({target_h},{target_w})."
                )
            image = np.pad(
                image,
                pad_width=((0, pad_h), (0, pad_w), (0, 0)),
                mode="constant",
                constant_values=0,
            )
            if depth_map is not None:
                depth_map = np.pad(
                    depth_map,
                    pad_width=((0, pad_h), (0, pad_w)),
                    mode="constant",
                    constant_values=0,
                )

    return image, depth_map, intrinsic, track