from pathlib import Path
from PIL import Image
import numpy as np
import logging
from PIL import Image
import numpy as np
import cv2
import seaborn as sns
from typing import List, Optional

log = logging.getLogger(__name__)

def as_uint8(image: np.ndarray) -> np.ndarray:
    """Convert the image to uint8.

    Args:
        image (np.ndarray): the image to convert.

    Returns:
        np.ndarray: the converted image.
    """
    if image.dtype in [np.float16, np.float32, np.float64]:
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
    elif image.dtype == bool:
        image = image.astype(np.uint8) * 255
    elif image.dtype != np.uint8:
        logging.warning(f"Unknown image type {image.dtype}. Converting to uint8.")
        image = image.astype(np.uint8)
    return image

def as_float32(image: np.ndarray) -> np.ndarray:
    """Convert the image to float32.

    Args:
        image (np.ndarray): the image to convert.

    Returns:
        np.ndarray: the converted image.
    """
    if image.dtype in [np.float16, np.float32, np.float64]:
        image = image.astype(np.float32)
    elif image.dtype == bool:
        image = image.astype(np.float32)
    elif image.dtype != np.uint8:
        logging.warning(f"Unknown image type {image.dtype}. Converting to float32.")
        image = image.astype(np.float32)
    else:
        image = image.astype(np.float32) / 255
    return image

def save(path: Path, image: np.ndarray, mkdir: bool = True) -> Path:
    """Save the given image using PIL.

    Args:
        path (Path): the path to write the image to. Also determines the type.
        image (np.ndarray): the image, in [C, H, W] or [H, W, C] order. (If the former, transposes).
            If in float32, assumed to be a float image. Converted to uint8 before saving.
    """
    path = Path(path)
    if not path.parent.exists() and mkdir:
        path.parent.mkdir(parents=True)

    if len(image.shape) == 3 and image.shape[0] in [3, 4]:
        image = image.transpose(1, 2, 0)

    image = as_uint8(image)

    Image.fromarray(image).save(str(path))
    return path

def ensure_cdim(x: np.ndarray, c: int = 3) -> np.ndarray:
    if x.ndim == 2:
        x = x[:, :, np.newaxis]
    elif x.ndim == 3:
        pass
    else:
        raise ValueError(f"bad input ndim: {x.shape}")

    if x.shape[2] < c:
        return np.concatenate([x] * c, axis=2)
    elif x.shape[2] == c:
        return x
    else:
        raise ValueError(f"bad input shape: {x.shape}")

def combine_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    channel=0,
    normalize=True,
) -> np.ndarray:
    """Visualize a heatmap on an image.

    Args:
        image (Union[np.ndarray, torch.Tensor]): 2D float image, [H, W]
        heatmap (Union[np.ndarray, torch.Tensor]): 2D float heatmap, [H, W], or [C, H, W] array of heatmaps.
        channel (int, optional): Which channel to use for the heatmap. For an RGB image, channel 0 would render the heatmap in red.. Defaults to 0.
        normalize (bool, optional): Whether to normalize the heatmap. This can lead to all-red images if no landmark was detected. Defaults to True.

    Returns:
        np.ndarray: A [H,W,3] numpy image.
    """
    image_arr = ensure_cdim(as_float32(image), c=3)
    heatmap_arr = ensure_cdim(heatmap, c=1)

    heatmap_arr = heatmap_arr.transpose(2, 0, 1)
    image_arr = image_arr.transpose(2, 0, 1)

    seg = False
    if heatmap_arr.dtype == bool:
        heatmap_arr = heatmap_arr.astype(np.float32)
        seg = True

    _, h, w = heatmap_arr.shape
    heat_sum = np.zeros((h, w), dtype=np.float32)
    for heat in heatmap_arr:
        heat_min = heat.min()
        heat_max = 4 if seg else heat.max()
        heat_min_minus_max = heat_max - heat_min
        heat = heat - heat_min
        if heat_min_minus_max > 1.0e-3:
            heat /= heat_min_minus_max

        heat_sum += heat

    for c in range(3):
        image_arr[c] = ((1 - heat_sum) * image_arr[c]) + (heat_sum if c == channel else 0)

    # log.debug(f"Combined heatmap with shape {image_arr.shape} and dtype {image_arr.dtype}")

    return as_uint8(image_arr.transpose(1, 2, 0))


def draw_keypoints(
    image: np.ndarray,
    keypoints: np.ndarray,
    names: Optional[List[str]] = None,
    colors: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    num_classes: Optional[int] = None,
    label_order: Optional[np.ndarray] = None,
    palette: str = "hls",
    seed: Optional[int] = None,
) -> np.ndarray:
    """Draw keypoints on an image (copy).

    Args:
        image (np.ndarray): the image to draw on.
        keypoints (np.ndarray): the keypoints to draw. [N, 2] array of [x, y] coordinates.
            -1 indicates no keypoint present.
        names (List[str], optional): the names of the keypoints. Defaults to None.
        colors (np.ndarray, optional): the colors to use for each keypoint. Defaults to None. If labels is provided this is overwritten.
        labels (np.ndarray, optional): the labels for each keypoint, if they should be assigned to classes. Defaults to None.
        label_order (np.ndarray, optional): the order of the labels, corresponding to colors. Defaults to None.

    Returns:
        np.ndarray: the image with the keypoints drawn.

    """

    if len(keypoints) == 0:
        return image

    if seed is not None:
        np.random.seed(seed)
    image = ensure_cdim(as_uint8(image)).copy()
    keypoints = np.array(keypoints)

    if labels is not None:
        if label_order is None:
            unique_labels = sorted(list(np.unique(labels)))
            num_colors = num_classes if num_classes is not None else len(unique_labels)
            label_to_idx = {label: i for i, label in enumerate(unique_labels)}
        else:
            num_colors = len(label_order)
            label_to_idx = {label: i for i, label in enumerate(label_order)}
        if colors is None:
            colors = np.array(sns.color_palette(palette, num_colors))
            colors = colors[np.random.permutation(colors.shape[0])]
        colors = np.array(colors)

        colors = np.array([colors[label_to_idx[l]] for l in labels])

    if colors is None:
        colors = np.array(sns.color_palette(palette, keypoints.shape[0]))
        colors = colors[np.random.permutation(colors.shape[0])]
    else:
        colors = np.array(colors)

    if np.any(colors < 1):
        colors = (colors * 255).astype(int)

    fontscale = 0.75 / 512 * image.shape[0]
    thickness = max(int(1 / 256 * image.shape[0]), 1)
    offset = max(5, int(5 / 512 * image.shape[0]))
    radius = max(1, int(15 / 512 * image.shape[0]))

    for i, keypoint in enumerate(keypoints):
        if np.any(keypoint < 0):
            continue
        color = colors[i].tolist()
        x, y = keypoint

        # Draw a circle with a white outline.
        image = cv2.circle(image, (int(x), int(y)), radius + 1, (255, 255, 255), -1)
        image = cv2.circle(image, (int(x), int(y)), radius, color, -1)

        if names is not None:
            image = cv2.putText(
                image,
                names[i],
                (int(x) + offset, int(y) - offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontscale,
                color,
                thickness,
                cv2.LINE_AA,
            )
    return image


def draw_masks(
    image: np.ndarray,
    masks: np.ndarray,
    alpha: float = 0.3,
    threshold: float = 0.5,
    names: Optional[List[str]] = None,
    colors: Optional[np.ndarray] = None,
    name_colors: Optional[np.ndarray] = None,
    palette: str = "hls",
    seed: Optional[int] = None,
    contours: bool = True,
    contour_thickness: int = 1,
    horizontal_alignment: str = "center",
    vertical_alignment: str = "center",
) -> np.ndarray:
    """Draw contours of masks on an image (copy).

    TODO: add options for thresholding which masks to draw based on portion of image size?

    Args:
        image (np.ndarray): the image to draw on.
        masks (np.ndarray): the masks to draw. [num_masks, H, W] array of masks.
        alpha (float, optional): the alpha value for the mask. Defaults to 0.3.
        threshold (float, optional): the threshold for the mask. Defaults to 0.5.
        names (List[str], optional): the names of the masks. Defaults to None.
        colors (np.ndarray, optional): the colors to use for each mask,
            as an [num_masks, 3] array of RGB values in [0,255]. Defaults to None.
        name_colors (np.ndarray, optional): the colors to use for each mask name,
            as a [num_masks, 3] array of RGB values in [0,255]. Defaults to None.
        palette (str, optional): the name of the color palette to use. Defaults to "hls".
        seed (Optional[int], optional): the seed for the random number generator. Defaults to None.
        contours (bool, optional): whether to draw contours around the masks. Defaults to True.
        contour_thickness (int, optional): the thickness of the contours. Defaults to 1.
        horizontal_alignment (str, optional): the horizontal alignment of the names. One of "left", "center", "right". Defaults to "center".
        vertical_alignment (str, optional): the vertical alignment of the names. One of "top", "center", "bottom". Defaults to "center".

    Returns:
        np.ndarray: the image with the masks drawn.
    """

    masks = np.array(masks).astype(np.float32)
    image = as_float32(image).copy()
    image = ensure_cdim(image)
    if colors is None:
        colors = np.array(sns.color_palette(palette, masks.shape[0]))
        if seed is not None:
            np.random.seed(seed)
        colors = colors[np.random.permutation(colors.shape[0])]
    else:
        colors = np.array(colors)

    if np.any(colors > 1):
        colors = colors.astype(float) / 255

    if name_colors is None:
        name_colors = colors
    else:
        name_colors = np.array(name_colors)
        if np.any(name_colors > 1):
            name_colors = name_colors.astype(float) / 255

    # image *= 1 - alpha
    for i, mask in enumerate(masks):
        bool_mask = mask > threshold

        area = bool_mask.sum()
        h, w = mask.shape
        fraction = area / (h * w)

        # if fraction > 0.9:
        #     continue
        if not bool_mask.any():
            continue

        image[bool_mask] = colors[i] * alpha + image[bool_mask] * (1 - alpha)

        if contours:
            contours_, _ = cv2.findContours(
                bool_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            image = as_uint8(image)
            c = tuple((255 * colors[i]).astype(int).tolist())
            cv2.drawContours(image, contours_, -1, c, contour_thickness)
            image = as_float32(image)

    image = as_uint8(image)

    fontscale = 0.75 / 512 * image.shape[0]
    # thickness = max(int(1 / 256 * image.shape[0]), 1)
    thickness = max(contour_thickness // 2, 1)
    if names is not None:
        for i, mask in enumerate(masks):
            bool_mask = mask > threshold
            ys, xs = np.argwhere(bool_mask).T
            if len(ys) == 0:
                continue

            # Align the text on the mask. TODO: Make the anchor point dynamic.
            text_size, _ = cv2.getTextSize(names[i], cv2.FONT_HERSHEY_SIMPLEX, fontscale, thickness)
            if horizontal_alignment == "left":
                mx = np.min(xs)
            elif horizontal_alignment == "center":
                mx = (np.min(xs) + np.max(xs)) / 2
            elif horizontal_alignment == "right":
                mx = np.max(xs)
            else:
                raise ValueError(f"Invalid horizontal alignment: {horizontal_alignment}")

            if vertical_alignment == "top":
                my = np.min(ys)
            elif vertical_alignment == "center":
                my = (np.min(ys) + np.max(ys)) / 2
            elif vertical_alignment == "bottom":
                my = np.max(ys)
            else:
                raise ValueError(f"Invalid vertical alignment: {vertical_alignment}")

            x = int(mx - text_size[0] // 2)
            y = int(my - text_size[1] // 2)

            # draw the larger outline in white
            image = cv2.putText(
                image,
                names[i],
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontscale,
                (255, 255, 255),
                thickness + max(1, thickness // 5),
                cv2.LINE_AA,
            )

            # draw the text in the mask color
            image = cv2.putText(
                image,
                names[i],
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontscale,
                (255 * name_colors[i]).tolist(),
                thickness,
                cv2.LINE_AA,
            )

    return image

def _neglog(image: np.ndarray, epsilon: float = 0.01) -> np.ndarray:
    """Take the negative log transform of an intensity image.

    Args:
        image (np.ndarray): a single 2D image, or N such images.
        epsilon (float, optional): positive offset from 0 before taking the logarithm.

    Returns:
        np.ndarray: the image or images after a negative log transform, scaled to [0, 1]
    """
    image = np.array(image)
    shape = image.shape
    if len(shape) == 2:
        image = image[np.newaxis, :, :]

    # shift image to avoid invalid values
    image += image.min(axis=(1, 2), keepdims=True) + epsilon

    # negative log transform
    image = -np.log(image)

    # linear interpolate to range [0, 1]
    image_min = image.min(axis=(1, 2), keepdims=True)
    image_max = image.max(axis=(1, 2), keepdims=True)
    if np.any(image_max == image_min):
        log.debug(
            f"mapping constant image to 0. This probably indicates the projector is pointed away from the volume."
        )
        # TODO(killeen): for multiple images, only fill the bad ones
        image[:] = 0
        if image.shape[0] > 1:
            log.error("TODO: zeroed all images, even though only one might be bad.")
    else:
        image = (image - image_min) / (image_max - image_min)

    if np.any(np.isnan(image)):
        log.warning(f"got NaN values from negative log transform.")

    if len(shape) == 2:
        return image[0]
    else:
        return image


def process_drr(
    image: np.ndarray,
    neglog: bool = True,
    clahe: bool = True,
    invert: bool = True,
) -> np.ndarray:
    """Process a raw DRR for visualization."""
    # Cast to uint8
    if neglog:
        image = _neglog(image)
    image = as_uint8(image)

    # apply clahe and invert
    if clahe:
        clahe = cv2.createCLAHE(clipLimit=4, tileGridSize=(8, 8))
        image = clahe.apply(image)

    if invert:
        image = 255 - image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def resize_by_height(image: np.ndarray, height: int) -> np.ndarray:
    """Resize a numpy array image"""
    h, w, _ = image.shape

    new_w = int(height * w / h)
    new_h = height

    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_NEAREST)


def resize_by_width(image: np.ndarray, width: int) -> np.ndarray:
    """Resize a numpy array image"""
    h, w, _ = image.shape

    new_w = width
    new_h = int(width * h / w)

    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_NEAREST)


def resize_by_short_side(image: np.ndarray, size: int) -> np.ndarray:
    """Resize a numpy array image"""
    h, w, _ = image.shape

    if h < w:
        return resize_by_height(image, size)
    else:
        return resize_by_width(image, size)
