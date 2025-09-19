
import albumentations as A
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from typing import TypeVar
import logging

log = logging.getLogger(__name__)


T = TypeVar("T", bound=np.number)
SampleArg = tuple[T, T] | T


def sample(x: SampleArg) -> T:
    return np.random.uniform(x[0], x[1]) if isinstance(x, tuple) else x


def window_fn(
    images: np.ndarray,
    lower: SampleArg = 0.01,
    upper: SampleArg = 0.99,
    convert: bool = True,
) -> np.ndarray:
    """Apply a random window to an intensity image.

    Args:
        images (np.ndarray): [H,W,C] image
        upper (float, optional): The upper quantile of the window. Defaults to 0.99.
        lower (float, optional): The lower quantile of the window. Defaults to 0.01.

    Returns:
        np.ndarray: the image or images after having a random window applied.
    """
    eps = 1e-7
    upper = sample(upper)
    if upper == 1.0:
        upper = images.max()
    else:
        upper = np.quantile(images, upper, method="lower")

        log.info(f"Upper: {upper}")

    lower = sample(lower)
    if lower == 0.0:
        lower = images.min()
    else:
        lower = np.quantile(images, lower, method="higher")
        log.info(f"Lower: {lower}")

    if upper == lower:
        upper = images.max()
        lower = images.min()

    if upper == lower:
        images = np.zeros_like(images)
    else:
        images = images - lower
        images = images / (upper - lower)
        images = np.clip(images, 0, 1)

    if convert:
        images = (images * 255).astype(np.uint8)
    return images


def window(
    lower: SampleArg = 0.01,
    upper: SampleArg = 0.99,
    convert: bool = True,
):
    """Apply a random window to intensity images.

    Args:
        upper (float, optional): The upper quantile of the window. Defaults to 0.99.
        lower (float, optional): The lower quantile of the window. Defaults to 0.01.

    Returns:
        np.ndarray: the image or images after having a random window applied.
    """

    def _window(images: np.ndarray, **kwargs) -> np.ndarray:
        return window_fn(images, lower, upper, convert=convert)

    def f_id(x, **kwargs):
        return x

    return A.Lambda(
        image=_window,
        mask=f_id,
        keypoints=f_id,
        bboxes=f_id,
        name="window",
    )


def adjustable_window_fn(
    images: np.ndarray,
    x_lo: SampleArg,
    x_hi: SampleArg,
    y_lo: SampleArg = 0.1,
    y_hi: SampleArg = 0.9,
    quantile: bool = False,
):
    """Window an image within an adjustable window to [0,1].

    This is a piecewise lerp that will map:
    - [image_min, x_lo] -> [0, y_lo]
    - [x_lo, x_hi] -> [y_lo, y_hi]
    - [x_hi, image_max] -> [y_hi, 1]

    If y_lo == 0, then x_lo will be the minimum value of the image.
    If y_hi == 1, then x_hi will be the maximum value of the image.

    Args:
        images (np.ndarray): the image or images
        x_lo (float): The lower image value of the window in input space, to lerp to y_lo.
        x_hi (float): The upper image value of the window in input space, to lerp to y_high.
        y_lo (float, optional): The lower quantile of the window for the y-axis. Defaults to 0.1.
        y_hi (float, optional): The upper quantile of the window for the y-axis. Defaults to 0.9.
        quantile (bool, optional): Whether to treat x_lo and x_hi as quantiles. Defaults to False.

    Returns:
        np.ndarray: the image or images after having an adjustable window applied. Will be in the range [0,1].

    """
    assert 0 <= y_lo <= 1
    assert 0 <= y_hi <= 1

    x_lo = sample(x_lo)
    x_hi = sample(x_hi)
    y_lo = sample(y_lo)
    y_hi = sample(y_hi)

    image_min = images.min()
    image_max = images.max()

    if quantile:
        x_lo = np.quantile(images, x_lo, method="higher")
        x_hi = np.quantile(images, x_hi, method="lower")

    # TODO: bug when y_lo == 0 or y_hi == 1

    left_mask = images <= x_lo
    right_mask = images > x_hi
    middle_mask = ~left_mask & ~right_mask

    out_images = np.empty_like(images)
    if y_lo > 0:
        out_images[left_mask] = np.interp(images[left_mask], [image_min, x_lo], [0, y_lo])
    out_images[middle_mask] = np.interp(images[middle_mask], [x_lo, x_hi], [y_lo, y_hi])
    if y_hi < 1:
        out_images[right_mask] = np.interp(images[right_mask], [x_hi, image_max], [y_hi, 1])

    return out_images

def gaussian_mixture_window_fn(
    image: np.ndarray,
    n_components: int = 3,
    channel: int | None = 0,
    n_points: int = 1000,
):
    """Window the channels of the image separately, then combine them.

    Fits a Gaussian Mixture Model to each channel, then centers each window around the mean of the
    channel, with the range of the window corresponding to 3 SDs of the channel.

    Args:
        image (np.ndarray): The image to window, [H,W,C]
        n_components (int, optional): The number of components in the mixture. Defaults to 2.
            Determines the output channel dimension, as num_components + 1.
        channel (int | None, optional): Channel to treat as the input (assuming grayscale image) or
            None to take the intensity image.

    Returns:

    """

    if channel is not None:
        image = image[..., channel]
    else:
        image = image.mean(axis=-1)

    if np.isclose(image.min(), image.max()):
        return image

    gmm = GaussianMixture(n_components=n_components, covariance_type="spherical", max_iter=100)

    points = image.ravel().reshape(-1, 1)
    decimation = max(1, points.shape[0] // n_points)
    gmm.fit(points[::decimation])  # Subsample to speed up fitting
    labels = np.array(gmm.predict(points)).reshape(image.shape)
    # labels = np.array(gmm.fit_predict(image.ravel().reshape(-1, 1))).reshape(image.shape)

    out_image = np.empty((image.shape[0], image.shape[1], n_components), dtype=np.float32)
    means = gmm.means_.ravel()
    stds = np.sqrt(gmm.covariances_.ravel())
    component_labels = np.argsort(means)
    means = means[component_labels]
    stds = stds[component_labels]

    for i, label in enumerate(component_labels):
        mask = labels == label
        if mask.sum() == 0:
            out_image[:, :, i] = window_fn(image)
            continue
        mask_min = image[mask].min()
        mask_max = image[mask].max()
        x_lo = max(means[i] - 3 * stds[i], mask_min)
        x_hi = min(means[i] + 3 * stds[i], mask_max)

        # Lerp to [0,1]
        out_image[:, :, i] = adjustable_window_fn(image, x_lo, x_hi, 0.05, 0.95)

    return out_image

def kmeans_window_fn(
    image: np.ndarray,
    n_clusters: int = 3,
    channel: int | None = 0,
    n_points: int = 1000,
):
    """Window the channels of the image separately, then combine them.

    Fits a Gaussian Mixture Model to each channel, then centers each window around the mean of the
    channel, with the range of the window corresponding to 3 SDs of the channel.

    Args:
        image (np.ndarray): The image to window, [H,W,C]
        n_components (int, optional): The number of components in the mixture. Defaults to 2.
            Determines the output channel dimension, as num_components + 1.
        channel (int | None, optional): Channel to treat as the input (assuming grayscale image) or
            None to take the intensity image.
        n_points (int, optional): The number of points to use to fit the KMeans model. Defaults to 1000.

    Returns:

    """

    if channel is not None:
        image = image[..., channel]
    else:
        image = image.mean(axis=-1)

    if np.isclose(image.min(), image.max()):
        return np.stack([image] * n_clusters, axis=-1)

    kmeans = KMeans(n_clusters=n_clusters, max_iter=100)
    points = image.ravel().reshape(-1, 1)
    decimation = max(1, points.shape[0] // n_points)
    kmeans.fit(points[::decimation])  # Subsample to speed up fitting
    labels = np.array(kmeans.predict(points)).reshape(image.shape)
    # labels = np.array(kmeans.fit_predict(image.ravel().reshape(-1, 1))).reshape(image.shape)

    out_image = np.empty((image.shape[0], image.shape[1], n_clusters), dtype=np.float32)
    means = kmeans.cluster_centers_.ravel()
    component_labels = np.argsort(means)
    means = means[component_labels]

    for i, label in enumerate(component_labels):
        mask = labels == label
        if mask.sum() == 0:
            out_image[:, :, i] = window_fn(image)
            continue
        mask_min = image[mask].min()
        mask_max = image[mask].max()
        std = np.std(image[mask])

        # TODO: this should instead be the quantiles of the cluster
        x_lo = max(means[i] - 3 * std, mask_min)
        x_hi = min(means[i] + 3 * std, mask_max)

        # Lerp to [0,1]
        out_image[:, :, i] = adjustable_window_fn(image, x_lo, x_hi, 0.1, 0.9)

    return out_image


MIXTURE_WINDOW_FNS = dict(gmm=gaussian_mixture_window_fn, kmeans=kmeans_window_fn)


def mixture_window_keep_original_fn(
    image: np.ndarray, n_channels: int = 3, channel: int | None = 0, model: str = "gmm"
):
    out_image = np.empty((image.shape[0], image.shape[1], n_channels), dtype=np.float32)
    if channel is not None:
        out_image[:, :, 0] = window_fn(image, 0, 1.0, convert=False)[..., channel]
    else:
        out_image[:, :, 0] = window_fn(image, 0, 1.0, convert=False).mean(axis=-1)

    mixture_window_fn = MIXTURE_WINDOW_FNS[model]
    out_image[:, :, 1:] = mixture_window_fn(image, n_channels - 1, channel)
    # log.info(f"Image shape: {image.shape}")
    # log.info(f"Mixture shape: {mixture.shape}")
    return out_image


def mixture_window_keep_channel_fn(
    image: np.ndarray,
    n_channels: int = 3,
    channel: int | None = 0,
    keep_channel: int = 0,
    model: str = "gmm",
):
    """Window based on a mixture model with n_channels components, but only keep one of them."""
    mixture_window_fn = MIXTURE_WINDOW_FNS[model]
    mixture = mixture_window_fn(image, n_channels, channel)
    out_image = mixture[:, :, keep_channel].copy()
    out_image = np.stack([out_image] * n_channels, axis=-1)

    return out_image

def mixture_window(
    n_channels: int = 3,
    keep_original: bool = False,
    keep_channel: int | None = None,
    channel: int | None = 0,
    model: str = "gmm",
):
    """Window the image differently into the different channels.

    Fits a cluster to the pixel data, then centers each window around the mean of the channel, with
    the range of the window corresponding to 3 SDs of the channel.

    Args:
        n_channels (int, optional): The number of output channels. Defaults to 3.
            Determines the number of components.
        keep_original (bool, optional): Whether to keep the original image in the output, windowed . Defaults to False.
        channel (int | None, optional): Channel to treat as the input (assuming grayscale image) or
            None to take the intensity image.


    Returns:

    """

    def _mixture_window(image: np.ndarray, **kwargs) -> np.ndarray:
        if keep_original:
            return mixture_window_keep_original_fn(image, n_channels, channel, model=model)
        elif keep_channel is not None:
            return mixture_window_keep_channel_fn(
                image, n_channels, channel, keep_channel, model=model
            )
        else:
            mixture_window_fn = MIXTURE_WINDOW_FNS[model]
            return mixture_window_fn(image, n_channels, channel)

    def f_id(x, **kwargs):
        return x

    return A.Lambda(
        image=_mixture_window,
        mask=f_id,
        keypoints=f_id,
        bboxes=f_id,
        name="channel_mixture_window",
    )


def build_aug_dcm(
    image_size: int | tuple[int, int], bbox: bool = False, mixture: bool = False
) -> A.Compose:
    """Build transformation pipeline for real images, from a raw pixel array."""
    if isinstance(image_size, int):
        h = w = image_size
    else:
        h, w = image_size

    compose_kwargs = dict()
    if bbox:
        compose_kwargs["bbox_params"] = A.BboxParams(
            format="albumentations",  # xyxy, normalized to [0, 1]
            min_area=0,
            min_visibility=0,
            label_fields=["bbox_indices"],
        )

    if mixture:
        return A.Compose(
            [
                mixture_window(keep_original=True, model="kmeans"),
                A.LongestMaxSize(max_size=max(h, w), always_apply=True),
            ],
            **compose_kwargs,
        )

    else:
        return A.Compose(
            [
                window(0.0, 1.0, convert=False),
                A.LongestMaxSize(max_size=max(h, w), always_apply=True),
            ],
            **compose_kwargs,
        )
