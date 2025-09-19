import click
import logging
from pathlib import Path
from rich.logging import RichHandler
from pydicom import dcmread
import numpy as np
from PIL import Image

from .model_wrapper import SALitModule
from .build_model import create_sam_model
from .utils.file_utils import download
from .utils import image_utils

log = logging.getLogger("segment")
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)

RED = [145, 56, 49]
GREEN = [85, 128, 70]
BLUE = [66, 135, 245]
MAGENTA = [197, 58, 224]
IOU_THRESHOLD = 2e-1


def pad_to_square(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Pad the image to make it square.

    Returns:
    - padded image
    - padding (top, bottom, left, right)
    """
    h, w = image.shape[:2]
    if h == w:
        return image, (0, 0, 0, 0)

    val = np.mean(image)
    if h > w:
        pad = (h - w) // 2
        return np.pad(image, ((0, 0), (pad, pad)), mode="constant", constant_values=val), (
            0,
            0,
            pad,
            pad,
        )
    else:
        pad = (w - h) // 2
        return np.pad(image, ((pad, pad), (0, 0)), mode="constant", constant_values=val), (
            pad,
            pad,
            0,
            0,
        )


@click.command()
@click.argument(
    "prompt",
    nargs=-1,
)
@click.option(
    "--ckpt-path",
    default="/home/killeen/projects/fluorosam/model_weights.pth",
    help="Path to the checkpoint file. Can also be a URL.",
)
@click.option(
    "--ckpt-url",
    default=None,
    help="URL to the checkpoint file.",
)
@click.option(
    "--output-dir", "-o", type=str, default=None, help="Output directory for the predictions."
)
@click.option(
    "--input-path", "-i", type=click.Path(exists=True), default=None, help="Path to the image."
)
@click.option(
    "--size",
    "-s",
    type=int,
    default=None,
    help="Size of the image to write out.",
)
def main(
    prompt: list[str],
    ckpt_path: str,
    ckpt_url: str | None,
    input_path: Path | None,
    output_dir: Path | None,
    size: int | None,
):
    # Get the image Path
    if input_path is None:
        log.error("Please provide an input path.")
        return
    image_path = Path(input_path)
    # Set the output directory
    if output_dir is None:
        output_dir = image_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(ckpt_path)
    # Load the model
    if ckpt_url is not None:
        download(ckpt_url, ckpt_path)
    if not ckpt_path.exists():
        log.error(f"Checkpoint not found at {ckpt_path}")
        return

    model = create_sam_model(backbone="swin-l", pretrained=True, weight_path=str(ckpt_path))
    model = SALitModule(model)
    # model = SALitModule.load_from_checkpoint(ckpt_path, strict=False)
    model.eval()

    # log.info(f"Loaded the model from {ckpt_path}:\n{model}")

    ext = image_path.suffix
    if ext not in [".dcm", ".png", ".jpg", ".jpeg"]:
        log.error(f"Unsupported file format: {ext}")
        return
    elif ext == ".dcm":
        ds = dcmread(image_path)
        image = ds.pixel_array.astype(np.float32)
        image = image / 65535
        # image = (image - image.min()) / (image.max() - image.min())
        vis_image = image_utils.process_drr(image, neglog=False, clahe=False)
        dcm = True
    else:

        image = np.array(Image.open(image_path).convert("F")).astype(np.float32)
        image, _ = pad_to_square(image)
        vis_image = (image * 255).astype(np.uint8)
        dcm = False

    # image, pad = pad_to_square(image)

    image_dir = output_dir / image_path.stem
    for path in image_dir.glob("*"):
        path.unlink()
    image_dir.mkdir(parents=True, exist_ok=True)
    preview_path = image_dir / f"preview.png"
    if not preview_path.exists():
        Image.fromarray(vis_image).save(preview_path)

    log.info(f"raw image: {image.shape} {image.mean()} {image.min()} {image.max()}")

    embedding_path = image_dir / f"embedding.npy"
    if embedding_path.exists() and False:
        image_embedding = np.load(embedding_path)
    else:
        image_embedding = model.encode_image(
            image,
            mixture=True,
        )
        # np.save(embedding_path, image_embedding)
        log.info(f"Saved the image embedding at {embedding_path}")

    if prompt is None:
        log.info("Prompt not provided. Skipping mask prediction.")
        return

    for i, p in enumerate(prompt):
        log.info(f"\nPredicting mask for the prompt '{p}'")
        mask, pred_iou = model.predict_mask(
            image_embedding, p, original_size=image.shape[:2], return_iou=True
        )

        log.info(f"Predicted mask for the prompt '{p}'.")
        log.info(f"mask: {mask.shape} {mask.min()} {mask.max()}")
        heatmap_vis = image_utils.combine_heatmap(vis_image, mask)
        mask_vis = image_utils.draw_masks(
            vis_image,
            [mask],
            colors=[MAGENTA],
            # names=[f"({pred_iou:.02f}){p}"],
            threshold=0.5,
            contour_thickness=int(0.010 * vis_image.shape[0]),
            horizontal_alignment="center",
            vertical_alignment="bottom",
        )  # Maybe threshold for detection is higher than for the mask

        # if not np.all(pad == 0):
        #     mask = mask[pad[0] : -pad[1], pad[2] : -pad[3]]
        #     heatmap_vis = heatmap_vis[pad[0] : -pad[1], pad[2] : -pad[3]]
        #     mask_vis = mask_vis[pad[0] : -pad[1], pad[2] : -pad[3]]

        # this_vis_image = np.concatenate([vis_image, heatmap_vis, mask_vis], axis=1)
        this_vis_image = mask_vis
        if size is not None:
            this_vis_image = image_utils.resize_by_short_side(this_vis_image, size)

        vis_path = image_dir / f"{i:02d} {p} ({pred_iou:.02f}).png"
        Image.fromarray(this_vis_image).save(vis_path)
        log.info(f"Saved the mask at {vis_path}")


if __name__ == "__main__":
    main()
