from typing import Any, Dict, Tuple, List, Optional

import time
import numpy as np
import torch
from lightning import LightningModule
import clip
import logging
from rich.logging import RichHandler

from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch import nn

from .utils import image_utils
from .utils.augmentations import build_aug_dcm
from .mobilesamv2.modeling import Sam

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set log level
    format="%(message)s",  # Simple message format
    datefmt="[%X]",  # Time format
    handlers=[RichHandler(rich_tracebacks=True)]  # Use RichHandler
)

# Get a named logger
log = logging.getLogger("segment")


class SALitModule(nn.Module):
    """Segment-anything module for FluoroSAM model.
    """

    sam: Sam

    def __init__(
        self,
        model: Sam,
        prompts_per_mask: int = 8,
        prompt_types: list[str] = ["text"],
        visualize: bool = True,
        save_freq: int = 3600,  # 7200,  # 1800,
        take_max_iou: bool = True,
        device: str = "cpu",
    ) -> None:
        """Initialize a `MNISTLitModule`.

        Args:
            net: The model to train.
            optimizer: The optimizer to use for training.
            scheduler: The learning rate scheduler to use for training.
            prompt_batch_size: The maximum number of true masks for each image to sample.
            take_max_iou: During evaluation, take the mask with the highest GT IoU, rather than the highest predicted IoU.

        """
        super(SALitModule, self).__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        # self.save_hyperparameters(logger=False)

        model = model.to(device)

        self.prompts_per_mask = prompts_per_mask
        self.sam = model
        self._image_size = (model.image_encoder.img_size, model.image_encoder.img_size)
        self.prompt_types = prompt_types

        start_time = time.time() - (58 * 60)
        self.vis_timer = {"train": start_time, "val": start_time, "test": start_time}
        self.save_freq = save_freq

        weight = torch.tensor([0.05, 0.95], dtype=torch.float32)
        self.cross_entropy = torch.nn.CrossEntropyLoss(weight=weight)

        self.enable_visualize = visualize

        self.test_log_keys = [
            "dice_loss",
            "iou_loss",
        ]

        self.take_max_iou = take_max_iou

        self.reset_test_stats()
        self.device = device

    def reset_test_stats(self):
        self.test_num_losses_agg_promptcount = [0] * (self.prompts_per_mask + 2)
        self.test_agg_promptcount = {
            k: [0] * (self.prompts_per_mask + 2) for k in self.test_log_keys
        }
        self.test_agg_promptcount = {
            k: [0] * (self.prompts_per_mask + 2) for k in self.test_log_keys
        }
        self.test_num_losses_agg = 0
        self.test_agg = {k: 0 for k in self.test_log_keys}

        self.test_agg_rows = []

    def forward(
        self,
        batched_input: List[Dict[str, Any]],
        multimask_output: bool,
    ) -> List[Dict[str, torch.Tensor]]:
        return self.sam.forward(batched_input=batched_input, multimask_output=multimask_output)

    def encode_image(self, image: np.ndarray, mixture: bool = True) -> np.ndarray:
        """Encode an image into a feature vector.

        Args:
            image: The image to encode, as a (H,W) numpy array, in range [0,1]

        Returns:
            image_embedding: The encoded image.
        """

        # self._actual_original_size = image.shape[:2]

        self._original_size = image.shape[:2]
        image = np.array(image, dtype=np.float32)
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[-1] == 1:
            image = np.concatenate([image] * 3, axis=-1)
        else:
            image = image[..., :3]

        aug = build_aug_dcm(self._image_size, bbox=False, mixture=mixture)
        image = aug(image=image)["image"]

        vis_image = np.concatenate([image[:, :, 0], image[:, :, 1], image[:, :, 2]], axis=1)
        log.info(f"vis_image: {vis_image.shape}")
        image_utils.save("debug/sam/model_input.png", vis_image)

        image = image.transpose(2, 0, 1)
        image = torch.tensor(image).unsqueeze(0).to(self.device)
        image = self.sam.preprocess(image)

        log.info(
            f"input_image: {image.shape}, mean/min/max: {image.mean():.02f}/{image.min():.02f}/{image.max():.02f}, device: {image.device}, dtype: {image.dtype}"
        )

        image_embedding = self.sam.image_encoder(image)
        image_embedding = image_embedding.detach().cpu().numpy()
        return image_embedding

    def predict_mask(
        self,
        image_embedding: np.ndarray | torch.Tensor,
        text_prompt: str,
        points: tuple[np.ndarray, np.ndarray] | None = None,
        original_size: tuple[int, int] | None = None,
        pad: tuple[int, int, int, int] | None = None,
        iou_threshold: float = 0,
        return_iou: bool = False,
    ) -> np.ndarray:
        """Predict a mask given an encoded image and a text prompt.

        Args:
            image_embedding: The encoded image.
            text_prompt: The text prompt to use.
            points: The points to use for the prompt.
            original_size: The original size of the image. Should be saved after encoding the image.
            pad: The padding applied to the image. Should be saved after encoding.
            iou_threshold: The IoU threshold to use for selecting the mask. If the max IoU is below
                this threshold, the mask is zeroed.

        Returns:
            logits: The predicted logits, as a heatmap (H,W).
        """
        if original_size is None:
            assert hasattr(self, "_original_size"), "original_size must be provided"
            original_size = self._original_size

        if isinstance(image_embedding, np.ndarray):
            curr_embedding = torch.tensor(image_embedding).to(self.device)
        else:
            curr_embedding = image_embedding.to(self.device)

        text_inputs = clip.tokenize([text_prompt]).to(self.device)

        if points is not None:
            points = (
                torch.tensor(points[0]).to(self.device),
                torch.tensor(points[1]).to(self.device),
            )

        sparse_embeddings, dense_embeddings, _ = self.sam.prompt_encoder(
            points=points,
            boxes=None,
            masks=None,
            text_inputs=text_inputs,
        )

        low_res_masks, iou_predictions = self.sam.mask_decoder(
            image_embeddings=curr_embedding,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
        )
        pred_multimask = self.sam.postprocess_masks(
            low_res_masks, self._image_size, original_size=original_size
        )

        # Get the index based on the highest IoU
        pred_idx_out = torch.argmax(iou_predictions, dim=1)
        pred_iou = iou_predictions[0, pred_idx_out[0]].item()
        # log.info(f"iou max: {iou_predictions.max().item()}")
        if pred_iou < iou_threshold:
            # log.info(f"max IoU below threshold: {torch.max(iou_predictions)}")
            if return_iou:
                return np.zeros(original_size, dtype=np.float32), pred_iou
            return np.zeros(original_size, dtype=np.float32)

        pred_idx_out = torch.argmax(iou_predictions, dim=1)
        pred_idx = pred_idx_out[:, None, None, None]
        # log.info(f"took pred_idx: {pred_idx}")
        pred_mask = torch.take_along_dim(pred_multimask, pred_idx, dim=1)[:, 0]
        pred_mask = pred_mask[0]

        mask = pred_mask.detach().cpu().numpy()
        pred_multimask = pred_multimask.squeeze(0).detach().cpu().numpy()

        # resize to acutal size
        # h, w = self._actual_original_size
        # mask = F.interpolate(
        #     pred_mask[None, None], size=(h, w), mode="bilinear", align_corners=False
        # ).squeeze(0).squeeze(0)

        # h, w = mask.shape[:2]
        # log.info(f"mask: {mask.shape}")
        # mask = mask[pad[0] : h - pad[1], pad[2] : w - pad[3]]

        if return_iou:
            pred_iou = iou_predictions[0, pred_idx_out[0]].item()
            return mask, pred_iou, pred_multimask
        return mask, pred_multimask
    