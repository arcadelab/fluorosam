from .utils.file_utils import download
from .efficientvit.models.efficientvit.backbone import (
    EfficientViTBackbone,
    efficientvit_backbone_b0,
    efficientvit_backbone_b1,
    efficientvit_backbone_b2,
    efficientvit_backbone_b3,
    efficientvit_backbone_l1,
    efficientvit_backbone_l2,
)
from .efficientvit.models.efficientvit.seg import SegHead, EfficientViTCyclops, EfficientViTHydra
from .efficientvit.models.efficientvit.sam import SamNeck, EfficientViTSamImageEncoder
from .efficientvit.models.utils import load_state_dict_from_file
from pathlib import Path
from torch import nn
import torch

from .mobilesamv2.modeling import (
    ImageEncoderViT,
    MaskDecoder,
    TravelMaskDecoder,
    PromptEncoder,
    Sam,
    TwoWayTransformer,
)

import logging
log = logging.getLogger(__name__)


# NUM_CLASSES = 129 # non-reduced
NUM_CLASSES = 57  # reduced


def _load_checkpoint(model: nn.Module, checkpoint):
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
    model.load_state_dict(state_dict, strict=False)
    return model


def _seg_backbone_b0():
    backbone = efficientvit_backbone_b0()
    head_kwargs = dict(
        fid_list=["stage4", "stage3", "stage2"],
        in_channel_list=[128, 64, 32],
        stride_list=[32, 16, 8],
        head_stride=1,
        head_width=32,
        head_depth=1,
        expand_ratio=4,
        middle_op="mbconv",
        final_expand=4,
    )
    return backbone, head_kwargs


def _backbone_b1():
    backbone = efficientvit_backbone_b1()
    head_kwargs = dict(
        fid_list=["stage4", "stage3", "stage2"],
        in_channel_list=[256, 128, 64],
        stride_list=[32, 16, 8],
        head_stride=1,
        head_width=64,
        head_depth=3,
        expand_ratio=4,
        middle_op="mbconv",
        final_expand=4,
    )
    return backbone, head_kwargs


def _backbone_b2():
    backbone = efficientvit_backbone_b2()
    head_kwargs = dict(
        fid_list=["stage4", "stage3", "stage2"],
        in_channel_list=[384, 192, 96],
        stride_list=[32, 16, 8],
        head_stride=1,
        head_width=96,
        head_depth=3,
        expand_ratio=4,
        middle_op="mbconv",
        final_expand=4,
    )
    return backbone, head_kwargs


def _backbone_b3():
    backbone = efficientvit_backbone_b3()
    head_kwargs = dict(
        fid_list=["stage4", "stage3", "stage2"],
        in_channel_list=[512, 256, 128],
        stride_list=[32, 16, 8],
        head_stride=1,
        head_width=128,
        head_depth=3,
        expand_ratio=4,
        middle_op="mbconv",
        final_expand=4,
    )
    return backbone, head_kwargs


def _assemble_seg_model(backbone, head_kwargs, travel=False, multihead=True):
    if travel and multihead:
        seg_head = SegHead(
            **head_kwargs,
            n_classes=NUM_CLASSES,
        )
        travel_head = SegHead(
            **head_kwargs,
            n_classes=NUM_CLASSES,
        )
        model = EfficientViTHydra(backbone, segs=seg_head, travels=travel_head)
    elif travel and not multihead:
        head = SegHead(
            **head_kwargs,
            n_classes=2 * NUM_CLASSES,
        )
        model = EfficientViTCyclops(backbone, head, segs=NUM_CLASSES, travels=NUM_CLASSES)
    else:
        seg_head = SegHead(
            **head_kwargs,
            n_classes=NUM_CLASSES,
        )
        model = EfficientViTHydra(backbone, segs=seg_head)

    return model


def prephix_efficientvit_seg_b0(travel=False, multihead=True):
    backbone, head_kwargs = _seg_backbone_b0()
    return _assemble_seg_model(backbone, head_kwargs, travel, multihead)


def prephix_efficientvit_seg_b1(travel=False, multihead=True):
    backbone, head_kwargs = _backbone_b1()
    return _assemble_seg_model(backbone, head_kwargs, travel, multihead)


def prephix_efficientvit_seg_b2(travel=False, multihead=True):
    backbone, head_kwargs = _backbone_b2()
    return _assemble_seg_model(backbone, head_kwargs, travel, multihead)


def prephix_efficientvit_seg_b3(travel=False, multihead=True):
    backbone, head_kwargs = _backbone_b3()
    return _assemble_seg_model(backbone, head_kwargs, travel, multihead)


REGISTERED_SEG_MODEL: dict[str, dict[str, str]] = {
    "cityscapes": {
        "b0": "assets/checkpoints/seg/cityscapes/b0.pt",
        "b1": "assets/checkpoints/seg/cityscapes/b1.pt",
        "b2": "assets/checkpoints/seg/cityscapes/b2.pt",
        "b3": "assets/checkpoints/seg/cityscapes/b3.pt",
        ################################################
        "l1": "assets/checkpoints/seg/cityscapes/l1.pt",
        "l2": "assets/checkpoints/seg/cityscapes/l2.pt",
    },
    "ade20k": {
        "b1": "assets/checkpoints/seg/ade20k/b1.pt",
        "b2": "assets/checkpoints/seg/ade20k/b2.pt",
        "b3": "assets/checkpoints/seg/ade20k/b3.pt",
        ################################################
        "l1": "assets/checkpoints/seg/ade20k/l1.pt",
        "l2": "assets/checkpoints/seg/ade20k/l2.pt",
    },
}
REGISTERED_SEG_URL: dict[str, dict[str, str]] = {
    "cityscapes": {
        "b0": "https://drive.google.com/file/d/1Ix1Dh3xlpaf0Wzh01Xmo-hAYkoXt1EAD/view?usp=sharing",
    },
}

VIT_PATCH_SIZE = 16


def build_sam(
    image_encoder: EfficientViTSamImageEncoder,
    image_size: int,
    prompt_embed_dim: int = 256,
    patch_size: int = VIT_PATCH_SIZE,
    codebook_size: int = 8192,
    use_vq: bool = True,
) -> Sam:
    image_embedding_size = image_size // patch_size
    return Sam(
        image_encoder=image_encoder,
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
            use_vq=use_vq,
            codebook_size=codebook_size,
        ),
        mask_decoder=TravelMaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[0.5, 0.5, 0.5],
        pixel_std=[0.5, 0.5, 0.5],
    )


def efficientvit_sam_b0(
    img_size: int = 224,
    prompt_embed_dim: int = 256,
    **kwargs,
):
    image_embedding_size = img_size // VIT_PATCH_SIZE
    backbone = efficientvit_backbone_b0()
    neck = SamNeck(
        fid_list=["stage4", "stage3", "stage2"],
        in_channel_list=[128, 64, 32],
        head_width=32,
        head_depth=1,
        expand_ratio=1,
        middle_op="fmbconv",
        out_dim=prompt_embed_dim,
        out_size=image_embedding_size,  # Results in img_size mask outputs from the decoder
    )
    image_encoder = EfficientViTSamImageEncoder(backbone, neck, img_size=img_size)
    return build_sam(image_encoder, img_size, prompt_embed_dim, **kwargs)


def efficientvit_sam_b1(
    img_size: int = 224,
    prompt_embed_dim: int = 256,
    **kwargs,
):
    image_embedding_size = img_size // VIT_PATCH_SIZE
    backbone = efficientvit_backbone_b1()
    neck = SamNeck(
        fid_list=["stage4", "stage3", "stage2"],
        in_channel_list=[256, 128, 64],
        head_width=64,
        head_depth=3,
        expand_ratio=1,
        middle_op="fmbconv",
        out_dim=prompt_embed_dim,
        out_size=image_embedding_size,  # Results in img_size mask outputs from the decoder
    )
    image_encoder = EfficientViTSamImageEncoder(backbone, neck, img_size=img_size)
    return build_sam(image_encoder, img_size, prompt_embed_dim)


def efficientvit_sam_b2(
    img_size: int = 224,
    prompt_embed_dim: int = 256,
    **kwargs,
):
    image_embedding_size = img_size // VIT_PATCH_SIZE
    backbone = efficientvit_backbone_b2()
    neck = SamNeck(
        fid_list=["stage4", "stage3", "stage2"],
        in_channel_list=[384, 192, 96],
        head_width=96,
        head_depth=3,
        expand_ratio=1,
        middle_op="fmbconv",
        out_dim=prompt_embed_dim,
        out_size=image_embedding_size,  # Results in img_size mask outputs from the decoder
    )
    image_encoder = EfficientViTSamImageEncoder(backbone, neck, img_size=img_size)
    return build_sam(image_encoder, img_size, prompt_embed_dim, **kwargs)


def efficientvit_sam_b3(
    img_size: int = 224,
    prompt_embed_dim: int = 256,
    **kwargs,
):
    image_embedding_size = img_size // VIT_PATCH_SIZE
    backbone = efficientvit_backbone_b3()
    neck = SamNeck(
        fid_list=["stage4", "stage3", "stage2"],
        in_channel_list=[512, 256, 128],
        head_width=128,
        head_depth=3,
        expand_ratio=1,
        middle_op="fmbconv",
        out_dim=prompt_embed_dim,
        out_size=image_embedding_size,  # Results in img_size mask outputs from the decoder
    )
    image_encoder = EfficientViTSamImageEncoder(backbone, neck, img_size=img_size)
    return build_sam(image_encoder, img_size, prompt_embed_dim, **kwargs)


from mmdet.models.backbones import SwinTransformer

class SwinWrapper(nn.Module):
    """Wrap the swin backbone to have the right shape output."""

    def __init__(self, config: dict):
        super().__init__()
        self.swin = SwinTransformer(**config)

    def forward(self, x):
        outputs = self.swin(
            x,
        )

        feature_dict = dict(
            input=x,
            stage1=outputs[0],
            stage2=outputs[1],
            stage3=outputs[2],
            stage4=outputs[3],
            stage_final=outputs[0],
        )

        return feature_dict


def swin_sam_small_patch4_window7(
    img_size: int = 448,
    prompt_embed_dim: int = 256,
    swin_ckpt: str = None,
    **kwargs,
):
    patch_size = 7
    image_embedding_size = img_size // patch_size
    init_cfg = (None if swin_ckpt is None else dict(type="Pretrained", checkpoint=swin_ckpt),)
    config = dict(
        embed_dims=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=init_cfg,
    )

    backbone = SwinWrapper(config)
    neck = SamNeck(
        fid_list=["stage4", "stage3", "stage2", "stage1"],
        in_channel_list=[768, 384, 192, 96],
        head_width=128,
        head_depth=3,
        expand_ratio=1,
        middle_op="fmbconv",
        out_dim=prompt_embed_dim,
        out_size=image_embedding_size,  # Results in img_size mask outputs from the decoder
    )

    image_encoder = EfficientViTSamImageEncoder(backbone, neck, img_size=img_size)
    return build_sam(image_encoder, img_size, prompt_embed_dim, patch_size=patch_size, **kwargs)


def swin_sam_base_patch4_window7(
    img_size: int = 448,
    prompt_embed_dim: int = 256,
    swin_ckpt: str = None,
    **kwargs,
):
    patch_size = 7
    image_embedding_size = img_size // patch_size
    init_cfg = (None if swin_ckpt is None else dict(type="Pretrained", checkpoint=swin_ckpt),)
    config = dict(
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=init_cfg,
    )

    backbone = SwinWrapper(config)
    neck = SamNeck(
        fid_list=["stage4", "stage3", "stage2", "stage1"],
        in_channel_list=[1024, 512, 256, 128],
        head_width=128,
        head_depth=3,
        expand_ratio=1,
        middle_op="fmbconv",
        out_dim=prompt_embed_dim,
        out_size=image_embedding_size,
    )

    image_encoder = EfficientViTSamImageEncoder(backbone, neck, img_size=img_size)
    return build_sam(image_encoder, img_size, prompt_embed_dim, patch_size=patch_size, **kwargs)


def swin_sam_large_patch4_window7(
    img_size: int = 448,
    prompt_embed_dim: int = 256,
    swin_ckpt: str = None,
    **kwargs,
):
    patch_size = 7
    image_embedding_size = img_size // patch_size
    init_cfg = (None if swin_ckpt is None else dict(type="Pretrained", checkpoint=swin_ckpt),)
    config = dict(
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=init_cfg,
    )

    backbone = SwinWrapper(config)
    neck = SamNeck(
        fid_list=["stage4", "stage3", "stage2", "stage1"],
        in_channel_list=[1536, 768, 384, 192],
        head_width=128,
        head_depth=3,
        expand_ratio=1,
        middle_op="fmbconv",
        out_dim=prompt_embed_dim,
        out_size=image_embedding_size,
    )

    image_encoder = EfficientViTSamImageEncoder(backbone, neck, img_size=img_size)
    return build_sam(image_encoder, img_size, prompt_embed_dim, patch_size=patch_size, **kwargs)


def create_sam_model(
    backbone: str,
    pretrained: bool = True,
    weight_path: str | None = None,
    backbone_ckpt: str | None = None,
    **kwargs,
) -> EfficientViTCyclops | EfficientViTHydra:
    """kwargs go to build_sam."""
    model_dict = {
        "b0": efficientvit_sam_b0,
        "b1": efficientvit_sam_b1,
        "b2": efficientvit_sam_b2,
        "b3": efficientvit_sam_b3,
        "swin-s": swin_sam_small_patch4_window7,
        "swin-b": swin_sam_base_patch4_window7,
        "swin-l": swin_sam_large_patch4_window7,
    }

    # TODO: load the model and test backbone weight loading.

    if backbone not in model_dict:
        raise ValueError(
            f"Do not find {backbone} in the model zoo. List of models: {list(model_dict.keys())}"
        )
    else:
        model = model_dict[backbone](**kwargs)

    if pretrained:
        if weight_path is None:
            raise NotImplementedError("TODO: add the url for the pretrained model")

        if weight_path.startswith("https"):
            log.info(f"Downloading weights from {weight_path}")
            weight_path = download(weight_path, f"{backbone}.pt", root="assets/checkpoints/sam")

        if not Path(weight_path).exists():
            raise FileNotFoundError(f"Weight path {weight_path} does not exist.")

        state_dict = load_state_dict_from_file(weight_path)
        new_state_dict = {}
        for name, w in state_dict.items():
            # log.info(f"Changing name from {name} to {name[4:]}")
            if name.startswith("sam."):
                new_state_dict[name[4:]] = w
        model.load_state_dict(new_state_dict, strict=False)
        log.info(
            f"Loaded weights from {weight_path} with keys: {list(new_state_dict.keys())[:5]} into {list(model.state_dict().keys())[:5]}..."
        )

    elif backbone_ckpt is not None and backbone_ckpt.endswith(".ckpt"):
        if not Path(backbone_ckpt).exists():
            raise FileNotFoundError(f"Backbone checkpoint {backbone_ckpt} does not exist.")
        state_dict = load_state_dict_from_file(backbone_ckpt)

        # Want to transfer the weights from "model.backbone" to "model.image_encoder.backbone"
        new_state_dict = {}
        for name, w in state_dict.items():
            if name.startswith("model.backbone"):
                # log.info(f"Changing name from {name} to {name[12:]}")
                new_state_dict[name[15:]] = w

        # log.info(f"model backbone keys: {model.image_encoder.backbone.state_dict().keys()}")
        # log.info(f"checkpoint keys: {new_state_dict.keys()}")
        model.image_encoder.backbone.load_state_dict(new_state_dict, strict=True)
        log.info(f"Loaded backbone weights from {backbone_ckpt}")
    # elif backbone_ckpt is not None and backbone_ckpt.endswith(".pth"):
    #     if not Path(backbone_ckpt).exists():
    #         raise FileNotFoundError(f"Backbone checkpoint {backbone_ckpt} does not exist.")
    #     state_dict = torch.load(backbone_ckpt, map_location="cpu")

    #     print(f"ckpt state_dict keys: {state_dict.keys()}")
    #     print(f"model state_dict keys: {model.image_encoder.backbone.state_dict().keys()}")

    #     # Want to transfer the weights from "model.backbone" to "model.image_encoder.backbone"
    #     new_state_dict = {}
    #     for name, w in state_dict.items():
    #         if name.startswith("backbone"):
    #             # log.info(f"Changing name from {name} to {name[12:]}")
    #             new_state_dict[name[9:]] = w

    #     # log.info(f"model backbone keys: {model.image_encoder.backbone.state_dict().keys()}")
    #     # log.info(f"checkpoint keys: {new_state_dict.keys()}")
    #     model.image_encoder.backbone.load_state_dict(new_state_dict, strict=False)
    #     log.info(f"Loaded backbone weights from {backbone_ckpt}")

    return model
