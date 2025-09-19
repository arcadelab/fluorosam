# EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction
# Han Cai, Junyan Li, Muyan Hu, Chuang Gan, Song Han
# International Conference on Computer Vision (ICCV), 2023

import torch
import torch.nn as nn
import logging

from ...models.efficientvit.backbone import (
    EfficientViTBackbone,
    EfficientViTLargeBackbone,
)
from ...models.nn import (
    ConvLayer,
    DAGBlock,
    FusedMBConv,
    IdentityLayer,
    MBConv,
    OpSequential,
    ResidualBlock,
    UpSampleLayer,
)
from ...models.utils import build_kwargs_from_config

__all__ = [
    "EfficientViTSeg",
    "efficientvit_seg_b0",
    "efficientvit_seg_b1",
    "efficientvit_seg_b2",
    "efficientvit_seg_b3",
    "efficientvit_seg_l1",
    "efficientvit_seg_l2",
]


log = logging.getLogger(__name__)


class SegHead(DAGBlock):
    def __init__(
        self,
        fid_list: list[str],
        in_channel_list: list[int],
        stride_list: list[int],
        head_stride: int,
        head_width: int,
        head_depth: int,
        expand_ratio: float,
        middle_op: str,
        final_expand: float | None,
        n_classes: int,
        dropout=0,
        norm="bn2d",
        act_func="hswish",
    ):
        inputs = {}
        for fid, in_channel, stride in zip(fid_list, in_channel_list, stride_list):
            factor = stride // head_stride
            # print(f"SegHead: fid={fid}, in_channel={in_channel}, stride={stride}, factor={factor}")
            if factor == 1:
                inputs[fid] = ConvLayer(in_channel, head_width, 1, norm=norm, act_func=None)
            else:
                inputs[fid] = OpSequential(
                    [
                        ConvLayer(in_channel, head_width, 1, norm=norm, act_func=None),
                        UpSampleLayer(factor=factor),
                    ]
                )

        middle = []
        for _ in range(head_depth):
            if middle_op == "mbconv":
                block = MBConv(
                    head_width,
                    head_width,
                    expand_ratio=expand_ratio,
                    norm=norm,
                    act_func=(act_func, act_func, None),
                )
            elif middle_op == "fmbconv":
                block = FusedMBConv(
                    head_width,
                    head_width,
                    expand_ratio=expand_ratio,
                    norm=norm,
                    act_func=(act_func, None),
                )
            else:
                raise NotImplementedError
            middle.append(ResidualBlock(block, IdentityLayer()))
        middle = OpSequential(middle)

        outputs = {
            "segout": OpSequential(
                [
                    (
                        None
                        if final_expand is None
                        else ConvLayer(
                            head_width, head_width * final_expand, 1, norm=norm, act_func=act_func
                        )
                    ),
                    ConvLayer(
                        head_width * (final_expand or 1),
                        n_classes,
                        1,
                        use_bias=True,
                        dropout=dropout,
                        norm=None,
                        act_func=None,
                    ),
                ]
            )
        }

        super(SegHead, self).__init__(inputs, "add", None, middle=middle, outputs=outputs)


class EfficientViTSeg(nn.Module):
    def __init__(
        self, backbone: EfficientViTBackbone | EfficientViTLargeBackbone, head: SegHead
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feed_dict = self.backbone(x)
        # for k, v in feed_dict.items():
        #     print(f"Backbone output: {k} {v.shape}")
        feed_dict = self.head(feed_dict)
        # for k, v in feed_dict.items():
        #     print(f"Head output: {k} {v.shape}")

        return dict(segs=feed_dict["segout"])


class EfficientViTHydra(nn.Module):
    """A Hydra has many heads. The output is a dict with the same keys as the heads."""

    def __init__(
        self,
        backbone: EfficientViTBackbone | EfficientViTLargeBackbone,
        **heads: SegHead,
    ) -> None:
        """Forward dict will have the same keys as provided here for each head."""
        super().__init__()
        self.backbone = backbone
        self.heads = nn.ModuleDict(heads)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feed_dict = self.backbone(x)
        outputs = {}
        for name, head in self.heads.items():
            feed_dict = head(feed_dict)
            outputs[name] = feed_dict["segout"]

        return outputs


class EfficientViTCyclops(nn.Module):
    """A hydra with one head but stacked outputs."""

    def __init__(
        self,
        backbone: EfficientViTBackbone | EfficientViTLargeBackbone,
        head: SegHead,
        **output_channels: int,
    ) -> None:
        """Forward dict will have the same keys as provided here for each head.

        Make sure output_channels sums to the total number of classes of the head."""
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.output_channels = output_channels

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feed_dict = self.backbone(x)
        feed_dict = self.head(feed_dict)
        y = feed_dict["segout"]
        outputs = {}
        total = 0
        for name, c in self.output_channels.items():
            outputs[name] = y[:, total : total + c]
            total += c

        return outputs


def efficientvit_seg_b0(dataset: str, **kwargs) -> EfficientViTSeg:
    from ...models.efficientvit.backbone import efficientvit_backbone_b0

    backbone = efficientvit_backbone_b0(**kwargs)

    if dataset == "cityscapes":
        head = SegHead(
            fid_list=["stage4", "stage3", "stage2"],
            in_channel_list=[128, 64, 32],
            stride_list=[32, 16, 8],
            head_stride=8,
            head_width=32,
            head_depth=1,
            expand_ratio=4,
            middle_op="mbconv",
            final_expand=4,
            n_classes=19,
            **build_kwargs_from_config(kwargs, SegHead),
        )
    else:
        raise NotImplementedError
    model = EfficientViTSeg(backbone, head)
    return model


def efficientvit_seg_b1(dataset: str, **kwargs) -> EfficientViTSeg:
    from ...models.efficientvit.backbone import efficientvit_backbone_b1

    backbone = efficientvit_backbone_b1(**kwargs)

    if dataset == "cityscapes":
        head = SegHead(
            fid_list=["stage4", "stage3", "stage2"],
            in_channel_list=[256, 128, 64],
            stride_list=[32, 16, 8],
            head_stride=8,
            head_width=64,
            head_depth=3,
            expand_ratio=4,
            middle_op="mbconv",
            final_expand=4,
            n_classes=19,
            **build_kwargs_from_config(kwargs, SegHead),
        )
    elif dataset == "ade20k":
        head = SegHead(
            fid_list=["stage4", "stage3", "stage2"],
            in_channel_list=[256, 128, 64],
            stride_list=[32, 16, 8],
            head_stride=8,
            head_width=64,
            head_depth=3,
            expand_ratio=4,
            middle_op="mbconv",
            final_expand=None,
            n_classes=150,
            **build_kwargs_from_config(kwargs, SegHead),
        )
    else:
        raise NotImplementedError
    model = EfficientViTSeg(backbone, head)
    return model


def efficientvit_seg_b2(dataset: str, **kwargs) -> EfficientViTSeg:
    from ...models.efficientvit.backbone import efficientvit_backbone_b2

    backbone = efficientvit_backbone_b2(**kwargs)

    if dataset == "cityscapes":
        head = SegHead(
            fid_list=["stage4", "stage3", "stage2"],
            in_channel_list=[384, 192, 96],
            stride_list=[32, 16, 8],
            head_stride=8,
            head_width=96,
            head_depth=3,
            expand_ratio=4,
            middle_op="mbconv",
            final_expand=4,
            n_classes=19,
            **build_kwargs_from_config(kwargs, SegHead),
        )
    elif dataset == "ade20k":
        head = SegHead(
            fid_list=["stage4", "stage3", "stage2"],
            in_channel_list=[384, 192, 96],
            stride_list=[32, 16, 8],
            head_stride=8,
            head_width=96,
            head_depth=3,
            expand_ratio=4,
            middle_op="mbconv",
            final_expand=None,
            n_classes=150,
            **build_kwargs_from_config(kwargs, SegHead),
        )
    else:
        raise NotImplementedError
    model = EfficientViTSeg(backbone, head)
    return model


def efficientvit_seg_b3(dataset: str, **kwargs) -> EfficientViTSeg:
    from ...models.efficientvit.backbone import efficientvit_backbone_b3

    backbone = efficientvit_backbone_b3(**kwargs)

    if dataset == "cityscapes":
        head = SegHead(
            fid_list=["stage4", "stage3", "stage2"],
            in_channel_list=[512, 256, 128],
            stride_list=[32, 16, 8],
            head_stride=8,
            head_width=128,
            head_depth=3,
            expand_ratio=4,
            middle_op="mbconv",
            final_expand=4,
            n_classes=19,
            **build_kwargs_from_config(kwargs, SegHead),
        )
    elif dataset == "ade20k":
        head = SegHead(
            fid_list=["stage4", "stage3", "stage2"],
            in_channel_list=[512, 256, 128],
            stride_list=[32, 16, 8],
            head_stride=8,
            head_width=128,
            head_depth=3,
            expand_ratio=4,
            middle_op="mbconv",
            final_expand=None,
            n_classes=150,
            **build_kwargs_from_config(kwargs, SegHead),
        )
    else:
        raise NotImplementedError
    model = EfficientViTSeg(backbone, head)
    return model


def efficientvit_seg_l1(dataset: str, **kwargs) -> EfficientViTSeg:
    from ...models.efficientvit.backbone import efficientvit_backbone_l1

    backbone = efficientvit_backbone_l1(**kwargs)

    if dataset == "cityscapes":
        head = SegHead(
            fid_list=["stage4", "stage3", "stage2"],
            in_channel_list=[512, 256, 128],
            stride_list=[32, 16, 8],
            head_stride=8,
            head_width=256,
            head_depth=3,
            expand_ratio=1,
            middle_op="fmbconv",
            final_expand=None,
            n_classes=19,
            act_func="gelu",
            **build_kwargs_from_config(kwargs, SegHead),
        )
    elif dataset == "ade20k":
        head = SegHead(
            fid_list=["stage4", "stage3", "stage2"],
            in_channel_list=[512, 256, 128],
            stride_list=[32, 16, 8],
            head_stride=8,
            head_width=128,
            head_depth=3,
            expand_ratio=4,
            middle_op="fmbconv",
            final_expand=8,
            n_classes=150,
            act_func="gelu",
            **build_kwargs_from_config(kwargs, SegHead),
        )
    else:
        raise NotImplementedError
    model = EfficientViTSeg(backbone, head)
    return model


def efficientvit_seg_l2(dataset: str, **kwargs) -> EfficientViTSeg:
    from ...models.efficientvit.backbone import efficientvit_backbone_l2

    backbone = efficientvit_backbone_l2(**kwargs)

    if dataset == "cityscapes":
        head = SegHead(
            fid_list=["stage4", "stage3", "stage2"],
            in_channel_list=[512, 256, 128],
            stride_list=[32, 16, 8],
            head_stride=8,
            head_width=256,
            head_depth=5,
            expand_ratio=1,
            middle_op="fmbconv",
            final_expand=None,
            n_classes=19,
            act_func="gelu",
            **build_kwargs_from_config(kwargs, SegHead),
        )
    elif dataset == "ade20k":
        head = SegHead(
            fid_list=["stage4", "stage3", "stage2"],
            in_channel_list=[512, 256, 128],
            stride_list=[32, 16, 8],
            head_stride=8,
            head_width=128,
            head_depth=3,
            expand_ratio=4,
            middle_op="fmbconv",
            final_expand=8,
            n_classes=150,
            act_func="gelu",
            **build_kwargs_from_config(kwargs, SegHead),
        )
    else:
        raise NotImplementedError
    model = EfficientViTSeg(backbone, head)
    return model
