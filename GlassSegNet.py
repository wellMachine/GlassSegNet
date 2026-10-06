
import torch
import torch.nn as nn
import torch.nn.functional as F
from dinov3.dinov3.hub.backbones import dinov3_vitb16


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Decoder(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2=None):
        if x2 is not None:
            if torch.onnx.is_in_onnx_export():
                x2 = F.interpolate(x2, size=(x1.size(2), x1.size(3)), mode='bilinear', align_corners=False)
                x = torch.cat([x1, x2], dim=1)
            else:
                diffY = x1.size()[2] - x2.size()[2]
                diffX = x1.size()[3] - x2.size()[3]
                x2 = F.pad(x2, [diffX // 2, diffX - diffX // 2,
                                diffY // 2, diffY - diffY // 2])
                x = torch.cat([x1, x2], dim=1)
        else:
            x = x1
        x = self.up(x)
        return self.conv(x)


class Adapter(nn.Module):
    def __init__(self, blk) -> None:
        super(Adapter, self).__init__()
        self.block = blk
        dim = blk.attn.qkv.in_features
        self.prompt_learn = nn.Sequential(
            nn.Linear(dim, 32),
            nn.GELU(),
            nn.Linear(32, dim),
            nn.GELU()
        )

    def forward(self, x_or_x_list, rope_or_rope_list=None):
        # 支持张量与列表两种输入形式，保持与 SelfAttentionBlock.forward 一致
        if isinstance(x_or_x_list, torch.Tensor):
            prompt = self.prompt_learn(x_or_x_list)
            promped = x_or_x_list + prompt
            return self.block(promped, rope_or_rope_list)
        else:
            promped_list = [t + self.prompt_learn(t) for t in x_or_x_list]
            return self.block(promped_list, rope_or_rope_list)

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class MSGRL(nn.Module):
    """Refine one feature level using the existing multi-branch convolution design."""

    def __init__(self, in_channel, out_channel):
        super(MSGRL, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4 * out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))
        x = self.relu(x_cat + self.conv_res(x))
        return x


class GlassSegNet(nn.Module):
    def __init__(self, checkpoint_path=None, dinov3_path=None, dinov2_path=None) -> None:
        super(GlassSegNet, self).__init__()

        # ===== DINOv3-B Encoder =====
        weight_path = dinov3_path if dinov3_path is not None else dinov2_path
        if weight_path:
            # 使用本地 DINOv3-B 权重（ViT-B/16）。不传 img_size/patch_size，避免与内部默认参数冲突。
            self.dino = dinov3_vitb16(
                pretrained=True,
                weights=weight_path,
            )
        else:
            # 若未提供路径，则使用默认权重（会从官方地址下载）
            self.dino = dinov3_vitb16(
                pretrained=True,
            )
        for param in self.dino.parameters():
            param.requires_grad = False
            # wrap blocks with Adapter for prompt tuning
        self.dino.blocks = nn.ModuleList([Adapter(blk) for blk in self.dino.blocks])

        # DINOv3-B 的通道维度为 768（ViT-B/16）
        self.align1 = nn.Conv2d(768, 144, 1)
        self.align2 = nn.Conv2d(768, 288, 1)
        self.align3 = nn.Conv2d(768, 576, 1)
        self.align4 = nn.Conv2d(768, 1152, 1)

        # Apply multi-branch convolutional refinement at each feature-pyramid level.

        self.msgrl1 = MSGRL(144, 64)
        self.msgrl2 = MSGRL(288, 64)
        self.msgrl3 = MSGRL(576, 64)
        self.msgrl4 = MSGRL(1152, 64)

        # Each MSGRL block outputs 64 channels for the decoder skip connections.
        self.up4 = Decoder(64, 128)
        self.up3 = Decoder(192, 128)  # 128 (来自上一路) + 64 (跳连)
        self.up2 = Decoder(192, 128)
        self.up1 = Decoder(192, 128)
        self.head = nn.Conv2d(128, 1, 1)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict,
        missing_keys, unexpected_keys, error_msgs,
    ):
        # Map legacy refinement-block names when loading existing checkpoints.
        for level in range(1, 5):
            old_prefix = f"{prefix}dgl{level}."
            new_prefix = f"{prefix}msgrl{level}."
            for key in list(state_dict):
                if key.startswith(old_prefix):
                    new_key = new_prefix + key[len(old_prefix):]
                    if new_key in state_dict:
                        error_msgs.append(
                            f"Checkpoint contains both {key!r} and {new_key!r}."
                        )
                    else:
                        state_dict[new_key] = state_dict[key]
                    del state_dict[key]
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )

    def forward(self, x):
        b, c, h, w = x.shape
        # 直接在输入尺寸（例如 352x352）下提取中间特征，并重排为 [B, C, H/16, W/16]
        feats = self.dino.get_intermediate_layers(
            x,
            n=[2, 5, 8, 11],
            reshape=True,
            return_class_token=False,
            return_extra_tokens=False,
            norm=True,
        )
        f2, f5, f8, f11 = feats[0], feats[1], feats[2], feats[3]

        s4_size = (h // 16, w // 16)
        s3_size = (h // 8, w // 8)
        s2_size = (h // 4, w // 4)
        s1_size = (h // 2, w // 2)

        x4_d = F.interpolate(self.align4(f11), size=s4_size, mode='bilinear', align_corners=False)
        x3_d = F.interpolate(self.align3(f8), size=s3_size, mode='bilinear', align_corners=False)
        x2_d = F.interpolate(self.align2(f5), size=s2_size, mode='bilinear', align_corners=False)
        x1_d = F.interpolate(self.align1(f2), size=s1_size, mode='bilinear', align_corners=False)

        x4 = self.msgrl4(x4_d)
        x3 = self.msgrl3(x3_d)
        x2 = self.msgrl2(x2_d)
        x1 = self.msgrl1(x1_d)

        x = self.up4(x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        out = self.head(x)
        # 保证输出与输入尺寸一致，避免尺寸不整除导致的误差
        if out.shape[-2:] != (h, w):
            out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)
        return out
