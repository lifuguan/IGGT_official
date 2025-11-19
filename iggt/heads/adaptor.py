import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import List, Tuple
from .utils import create_uv_grid, position_grid_to_embed
from detectron2.layers import ShapeSpec
from sam2.modeling.position_encoding import PositionEmbeddingSine

class Projects(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True)
        )

        self.residual_conv = nn.Sequential(
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dim_out)
        )

        self.output_proj = nn.Conv2d(dim_out, dim_out, kernel_size=1)

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.input_proj(x)  # -> [B, dim_out, H, W]
        residual = x
        x = self.residual_conv(x)  # -> [B, dim_out, H, W]
        x += residual  # Add residual connection
        x = self.output_proj(x)  # Final projection
        return x


class GeoProjector(nn.Module):

    def __init__(self, 
        dim_in: int,
        patch_size: int = 14,
        pos_embed: bool = False,
        intermediate_layer_idx: List[int] = [4, 11, 17, 23],
        out_channels: List[int] = [256, 256, 256, 256],
    ):
        super().__init__()

        self.out_channels = out_channels
        self.intermediate_layer_idx = intermediate_layer_idx
        self.patch_size = patch_size
        self.pos_embed = pos_embed

        self.norm = nn.LayerNorm(dim_in)

        # Projection layers for each output channel from tokens.
        self.projects = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=dim_in,
                    out_channels=oc,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
                for oc in out_channels
            ]
        )        

        # Resize layers for upsampling feature maps.
        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    in_channels=out_channels[0], out_channels=out_channels[0], kernel_size=4, stride=4, padding=0
                ),
                nn.ConvTranspose2d(
                    in_channels=out_channels[1], out_channels=out_channels[1], kernel_size=2, stride=2, padding=0
                ),
                nn.Identity(),
                nn.Conv2d(
                    in_channels=out_channels[3], out_channels=out_channels[3], kernel_size=3, stride=2, padding=1
                ),
            ]
        )

    def _apply_pos_embed(self, x: torch.Tensor, W: int, H: int, ratio: float = 0.1) -> torch.Tensor:
        """
        Apply positional embedding to tensor x.
        """
        patch_w = x.shape[-1]
        patch_h = x.shape[-2]
        pos_embed = create_uv_grid(patch_w, patch_h, aspect_ratio=W / H, dtype=x.dtype, device=x.device)
        pos_embed = position_grid_to_embed(pos_embed, x.shape[1])
        pos_embed = pos_embed * ratio
        pos_embed = pos_embed.permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1)
        return x + pos_embed
    
    def forward(
            self,
            aggregated_tokens_list: List[torch.Tensor],
            images: torch.Tensor,
            patch_start_idx: int,
            frames_start_idx: int = None,
            frames_end_idx: int = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if frames_start_idx is not None and frames_end_idx is not None:
            images = images[:, frames_start_idx:frames_end_idx].contiguous()

        B, S, _, H, W = images.shape

        patch_h, patch_w = H // self.patch_size, W // self.patch_size

        out = {}
        dpt_idx = 0
        keys = ["res2", "res3", "res4", "res5"]

        for layer_idx, key in zip(self.intermediate_layer_idx, keys):
            x = aggregated_tokens_list[layer_idx][:, :, patch_start_idx:]

            # Select frames if processing a chunk
            if frames_start_idx is not None and frames_end_idx is not None:
                x = x[:, frames_start_idx:frames_end_idx]

            x = x.view(B * S, -1, x.shape[-1])

            x = self.norm(x)

            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))

            x = self.projects[dpt_idx](x)
            if self.pos_embed:
                x = self._apply_pos_embed(x, W, H)
            x = self.resize_layers[dpt_idx](x)

            out[key] = x
            dpt_idx += 1

        return out

class SamProjector(GeoProjector):

    def __init__(self, 
        dim_in: int,
        patch_size: int = 14,
        pos_embed: bool = False,
        intermediate_layer_idx: List[int] = [4, 11, 17, 23],
        out_channels: List[int] = [256, 256, 256, 256],
    ):
        super().__init__(dim_in, patch_size, pos_embed, intermediate_layer_idx, out_channels)

        # Resize layers for upsampling feature maps.
        self.resize_layers = nn.ModuleList(
            [
            nn.Sequential(
                nn.ConvTranspose2d(in_channels=out_channels[0], out_channels=out_channels[0], kernel_size=4, stride=2, padding=1),
                Projects(dim_in=out_channels[0], dim_out=out_channels[0]),
                nn.ConvTranspose2d(in_channels=out_channels[0], out_channels=out_channels[0], kernel_size=4, stride=2, padding=1),
                Projects(dim_in=out_channels[0], dim_out=out_channels[0]),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(in_channels=out_channels[1], out_channels=out_channels[1], kernel_size=2, stride=2, padding=0),
                Projects(dim_in=out_channels[1], dim_out=out_channels[1])
            ),
            nn.Sequential(
                nn.Identity(),
                Projects(dim_in=out_channels[2], dim_out=out_channels[2])
            ),
            nn.Sequential(
                nn.Conv2d(
                    in_channels=out_channels[3], out_channels=out_channels[3], kernel_size=3, stride=2, padding=1
                ),
                Projects(dim_in=out_channels[3], dim_out=out_channels[3])
            )
            ]
        )
        self.pes = PositionEmbeddingSine(num_pos_feats=256)


    def output_shape(self):
        return {
            "res1": ShapeSpec(channels=self.out_channels[0], stride=2),
            "res2": ShapeSpec(channels=self.out_channels[1], stride=4),
            "res3": ShapeSpec(channels=self.out_channels[2], stride=8),
            "res4": ShapeSpec(channels=self.out_channels[3], stride=16),
        }

    def forward(
            self,
            aggregated_tokens_list: List[torch.Tensor],
            images: torch.Tensor,
            patch_start_idx: int,
            frames_start_idx: int = None,
            frames_end_idx: int = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if frames_start_idx is not None and frames_end_idx is not None:
            images = images[:, frames_start_idx:frames_end_idx].contiguous()

        B, S, _, H, W = images.shape

        patch_h, patch_w = H // self.patch_size, W // self.patch_size

        out, pos = {}, {}
        dpt_idx = 0
        keys = ["res1", "res2", "res3", "res4"]

        for layer_idx, key in zip(self.intermediate_layer_idx, keys):
            x = aggregated_tokens_list[layer_idx][:, :, patch_start_idx:]

            # Select frames if processing a chunk
            if frames_start_idx is not None and frames_end_idx is not None:
                x = x[:, frames_start_idx:frames_end_idx]

            x = x.view(B * S, -1, x.shape[-1])

            x = self.norm(x)

            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))

            x = self.projects[dpt_idx](x)
            x = self.resize_layers[dpt_idx](x)

            out[key] = x
            pos[key] = self.pes(x) # no need should be OK 
            dpt_idx += 1

        return out, pos

