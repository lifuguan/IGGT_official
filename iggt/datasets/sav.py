# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the sav_dataset directory of this source tree.
import json
import os
from typing import Dict, List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pycocotools.mask as mask_util


def show_anns(masks, colors: List, borders=True, ax=None) -> np.ndarray:
    """
    show the annotations and return the RGBA canvas as numpy array
    """
    if len(masks) == 0:
        return None

    sorted_annot_and_color = sorted(
        zip(masks, colors), key=(lambda x: x[0].sum()), reverse=True
    )
    H, W = sorted_annot_and_color[0][0].shape[0], sorted_annot_and_color[0][0].shape[1]

    canvas = np.ones((H, W, 4))
    canvas[:, :, 3] = 0  # set the alpha channel
    contour_thickness = max(1, int(min(5, 0.01 * min(H, W))))
    for mask, color in sorted_annot_and_color:
        canvas[mask] = np.concatenate([color, [0.55]])
        if borders:
            contours, _ = cv2.findContours(
                np.array(mask, dtype=np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
            )
            cv2.drawContours(
                canvas, contours, -1, (0.05, 0.05, 0.05, 1), thickness=contour_thickness
            )

    if ax is not None:
        ax.imshow(canvas)
    return canvas  # RGBA float32 in [0,1]


class SAVDataset:
    """
    SAVDataset is a class to load the SAV dataset and visualize the annotations.
    """

    def __init__(self, sav_dir=None, annot_sample_rate=4):
        self.sav_dir = sav_dir
        self.annot_sample_rate = annot_sample_rate
        self.manual_mask_colors = np.random.random((256, 3))
        self.auto_mask_colors = np.random.random((256, 3))

    def load_annotation_from_json(self, json_path: str) -> Dict:
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Annotation file not found: {json_path}")
        try:
            with open(json_path, 'r') as f:
                annotation_data = json.load(f)
            print(f"Successfully loaded annotation from: {json_path}")
            return annotation_data
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in {json_path}: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading annotation from {json_path}: {e}")

    def sample_masks_from_frames(
        self,
        json_path: str,
        frame_ids: Union[List[int], int],
        mask_sample_num: Optional[int] = None,
        random_seed: Optional[int] = None,
    ) -> Dict[int, List[np.ndarray]]:
        """
        指定N帧并采样他们的mask

        Args:
            json_path: Path to the annotation JSON file
            frame_ids: List of frame indices (or a single int) to sample masks from
            mask_sample_num: If set, sample up to this many masks per frame (randomly)
            random_seed: Optional random seed for reproducibility

        Returns:
            Dict mapping frame_id to list of decoded mask np.ndarrays
        """
        annotation_data = self.load_annotation_from_json(json_path)
        masklet = annotation_data.get("masklet", [])
        if isinstance(frame_ids, int):
            frame_ids = [frame_ids]
        if random_seed is not None:
            np.random.seed(random_seed)
        result = {}
        for frame_id in frame_ids:
            if frame_id < 0 or frame_id >= len(masklet):
                print(f"Frame {frame_id} is out of range.")
                continue
            frame_masks = masklet[frame_id]
            if not frame_masks:
                print(f"Frame {frame_id}: No masks found.")
                result[frame_id] = []
                continue
            indices = np.arange(len(frame_masks))
            if mask_sample_num is not None and mask_sample_num < len(frame_masks):
                indices = np.random.choice(indices, mask_sample_num, replace=False)
            else:
                indices = indices
            sampled_masks = []
            for idx in indices:
                rle = frame_masks[idx]
                try:
                    mask = mask_util.decode(rle) > 0
                    sampled_masks.append(mask)
                except Exception as e:
                    print(f"Warning: Failed to decode mask in frame {frame_id}: {e}")
            result[frame_id] = sampled_masks
        return result

    def visualize_masks_from_json(
        self, 
        json_path: str, 
        show_borders: bool = True,
        save_dir: Optional[str] = None,
        show: bool = True,
        interval: int = 1,
        mp4_path: Optional[str] = None,
        fps: int = 10,
        frame_ids: Optional[List[int]] = None,
        mask_sample_num: Optional[int] = None,
        random_seed: Optional[int] = None,
    ) -> None:
        """
        可视化一整个视频的所有帧的mask，支持mp4输出
        新增功能：可指定frame_ids和每帧采样mask数量

        Args:
            json_path: Path to the annotation JSON file
            show_borders: Whether to show mask borders (default: True)
            save_dir: If set, save each frame visualization to this directory (as PNG)
            show: If True, display the visualization (default: True)
            interval: Visualize every Nth frame (default: 1)
            mp4_path: If set, save the visualization as an mp4 video
            fps: FPS for mp4 video output
            frame_ids: Optional[List[int]], 指定要可视化的帧索引
            mask_sample_num: Optional[int], 每帧采样的mask数量
            random_seed: Optional[int], 随机种子
        """
        annotation_data = self.load_annotation_from_json(json_path)
        if "masklet" not in annotation_data:
            raise ValueError("No 'masklet' field found in annotation data")
        masklet = annotation_data["masklet"]
        num_frames = len(masklet)
        if num_frames == 0:
            print("No frames found in annotation")
            return

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

        # Prepare for mp4 output
        video_writer = None
        video_size = None

        # Determine which frames to visualize
        if frame_ids is not None:
            selected_frame_ids = frame_ids
        else:
            selected_frame_ids = list(range(0, num_frames, interval))

        if random_seed is not None:
            np.random.seed(random_seed)

        for frame_id in selected_frame_ids:
            if frame_id < 0 or frame_id >= num_frames:
                print(f"Frame {frame_id} is out of range, skipping.")
                continue
            frame_masks = masklet[frame_id]
            if len(frame_masks) == 0:
                print(f"Frame {frame_id}: No masks found")
                continue

            indices = np.arange(len(frame_masks))
            if mask_sample_num is not None and mask_sample_num < len(frame_masks):
                indices = np.random.choice(indices, mask_sample_num, replace=False)
            else:
                indices = indices

            colors = self.manual_mask_colors[:len(indices)]
            masks = []
            for idx in indices:
                rle = frame_masks[idx]
                try:
                    mask = mask_util.decode(rle) > 0
                    masks.append(mask)
                except Exception as e:
                    print(f"Warning: Failed to decode mask: {e}")
                    continue

            if len(masks) == 0:
                print(f"Frame {frame_id}: No valid masks could be decoded")
                continue

            # Generate RGBA canvas
            canvas = show_anns(masks, colors, borders=show_borders, ax=None)
            if canvas is None:
                continue

            # Convert RGBA float32 [0,1] to BGR uint8 for OpenCV
            canvas_rgb = (canvas[:, :, :3] * 255).astype(np.uint8)
            alpha = canvas[:, :, 3:4]
            # White background
            bg = np.ones_like(canvas_rgb, dtype=np.uint8) * 255
            canvas_rgb = (canvas_rgb * alpha + bg * (1 - alpha)).astype(np.uint8)
            canvas_bgr = cv2.cvtColor(canvas_rgb, cv2.COLOR_RGB2BGR)

            if video_size is None:
                video_size = (canvas_bgr.shape[1], canvas_bgr.shape[0])

            if save_dir is not None:
                out_path = os.path.join(save_dir, f"frame_{frame_id:04d}.png")
                cv2.imwrite(out_path, canvas_bgr)
                print(f"Saved: {out_path}")

            if mp4_path is not None:
                if video_writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(mp4_path, fourcc, fps, video_size)
                video_writer.write(canvas_bgr)

            # 不再弹出可视化窗口
            # if show:
            #     # Show using OpenCV for speed
            #     cv2.imshow("Masks Visualization", canvas_bgr)
            #     key = cv2.waitKey(int(1000 / fps))
            #     if key == 27:  # ESC to break
            #         break

        if video_writer is not None:
            video_writer.release()
            print(f"MP4 saved to: {mp4_path}")
        # 不再销毁窗口
        # if show:
        #     cv2.destroyAllWindows()

    def get_masks_info(self, json_path: str) -> Dict:
        annotation_data = self.load_annotation_from_json(json_path)
        info = {
            "total_frames": len(annotation_data.get("masklet", [])),
            "frames_with_masks": 0,
            "total_masks": 0,
            "masks_per_frame": []
        }
        for frame_id, frame_masks in enumerate(annotation_data.get("masklet", [])):
            num_masks = len(frame_masks)
            info["total_masks"] += num_masks
            info["masks_per_frame"].append(num_masks)
            if num_masks > 0:
                info["frames_with_masks"] += 1
        return info


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize SAVDataset annotation")
    parser.add_argument("--json_path", type=str, default="/mnt/juicefs/sam2_results/1K/00534f5868a6f72e77befbdb06e35ee9dc34e175dddf0e64e8b1922e494c8e24/auto_masks.json")
    parser.add_argument("--frame_id", type=int, default=0, help="Frame id to visualize (only used for single-frame mode)")
    parser.add_argument("--frame_ids", type=str, default=None, help="Comma separated list of frame ids to visualize, e.g. 0,5,10")
    parser.add_argument("--mask_sample_num", type=int, default=None, help="Sample up to N masks per frame (randomly)")
    parser.add_argument("--random_seed", type=int, default=None, help="Random seed for mask sampling")
    parser.add_argument("--show_borders", action="store_true", help="Show mask borders")
    parser.add_argument("--info_only", action="store_true", help="Only show mask information without visualization")
    parser.add_argument("--all_frames", action="store_true", help="Visualize all frames in the video")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save all frame visualizations")
    parser.add_argument("--interval", type=int, default=1, help="Visualize every Nth frame (default: 1)")
    parser.add_argument("--mp4_path", type=str, default="tests/vis.mp4", help="Path to save mp4 visualization")
    parser.add_argument("--fps", type=int, default=10, help="FPS for mp4/video visualization")
    args = parser.parse_args()

    # Parse frame_ids if provided
    frame_ids = None
    if args.frame_ids is not None:
        frame_ids = [int(x) for x in args.frame_ids.split(",") if x.strip() != ""]

    # Initialize dataset
    dataset = SAVDataset()
    args.all_frames = True
    try:
        # 强制保存成mp4，不弹出窗口
        dataset.visualize_masks_from_json(
            json_path=args.json_path,
            show_borders=args.show_borders,
            save_dir=args.save_dir,
            show=False,  # 不弹出窗口
            interval=args.interval,
            mp4_path=args.mp4_path,
            fps=args.fps,
            frame_ids=frame_ids,
            mask_sample_num=args.mask_sample_num,
            random_seed=args.random_seed,
        )
    except Exception as e:
        print(f"Error: {e}")
        print("\nUsage examples:")
        print(f"python {__file__} annotation.json --all_frames")
        print(f"python {__file__} annotation.json --frame_id 5")
        print(f"python {__file__} annotation.json --info_only")
        print(f"python {__file__} annotation.json --all_frames --save_dir ./vis")
        print(f"python {__file__} annotation.json --all_frames --mp4_path ./vis.mp4")
        print(f"python {__file__} annotation.json --frame_ids 0,5,10 --mask_sample_num 3")
