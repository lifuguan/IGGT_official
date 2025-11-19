#!/usr/bin/env python3
"""
Script to copy semantic annotations from obj_ids to processed_scannetpp_fix (multi-threaded)
"""

import os
import shutil
from pathlib import Path
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def get_image_frames(images_dir):
    """Get all image frame names that start with 'frame_' from the images directory"""
    if not os.path.exists(images_dir):
        return []
    
    image_files = []
    for file in os.listdir(images_dir):
        if ( (file.startswith('frame_') and file.endswith('.jpg') or file.endswith('.png'))):
            # Extract frame name (e.g., 'frame_0001' from 'frame_0001.jpg')
            frame_name = os.path.splitext(file)[0]
            image_files.append(frame_name)
    
    return image_files

def copy_single_semantic_annotation(scene_obj_ids_dir, target_obj_ids_dir, frame_name):
    """Copy a single semantic annotation file for a frame. Returns (frame_name, success, error_msg)"""
    possible_names = [
        f"{frame_name}.jpg.pth",
        f"{frame_name}.png.pth",
        f"frame_{frame_name}.jpg.pth",
        f"frame_{frame_name}.png.pth"
    ]
    source_file = None
    for name in possible_names:
        potential_path = os.path.join(scene_obj_ids_dir, name)
        if os.path.exists(potential_path):
            source_file = potential_path
            break

    if source_file:
        target_file = os.path.join(target_obj_ids_dir, os.path.basename(source_file))
        try:
            shutil.copy2(source_file, target_file)
            return (frame_name, True, None)
        except Exception as e:
            return (frame_name, False, f"Error copying {source_file}: {e}")
    else:
        return (frame_name, False, f"Warning: No semantic annotation found for frame {frame_name}")

def copy_semantic_annotations(processed_dir, obj_ids_dir, scene_name, max_workers=8):
    """Copy semantic annotations for a specific scene using multithreading. 
    Returns (success: bool, failed_frames: list)
    """
    scene_processed_dir = os.path.join(processed_dir, scene_name)
    scene_obj_ids_dir = os.path.join(obj_ids_dir, scene_name)
    
    # Check if both directories exist
    if not os.path.exists(scene_processed_dir):
        print(f"Warning: Processed scene directory {scene_processed_dir} does not exist")
        return False, ["scene_dir_missing"]
    
    if not os.path.exists(scene_obj_ids_dir):
        print(f"Warning: Obj_ids scene directory {scene_obj_ids_dir} does not exist")
        return False, ["obj_ids_dir_missing"]
    
    # Get images directory
    images_dir = os.path.join(scene_processed_dir, 'images')
    if not os.path.exists(images_dir):
        print(f"Warning: Images directory {images_dir} does not exist")
        return False, ["images_dir_missing"]
    
    # Create obj_ids directory in processed scene
    target_obj_ids_dir = os.path.join(scene_processed_dir, 'obj_ids')
    os.makedirs(target_obj_ids_dir, exist_ok=True)
    
    # Get all image frames
    image_frames = get_image_frames(images_dir)
    if not image_frames:
        print(f"Warning: No image files found in {images_dir}")
        return False, ["no_image_files"]
    
    # Copy semantic annotations using ThreadPoolExecutor
    copied_count = 0
    failed_frames = []
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_frame = {
            executor.submit(copy_single_semantic_annotation, scene_obj_ids_dir, target_obj_ids_dir, frame_name): frame_name
            for frame_name in image_frames
        }
        for future in tqdm(as_completed(future_to_frame), total=len(image_frames), desc=f"Scene {scene_name}", leave=False):
            frame_name = future_to_frame[future]
            try:
                frame, success, error_msg = future.result()
                if success:
                    copied_count += 1
                else:
                    if error_msg:
                        print(error_msg)
                    failed_frames.append(frame)
            except Exception as e:
                print(f"Exception in copying frame {frame_name}: {e}")
                failed_frames.append(frame_name)
    
    print(f"Scene {scene_name}: Copied {copied_count}/{len(image_frames)} semantic annotations")
    # If any frame failed, the whole scene is considered failed
    if len(failed_frames) > 0:
        return False, failed_frames
    else:
        return True, []

def main():
    parser = argparse.ArgumentParser(description='Copy semantic annotations from obj_ids to processed_scannetpp_fix (multi-threaded)')
    parser.add_argument('--processed_dir', type=str, 
                       default='/data/SamVGGT/datasets/processed_scannetpp_v2',
                       help='Path to processed_scannetpp_fix directory')
    parser.add_argument('--obj_ids_dir', type=str,
                       default='/mnt/juicefs/datasets/vanilla_scannetpp/2d_semantic/obj_ids',
                       help='Path to obj_ids directory')
    parser.add_argument('--scene', type=str, default=None,
                       help='Specific scene to process (if not provided, process all scenes)')
    parser.add_argument('--num_workers', type=int, default=32,
                       help='Number of threads for copying per scene')
    
    args = parser.parse_args()
    
    # Validate directories
    if not os.path.exists(args.processed_dir):
        print(f"Error: Processed directory {args.processed_dir} does not exist")
        return
    
    if not os.path.exists(args.obj_ids_dir):
        print(f"Error: Obj_ids directory {args.obj_ids_dir} does not exist")
        return
    
    if args.scene:
        # Process specific scene
        success, failed_frames = copy_semantic_annotations(args.processed_dir, args.obj_ids_dir, args.scene, max_workers=args.num_workers)
        if success:
            print(f"Successfully processed scene {args.scene}")
        else:
            print(f"Failed to process scene {args.scene}. Failed frames: {failed_frames}")
    else:
        # Process all scenes
        processed_scenes = [d for d in os.listdir(args.processed_dir) 
                          if os.path.isdir(os.path.join(args.processed_dir, d))]
        
        print(f"Found {len(processed_scenes)} scenes to process")
        
        successful_scenes = 0
        failed_scenes = []
        pbar = tqdm(processed_scenes, desc="Processing scenes", unit="scene")
        for scene_name in pbar:
            try:
                success, failed_frames = copy_semantic_annotations(
                    args.processed_dir, args.obj_ids_dir, scene_name, max_workers=args.num_workers)
                if success:
                    successful_scenes += 1
                else:
                    failed_scenes.append((scene_name, failed_frames))
                # Update tqdm postfix with current stats
                pbar.set_postfix(successful=successful_scenes, failed=len(failed_scenes))
            except Exception as e:
                print(f"Error processing scene {scene_name}: {e}")
                failed_scenes.append((scene_name, ["exception"]))
                pbar.set_postfix(successful=successful_scenes, failed=len(failed_scenes))
        
        print(f"Successfully processed {successful_scenes}/{len(processed_scenes)} scenes")
        if failed_scenes:
            print("Failed scenes and their failed frames:")
            for scene, frames in failed_scenes:
                print(f"  Scene: {scene}")
                # print(f"  Scene: {scene}, Failed frames: {frames}")

if __name__ == "__main__":
    main()


#   Scene: 47b37eb6f9
#   Scene: 4ea827f5a1
#   Scene: 5656608266
#   Scene: 56a0ec536c
#   Scene: 5d152fab1b
#   Scene: 5fb5d2dbf2
#   Scene: 6464461276
#   Scene: 646af5e14b
#   Scene: 6855e1ac32
#   Scene: 6b40d1a939
#   Scene: 6cc2231b9c
#   Scene: 712dc47104
#   Scene: 75d29d69b8
#   Scene: 7977624358
#   Scene: 7cd2ac43b4
#   Scene: 7e09430da7
#   Scene: 7f4d173c9c
#   Scene: 824d9cfa6e
#   Scene: 89214f3ca0
#   Scene: 893fb90e89
#   Scene: 8b2c0938d6
#   Scene: 8d563fc2cc
#   Scene: 8e00ac7f59
#   Scene: 94ee15e8ba
#   Scene: 95d525fbfd
#   Scene: 98b4ec142f
#   Scene: 9b74afd2d2
#   Scene: 9f139a318d
#   Scene: a08dda47a8
#   Scene: a1d9da703c
#   Scene: a4e227f506
#   Scene: aaa11940d3
#   Scene: ab6983ae6c
#   Scene: ad2d07fd11
#   Scene: b09431c547
#   Scene: b20a261fdf
#   Scene: b26e64c4b0
#   Scene: b97261909e
#   Scene: bc2fce1d81
#   Scene: bc400d86e1
#   Scene: bd9305480d
#   Scene: bf6e439e38
#   Scene: bfd3fd54d2
#   Scene: c0f5742640
#   Scene: c413b34238
#   Scene: c47168fab2
#   Scene: c5f701a8c7
#   Scene: c856c41c99
#   Scene: cbd4b3055e
#   Scene: cf1ffd871d
#   Scene: d2f44bf242
#   Scene: d6cbe4b28b
#   Scene: d7abfc4b17
#   Scene: dfac5b38df
#   Scene: e0abd740ba
#   Scene: e0de253456
#   Scene: e3ecd49e2b
#   Scene: e8e81396b6
#   Scene: e9ac2fc517
#   Scene: ed2216380b
#   Scene: ef69d58016
#   Scene: f25f5e6f63
#   Scene: f8062cb7ce
#   Scene: f8f12e4e6b
#   Scene: fb05e13ad1
#   Scene: fe1733741f