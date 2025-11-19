import os
import numpy as np
import cv2
from read_write_dense import read_array
import argparse
from tqdm import tqdm
from PIL import Image


def save_depth_and_mask_image(depth, depth_output_path, 
                              depth_thre, min_depth, image_size):
    depth_resized = cv2.resize(depth, image_size, interpolation=cv2.INTER_NEAREST)
    """将深度数据保存为npy格式"""
    mask = (depth_resized <= min_depth) | (depth_resized >= depth_thre).astype(np.uint8) 
    
    depth_resized[depth_resized <= 0] = np.nan # 把0和负数都设置为nan，防止被min_depth取代
    depth_resized[depth_resized < min_depth] = min_depth
    depth_resized[depth_resized > depth_thre] = depth_thre
    
    depth_resized = np.nan_to_num(depth_resized) # nan全都变为0
    depth_resized = depth_resized.astype(np.float32)
    # resize 到统一尺寸
    # 保存为npy格式
    np.save(depth_output_path, depth_resized)


def process_scene(scene_folder, depth_thre = 500.0, min_depth = 0.00, image_size=(480, 270)):
    scene = os.path.join(scene_folder, 'colmap', 'dense', 'stereo', 'depth_maps')
    depth_output_root_path = os.path.join(scene_folder, 'depths')
    mask_output_root_path = os.path.join(scene_folder, 'masks')
    os.makedirs(depth_output_root_path, exist_ok=True)
    
    # 遍历文件夹中的所有文件
    for file_name in tqdm(os.listdir(scene)):
        if file_name.endswith('.png.geometric.bin'):
            bin_file_path = os.path.join(scene, file_name)
            
                # 如果是空文件，则删除并跳过
            if os.path.getsize(bin_file_path) == 0:
                # print(f"🗑️ 空文件已删除: {bin_file_path}")
                os.remove(bin_file_path)
                continue
            
            try:
                depth = read_array(bin_file_path)
            except Exception as e: 
                # print(f"❌ 读取深度图失败: {bin_file_path}, 错误: {e}")
                os.remove(bin_file_path)
                continue
            
            # 生成深度图文件路径
            depth_output_path = os.path.join(depth_output_root_path, file_name.replace('.png.geometric.bin', '.npy'))
            save_depth_and_mask_image(depth, depth_output_path,
                                      depth_thre = depth_thre, min_depth = min_depth, image_size = image_size)
            

def main(root_path, scene_id, depth_thre = 500.0,  min_depth = 0.00):
    # 设置根目录
    if isinstance(depth_thre, str):
        depth_thre = eval(depth_thre)
    if isinstance(min_depth, str):
        min_depth = eval(min_depth)
    scene_folder = os.path.join(root_path, scene_id)
    image_folder = os.path.join(scene_folder, 'images_8')
    any_image = os.listdir(image_folder)[0]
    impath = os.path.join(image_folder, any_image)
    rgb_image = Image.open(impath)
    rgb_image = rgb_image.convert("RGB")
    image_size = rgb_image.size  # (W,H)
    # 遍历ROOT_DIR目录下的每个scene文件夹
    process_scene(scene_folder, depth_thre = depth_thre,
                    min_depth = min_depth, image_size = image_size)
    print("Generated Masks and Depths.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", default="datasets/dl3dv_finished/1K", help="Root path to scenes (e.g., datasets/DL3DV/1K)")
    parser.add_argument("--scene_id", default="7103edc158a862dbfa3c3454e4de584dad59c3c30055919f1dfa7fd7acfdd5c9", help="Scene ID (e.g., scene_0)")
    parser.add_argument("--depth_thre", default=100.0, help="max depth")
    parser.add_argument("--min_depth", default=0.00, help="min depth")

    args = parser.parse_args()

    main(
        args.root_path,
        args.scene_id,
        args.depth_thre,
        args.min_depth,
    )