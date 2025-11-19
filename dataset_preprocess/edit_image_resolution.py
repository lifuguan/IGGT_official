from PIL import Image  # 新增导入
import os
import shutil
import struct
import argparse
from read_write_model import read_images_binary

def get_image_size(image_path):
    with Image.open(image_path) as img:
        return img.width, img.height

CAMERA_MODEL_IDS = {
    'SIMPLE_PINHOLE': 0,
    'PINHOLE': 1,
    'SIMPLE_RADIAL': 2,
    'RADIAL': 3,
    'OPENCV': 4,
    'FULL_OPENCV': 5,
    'FOV': 6,
    'SIMPLE_RADIAL_FISHEYE': 7,
    'RADIAL_FISHEYE': 8,
    'OPENCV_FISHEYE': 9,
    'FOV_FISHEYE': 10,
    'THIN_PRISM_FISHEYE': 11,
}

def get_num_params(model_id):
    param_counts = {
        0: 3,
        1: 4,
        2: 4,
        3: 5,
        4: 8,
        5: 12,
        6: 5,
        7: 4,
        8: 5,
        9: 8,
        10: 5,
        11: 12,
    }
    return param_counts[model_id]

def read_cameras_binary(path):
    with open(path, "rb") as f:
        num_cameras = struct.unpack("Q", f.read(8))[0]
        cameras = {}
        for _ in range(num_cameras):
            camera_id = struct.unpack("i", f.read(4))[0]
            model_id = struct.unpack("i", f.read(4))[0]
            width = struct.unpack("q", f.read(8))[0]
            height = struct.unpack("q", f.read(8))[0]
            num_params = get_num_params(model_id)
            params = struct.unpack("d" * num_params, f.read(8 * num_params))
            cameras[camera_id] = {
                "model_id": model_id,
                "width": width,
                "height": height,
                "params": list(params)
            }
        return cameras

def write_cameras_binary(cameras, path):
    with open(path, "wb") as f:
        f.write(struct.pack("Q", len(cameras)))
        for camera_id, cam in cameras.items():
            f.write(struct.pack("i", camera_id))
            f.write(struct.pack("i", cam["model_id"]))
            f.write(struct.pack("q", cam["width"]))
            f.write(struct.pack("q", cam["height"]))
            f.write(struct.pack("d" * len(cam["params"]), *cam["params"]))

def scale_camera_params(cameras, new_width, new_height, original_width, original_height):
    scale_w = new_width / original_width
    scale_h = new_height / original_height

    for cam in cameras.values():
        cam["width"] = new_width
        cam["height"] = new_height
        model_id = cam["model_id"]

        if model_id == 1:  # PINHOLE
            cam["params"][0] *= scale_w
            cam["params"][1] *= scale_h
            cam["params"][2] *= scale_w
            cam["params"][3] *= scale_h
        elif model_id == 0:  # SIMPLE_PINHOLE
            cam["params"][0] *= scale_w
            cam["params"][1] *= scale_w
            cam["params"][2] *= scale_h
        elif model_id == 4:  # OPENCV
            cam["params"][0] *= scale_w
            cam["params"][1] *= scale_h
            cam["params"][2] *= scale_w
            cam["params"][3] *= scale_h
        else:
            print(f"⚠️ 未支持的相机模型 ID {model_id}，未缩放")

def main(root_path, scene_id):
    camera_path = os.path.join(root_path, scene_id, "colmap/sparse/0/cameras.bin")
    image_path = os.path.join(root_path, scene_id, "colmap/sparse/0/images.bin")
    backup_path = camera_path + ".backup"

    
    images = read_images_binary(image_path)
    if os.path.exists(backup_path):
        print(f"⏩ 已检测到 {backup_path}，跳过缩放。")
        return

    # 读取原始相机参数
    cameras = read_cameras_binary(camera_path)
    print("📐 原始相机分辨率:")
    for cid, cam in cameras.items():
        print(f" - ID {cid}: {cam['width']} x {cam['height']} (Model ID: {cam['model_id']})")
    first_camera = next(iter(cameras.values()))
    original_width = first_camera["width"]
    original_height = first_camera["height"]

    # 从图像获取新分辨率
    image_path = os.path.join(root_path, scene_id, "images_8/frame_00001.png")
    new_width, new_height = get_image_size(image_path)
    print(f"📷 从图像获取新分辨率: {new_width}x{new_height}")

    # 备份
    shutil.copy(camera_path, backup_path)
    print("✅ 备份完成:", backup_path)

    # 缩放参数
    scale_camera_params(cameras, new_width, new_height, original_width, original_height)

    # 写入
    write_cameras_binary(cameras, camera_path)
    print(f"🎯 相机更新完成 → 新分辨率: {new_width}x{new_height}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Root path to scenes (e.g., datasets/DL3DV/1K)")
    parser.add_argument("--scene", required=True, help="Scene ID (e.g., c01f...)")

    args = parser.parse_args()

    main(
        args.root,
        args.scene,
    )