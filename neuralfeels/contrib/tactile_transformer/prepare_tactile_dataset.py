import os
import shutil
import glob
import cv2
import random

src_root = "/root/autodl-tmp/6998neuralfeels/data/tacto_data"
dst_root = "/root/autodl-tmp/processed_tactile_dataset"
train_ratio = 0.9

# 创建目标目录结构
for split in ["train", "val"]:
    for folder in ["image", "depth", "mask"]:
        os.makedirs(os.path.join(dst_root, split, folder), exist_ok=True)

counter = 0
all_entries = []

# 收集所有图像路径及其 basename
for obj_name in os.listdir(src_root):
    obj_path = os.path.join(src_root, obj_name)
    if not os.path.isdir(obj_path):
        continue
    for seq_name in os.listdir(obj_path):
        seq_path = os.path.join(obj_path, seq_name)
        tactile_img_dir = os.path.join(seq_path, "tactile_images")
        depth_dir = os.path.join(seq_path, "gt_heightmaps")
        mask_dir = os.path.join(seq_path, "gt_contactmasks")
        if not os.path.exists(tactile_img_dir):
            continue

        images = sorted(glob.glob(os.path.join(tactile_img_dir, "*.jpg")))
        for img_path in images:
            basename = os.path.splitext(os.path.basename(img_path))[0]
            all_entries.append((img_path, depth_dir, mask_dir, basename))

# 随机打乱样本
random.shuffle(all_entries)
total = len(all_entries)
train_cutoff = int(train_ratio * total)

for idx, (img_path, depth_dir, mask_dir, basename) in enumerate(all_entries):
    split = "train" if idx < train_cutoff else "val"
    out_basename = f"{idx:05d}"

    # image
    img = cv2.imread(img_path)
    cv2.imwrite(os.path.join(dst_root, split, "image", out_basename + ".jpg"), img)

    # depth (if available)
    depth_path = os.path.join(depth_dir, basename + ".jpg")
    if os.path.exists(depth_path):
        depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(os.path.join(dst_root, split, "depth", out_basename + ".jpg"), depth)

    # mask (if available)
    mask_path = os.path.join(mask_dir, basename + ".jpg")
    if os.path.exists(mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(os.path.join(dst_root, split, "mask", out_basename + ".jpg"), mask)

# print(f"✅ Finished preparing dataset: {train_cutoff} train, {total - train_cutoff} val, total {total} samples.")
import os
import json

def rename_val_images(val_dir: str, col_ext: str = ".jpg"):
    image_dir = os.path.join(val_dir, "image")
    depth_dir = os.path.join(val_dir, "depth")
    mask_dir  = os.path.join(val_dir, "mask")

    # 获取所有图像文件名（带扩展名）
    filenames = sorted([
        f for f in os.listdir(image_dir) if f.endswith(col_ext)
    ])

    print(f"Found {len(filenames)} files to rename.")

    # 保存映射关系：{原文件名（不含扩展）: 新文件名（不含扩展）}
    mapping = {}

    for i, fname in enumerate(filenames):
        base, _ = os.path.splitext(fname)
        new_base = f"{i:05d}"
        new_name = new_base + col_ext

        # 原始路径
        old_image = os.path.join(image_dir, fname)
        old_depth = os.path.join(depth_dir, fname)
        old_mask  = os.path.join(mask_dir,  fname)

        # 新路径
        new_image = os.path.join(image_dir, new_name)
        new_depth = os.path.join(depth_dir, new_name)
        new_mask  = os.path.join(mask_dir,  new_name)

        print(f"[{i:04d}] {base} -> {new_base}")

        # 重命名（如果文件存在才执行）
        if os.path.exists(old_image):
            os.rename(old_image, new_image)
        if os.path.exists(old_depth):
            os.rename(old_depth, new_depth)
        if os.path.exists(old_mask):
            os.rename(old_mask, new_mask)

        mapping[base] = new_base

    # 保存反向映射为 JSON
    map_file = os.path.join(val_dir, "renaming_map.json")
    with open(map_file, "w") as f:
        json.dump(mapping, f, indent=2)

    print(f"✅ Renaming complete. Mapping saved to {map_file}")

# 使用方式
if __name__ == "__main__":
    rename_val_images("/root/autodl-tmp/processed_tactile_dataset/val")