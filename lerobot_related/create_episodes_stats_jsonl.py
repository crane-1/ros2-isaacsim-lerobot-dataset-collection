import pandas as pd
import os
from os import listdir
from os.path import isfile, join
import pyarrow as pa
import pyarrow.parquet as pq
import json
import numpy as np
from pathlib import Path
from PIL import Image as PILImage
import cv2
import shutil

def estimate_num_samples(
    dataset_len: int, min_num_samples: int = 100, max_num_samples: int = 10_000, power: float = 0.75
) -> int:
    """Heuristic to estimate the number of samples based on dataset size.
    The power controls the sample growth relative to dataset size.
    Lower the power for less number of samples.

    For default arguments, we have:
    - from 1 to ~500, num_samples=100
    - at 1000, num_samples=177
    - at 2000, num_samples=299
    - at 5000, num_samples=594
    - at 10000, num_samples=1000
    - at 20000, num_samples=1681
    """
    if dataset_len < min_num_samples:
        min_num_samples = dataset_len
    return max(min_num_samples, min(int(dataset_len**power), max_num_samples))

def load_image_as_numpy(
    fpath: str | Path, dtype: np.dtype = np.float32, channel_first: bool = True
) -> np.ndarray:
    img = PILImage.open(fpath).convert("RGB")
    img_array = np.array(img, dtype=dtype)
    if channel_first:  # (H, W, C) -> (C, H, W)
        img_array = np.transpose(img_array, (2, 0, 1))
    if np.issubdtype(dtype, np.floating):
        img_array /= 255.0
    return img_array

def sample_indices(data_len: int) -> list[int]:
    num_samples = estimate_num_samples(data_len)
    return np.round(np.linspace(0, data_len - 1, num_samples)).astype(int).tolist()


def auto_downsample_height_width(img: np.ndarray, target_size: int = 150, max_size_threshold: int = 300):
    _, height, width = img.shape

    if max(width, height) < max_size_threshold:
        # no downsampling needed
        return img

    downsample_factor = int(width / target_size) if width > height else int(height / target_size)
    return img[:, ::downsample_factor, ::downsample_factor]


def sample_images(image_paths: list[str]) -> np.ndarray:
    if not image_paths:
        return np.array([]) # 容错处理
    
    sampled_indices = sample_indices(len(image_paths))
    images = None
    for i, idx in enumerate(sampled_indices):
        path = image_paths[idx]
        img = load_image_as_numpy(path, dtype=np.uint8, channel_first=True)
        img = auto_downsample_height_width(img)
        if images is None:
            images = np.empty((len(sampled_indices), *img.shape), dtype=np.uint8)
        images[i] = img
    return images


def get_feature_stats(array: np.ndarray, axis: tuple, keepdims: bool) -> dict[str, np.ndarray]:
    return {
        "min": np.min(array, axis=axis, keepdims=keepdims).tolist(),
        "max": np.max(array, axis=axis, keepdims=keepdims).tolist(),
        "mean": np.mean(array, axis=axis, keepdims=keepdims).tolist(),
        "std": np.std(array, axis=axis, keepdims=keepdims).tolist(),
        "count": np.array([len(array)]).tolist(),
    }


def get_feature_stats_img(array: np.ndarray, axis: tuple, keepdims: bool) -> dict[str, np.ndarray]:
    return {
        "min": np.min(array, axis=axis, keepdims=keepdims),
        "max": np.max(array, axis=axis, keepdims=keepdims),
        "mean": np.mean(array, axis=axis, keepdims=keepdims),
        "std": np.std(array, axis=axis, keepdims=keepdims),
        "count": np.array([len(array)]),
    }

base_dir = '/home/qwe/isaacsim_piper/isaacsim_piper/isaacsim_piper/my_data/'
mypath = os.path.join(base_dir, 'data/chunk_000')
video_base_path = os.path.join(base_dir, 'videos/chunk_000')
meta_dir = os.path.join(base_dir, 'meta')

onlyfiles = [f for f in listdir(mypath) if f.endswith('.parquet')]
onlyfiles.sort()

jsonl_data = []

# 定义你需要统计的图像观测模态
# camera_names = ['observation.images.top', 'observation.images.wrist', 'observation.images.side']  # 根据你的实际模态名称调整
camera_names = ['observation.images.top_front', 'observation.images.top_back','observation.images.wrist_l', 'observation.images.wrist_r']

for file in onlyfiles:
    df = pd.read_parquet(os.path.join(mypath, file))
    print(f"Processing file: {file}")
    
    episode_dic = {}
    episode_dic['episode_index'] = int(df['episode_index'].iloc[0])
    episode_dic['stats'] = {}

    # 为每一个相机模态计算统计信息
    for cam in camera_names:
        video_path = os.path.join(video_base_path, cam, file.replace('.parquet', '.mp4'))
        temp_img_path = os.path.join(video_base_path, cam, 'temp_imgs')
        
        # 目录清理与创建
        if os.path.exists(temp_img_path):
            shutil.rmtree(temp_img_path)
        os.makedirs(temp_img_path, exist_ok=True)

        print(f"  Reading video: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"  [Error] Could not open video: {video_path}")
            continue

        img_index = 0
        while True:
            ret, cv2_img = cap.read()
            if not ret: break
            img_name = os.path.join(temp_img_path, f"img{img_index:04d}.png")
            cv2.imwrite(img_name, cv2_img)
            img_index += 1
        cap.release()

        # 获取生成的图片路径
        img_paths = [join(temp_img_path, f) for f in listdir(temp_img_path) if f.endswith('.png')]
        img_paths.sort()

        if len(img_paths) > 0:
            ep_ft_array = sample_images(img_paths)
            temp_video_stats = get_feature_stats_img(ep_ft_array, axis=(0, 2, 3), keepdims=True)
            video_stats = {k: v if k == "count" else np.squeeze(v / 255.0, axis=0) for k, v in temp_video_stats.items()}
            video_stats = {k: v.tolist() for k, v in video_stats.items()}
            episode_dic['stats'][cam] = video_stats
        
        shutil.rmtree(temp_img_path)

    # 统计 Observation State (关节位置)
    # 假设它是 list 格式存储在 parquet 中
    obs_state_data = np.stack(df['observation.state'].values)
    episode_dic['stats']['observation.state'] = get_feature_stats(obs_state_data, axis=0, keepdims=False)

    # 统计 Action
    action_data = np.stack(df['action'].values)
    episode_dic['stats']['action'] = get_feature_stats(action_data, axis=0, keepdims=False)

    # 统计其他标量数据 (转为 numpy 计算)
    for key in ['episode_index', 'frame_index', 'timestamp', 'index']:
        data = df[key].to_numpy().astype(float)
        episode_dic['stats'][key] = get_feature_stats(data, axis=0, keepdims=True)

    # 特殊处理 task_index (如果有的话)
    if 'task_index' in df.columns:
        episode_dic['stats']['task_index'] = get_feature_stats(df['task_index'].to_numpy().astype(float), axis=0, keepdims=True)

    jsonl_data.append(episode_dic)

# 保存
output_file = os.path.join(meta_dir, 'episodes_stats.jsonl')

with open(output_file, 'w') as f:
    for l in jsonl_data:
        f.write(json.dumps(l) + "\n")
print(f"Done! Stats saved to {output_file}")
