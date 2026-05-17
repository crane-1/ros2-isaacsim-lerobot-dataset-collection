#!/usr/bin/env python

import argparse
import logging
import shutil
from pathlib import Path
from typing import Any

import jsonlines
import pandas as pd
import pyarrow as pa
import tqdm
from datasets import Dataset, Features, Image

# 导入 LeRobot 必要的组件
from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.utils import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_DATA_FILE_SIZE_IN_MB,
    DEFAULT_DATA_PATH,
    DEFAULT_VIDEO_PATH,
    LEGACY_EPISODES_PATH,
    LEGACY_EPISODES_STATS_PATH,
    LEGACY_TASKS_PATH,
    cast_stats_to_numpy,
    flatten_dict,
    get_parquet_file_size_in_mb,
    get_parquet_num_frames,
    load_info,
    update_chunk_file_indices,
    write_episodes,
    write_info,
    write_stats,
    write_tasks,
    get_file_size_in_mb,
)
from lerobot.datasets.video_utils import concatenate_video_files, get_video_duration_in_s
from lerobot.utils.utils import init_logging

V21 = "v2.1"
V30 = "v3.0"

def load_jsonlines(fpath: Path) -> list[Any]:
    with jsonlines.open(fpath, "r") as reader:
        return list(reader)

def legacy_load_episodes(local_dir: Path) -> dict:
    episodes = load_jsonlines(local_dir / LEGACY_EPISODES_PATH)
    return {item["episode_index"]: item for item in sorted(episodes, key=lambda x: x["episode_index"])}

def legacy_load_episodes_stats(local_dir: Path) -> dict:
    episodes_stats = load_jsonlines(local_dir / LEGACY_EPISODES_STATS_PATH)
    return {
        item["episode_index"]: cast_stats_to_numpy(item["stats"])
        for item in sorted(episodes_stats, key=lambda x: x["episode_index"])
    }

def legacy_load_tasks(local_dir: Path) -> tuple[dict, dict]:
    tasks = load_jsonlines(local_dir / LEGACY_TASKS_PATH)
    tasks = {item["task_index"]: item["task"] for item in sorted(tasks, key=lambda x: x["task_index"])}
    task_to_task_index = {task: task_index for task_index, task in tasks.items()}
    return tasks, task_to_task_index

def get_image_keys(root):
    info = load_info(root)
    features = info["features"]
    return [key for key, ft in features.items() if ft["dtype"] == "image"]

def get_video_keys(root):
    info = load_info(root)
    features = info["features"]
    return [key for key, ft in features.items() if ft["dtype"] == "video"]

def convert_info(root, new_root, data_size, video_size):
    info = load_info(root)
    info["codebase_version"] = V30
    if "total_chunks" in info: del info["total_chunks"]
    if "total_videos" in info: del info["total_videos"]
    info["data_files_size_in_mb"] = data_size
    info["video_files_size_in_mb"] = video_size
    info["data_path"] = DEFAULT_DATA_PATH
    info["video_path"] = DEFAULT_VIDEO_PATH if info.get("video_path") else None
    info["fps"] = int(info["fps"])
    for key in info["features"]:
        if info["features"][key]["dtype"] != "video":
            info["features"][key]["fps"] = info["fps"]
    write_info(info, new_root)

def convert_tasks(root, new_root):
    tasks, _ = legacy_load_tasks(root)
    df_tasks = pd.DataFrame({"task_index": list(tasks.keys())}, index=list(tasks.values()))
    write_tasks(df_tasks, new_root)

def concat_data_files(paths_to_cat, new_root, chunk_idx, file_idx, image_keys):
    dataframes = [pd.read_parquet(file) for file in paths_to_cat]
    concatenated_df = pd.concat(dataframes, ignore_index=True)
    path = new_root / DEFAULT_DATA_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    schema = pa.Schema.from_pandas(concatenated_df)
    if image_keys:
        features = Features.from_arrow_schema(schema)
        for key in image_keys: features[key] = Image()
        schema = features.arrow_schema
    
    concatenated_df.to_parquet(path, index=False, schema=schema)

def convert_data(root: Path, new_root: Path, data_file_size_in_mb: int):
    data_dir = root / "data"
    ep_paths = sorted(data_dir.glob("*/*.parquet"))
    image_keys = get_image_keys(root)

    ep_idx, chunk_idx, file_idx, size_in_mb, num_frames = 0, 0, 0, 0, 0
    paths_to_cat, episodes_metadata = [], []

    for ep_path in tqdm.tqdm(ep_paths, desc="Converting data files"):
        ep_size_in_mb = get_parquet_file_size_in_mb(ep_path)
        ep_num_frames = get_parquet_num_frames(ep_path)
        episodes_metadata.append({
            "episode_index": ep_idx,
            "data/chunk_index": chunk_idx,
            "data/file_index": file_idx,
            "dataset_from_index": num_frames,
            "dataset_to_index": num_frames + ep_num_frames,
        })
        size_in_mb += ep_size_in_mb
        num_frames += ep_num_frames
        ep_idx += 1

        if size_in_mb < data_file_size_in_mb:
            paths_to_cat.append(ep_path)
        else:
            if paths_to_cat: concat_data_files(paths_to_cat, new_root, chunk_idx, file_idx, image_keys)
            size_in_mb, paths_to_cat = ep_size_in_mb, [ep_path]
            chunk_idx, file_idx = update_chunk_file_indices(chunk_idx, file_idx, DEFAULT_CHUNK_SIZE)

    if paths_to_cat: concat_data_files(paths_to_cat, new_root, chunk_idx, file_idx, image_keys)
    return episodes_metadata

def convert_videos_of_camera(root: Path, new_root: Path, video_key: str, video_file_size_in_mb: int):
    videos_dir = root / "videos"
    ep_paths = sorted(videos_dir.glob(f"*/{video_key}/*.mp4"))
    
    ep_idx, chunk_idx, file_idx, size_in_mb, duration_in_s = 0, 0, 0, 0, 0.0
    paths_to_cat, episodes_metadata = [], []

    for ep_path in tqdm.tqdm(ep_paths, desc=f"Converting videos: {video_key}"):
        ep_size_in_mb = get_file_size_in_mb(ep_path)
        ep_duration_in_s = get_video_duration_in_s(ep_path)

        if size_in_mb + ep_size_in_mb >= video_file_size_in_mb and paths_to_cat:
            target_path = new_root / DEFAULT_VIDEO_PATH.format(video_key=video_key, chunk_index=chunk_idx, file_index=file_idx)
            concatenate_video_files(paths_to_cat, target_path)
            for i in range(len(paths_to_cat)):
                idx = ep_idx - len(paths_to_cat) + i
                episodes_metadata[idx][f"videos/{video_key}/chunk_index"] = chunk_idx
                episodes_metadata[idx][f"videos/{video_key}/file_index"] = file_idx
            chunk_idx, file_idx = update_chunk_file_indices(chunk_idx, file_idx, DEFAULT_CHUNK_SIZE)
            size_in_mb, duration_in_s, paths_to_cat = 0, 0.0, []

        episodes_metadata.append({
            "episode_index": ep_idx,
            f"videos/{video_key}/from_timestamp": duration_in_s,
            f"videos/{video_key}/to_timestamp": duration_in_s + ep_duration_in_s,
        })
        paths_to_cat.append(ep_path)
        size_in_mb += ep_size_in_mb
        duration_in_s += ep_duration_in_s
        ep_idx += 1

    if paths_to_cat:
        target_path = new_root / DEFAULT_VIDEO_PATH.format(video_key=video_key, chunk_index=chunk_idx, file_index=file_idx)
        concatenate_video_files(paths_to_cat, target_path)
        for i in range(len(paths_to_cat)):
            idx = ep_idx - len(paths_to_cat) + i
            episodes_metadata[idx][f"videos/{video_key}/chunk_index"] = chunk_idx
            episodes_metadata[idx][f"videos/{video_key}/file_index"] = file_idx

    return episodes_metadata

def convert_videos(root, new_root, video_size):
    video_keys = sorted(get_video_keys(root))
    if not video_keys: return None
    
    metadata_per_cam = [convert_videos_of_camera(root, new_root, cam, video_size) for cam in video_keys]
    num_episodes = len(metadata_per_cam[0])
    episods_metadata = []
    for ep_idx in range(num_episodes):
        ep_dict = {"episode_index": ep_idx}
        for cam_idx in range(len(video_keys)):
            ep_dict.update(metadata_per_cam[cam_idx][ep_idx])
        episods_metadata.append(ep_dict)
    return episods_metadata

def convert_episodes_metadata(root, new_root, episodes_metadata, episodes_video_metadata=None):
    episodes_legacy = legacy_load_episodes(root)
    ep_stats = legacy_load_episodes_stats(root)
    
    def gen():
        for i in range(len(episodes_metadata)):
            ep_dict = {**episodes_metadata[i], **(episodes_video_metadata[i] if episodes_video_metadata else {}), 
                       **list(episodes_legacy.values())[i], **flatten_dict({"stats": list(ep_stats.values())[i]})}
            ep_dict["meta/episodes/chunk_index"], ep_dict["meta/episodes/file_index"] = 0, 0
            yield ep_dict

    ds = Dataset.from_generator(gen)
    write_episodes(ds, new_root)
    write_stats(aggregate_stats(list(ep_stats.values())), new_root)

def run_local_conversion(input_dir: str, output_dir: str, data_size: int, video_size: int):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if output_path.exists():
        print(f"警告: 输出目录 {output_path} 已存在，正在删除...")
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)

    print(f">>> 开始转换: {input_path} -> {output_path}")
    
    convert_info(input_path, output_path, data_size, video_size)
    convert_tasks(input_path, output_path)
    ep_meta = convert_data(input_path, output_path, data_size)
    video_meta = convert_videos(input_path, output_path, video_size)
    convert_episodes_metadata(input_path, output_path, ep_meta, video_meta)
    
    print(f"\n>>> 转换成功完成！数据保存在: {output_path}")

if __name__ == "__main__":
    init_logging()
    parser = argparse.ArgumentParser(description="LeRobot v2.1 到 v3.0 本地转换工具 (无需网络/HF验证)")
    parser.add_argument("--input-dir", type=str, required=True, help="包含 v2.1 数据的本地目录 (包含 data/, meta/, videos/ 等)")
    parser.add_argument("--output-dir", type=str, required=True, help="转换后 v3.0 数据的保存目录")
    parser.add_argument("--data-size", type=int, default=100, help="数据分片大小 (MB)")
    parser.add_argument("--video-size", type=int, default=500, help="视频分片大小 (MB)")

    args = parser.parse_args()
    run_local_conversion(args.input_dir, args.output_dir, args.data_size, args.video_size)
