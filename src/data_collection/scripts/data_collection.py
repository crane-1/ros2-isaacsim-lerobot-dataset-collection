#!/usr/bin/python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image
import threading
import copy
import os
import cv2
import numpy as np
from cv_bridge import CvBridge
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import json
import time

bridge = CvBridge()

# 全局变量（原有 + 新增零位控制）
record_data = False
running = True
vid_H = 336
vid_W = 336
wrist_camera_image = np.zeros((vid_H, vid_W, 3), np.uint8)
top_camera_image = np.zeros((vid_H, vid_W, 3), np.uint8)
side_camera_image = np.zeros((vid_H, vid_W, 3), np.uint8)
current_joint_states = [0.0]* 8
msg_pub_to_sim = [0.0] * 8  # 包含 6 个臂关节 + 2 个夹爪关节

# 新增：零位发布控制
time_s = 0.0                      # 强制发布零位到这个时间为止
time_w = 0.0                      # 强制发布零位到这个时间为止
ZERO_DURATION_S = 3.0                       # 按 s 后强制零位 3 秒
ZERO_DURATION_W = 5.0                       # 按 w 后强制零位 5 秒
force_to_zero_flag = False                # 是否正在强制零位
s_pushed = False  # 是否按过 s 键（开始新 episode）

class RobotSubscriber(Node):
    def __init__(self):
        super().__init__('robot_subscriber')
        self.joint_sub = self.create_subscription(JointState, '/joint_states_single', self.joint_callback, 10)
        self.joint_pika_sub = self.create_subscription(JointState, '/joint_states', self.joint_pika_callback, 10)
        self.gripper_sub = self.create_subscription(JointState, '/gripper/joint_state', self.gripper_callback, 10)  # 夹爪状态也触发 recording 开始
        self.wrist_sub = self.create_subscription(Image, '/wrist_rgb', self.wrist_callback, 10)
        self.top_sub = self.create_subscription(Image, '/top_rgb', self.top_callback, 10)
        self.side_sub = self.create_subscription(Image, '/side_rgb', self.side_callback, 10)

        # 新增：发布融合后的 8 关节状态给控制器/仿真
        self.pub_sim = self.create_publisher(JointState, '/joint_states_sim', 10)
        self.create_timer(0.1, self.publish_sim_callback)  # 10Hz 发布

    def joint_callback(self, msg):
        global current_joint_states
        current_joint_states = list(msg.position)
        self.check_start_recording()

    def joint_pika_callback(self, msg: JointState):
        global msg_pub_to_sim
        if len(msg.position) < 6:
            self.get_logger().warn("arm joints < 6, ignoring")
            return
        if force_to_zero_flag:
            msg_pub_to_sim[:6] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        else:
            msg_pub_to_sim[:6] = list(msg.position[:6])
        self.check_start_recording()

    def gripper_callback(self, msg):    
        # 这个回调主要为了确保我们能收到 pika 发布的 gripper/joint_state，触发 recording 的开始
        global msg_pub_to_sim
        if not msg.position:
            self.get_logger().warn("empty gripper position, ignoring")
            return
        if force_to_zero_flag:
            msg_pub_to_sim[6] = 0.0
            msg_pub_to_sim[7] = 0.0
        else:   
            half = msg.position[0] / 2.0
            msg_pub_to_sim[6] = half  # left finger
            msg_pub_to_sim[7] = -half  # right finger
        self.check_start_recording()

    def wrist_callback(self, msg):
        global wrist_camera_image
        try:
            wrist_camera_image = cv2.resize(bridge.imgmsg_to_cv2(msg, "bgr8"), (vid_W, vid_H))
            self.received_data['wrist'] = True
        except: pass

    def top_callback(self, msg):
        global top_camera_image
        try:
            top_camera_image = cv2.resize(bridge.imgmsg_to_cv2(msg, "bgr8"), (vid_W, vid_H))
            self.received_data['top'] = True
        except: pass

    def side_callback(self, msg):
        global side_camera_image
        try:
            side_camera_image = cv2.resize(bridge.imgmsg_to_cv2(msg, "bgr8"), (vid_W, vid_H))
            self.received_data['side'] = True
        except: pass

    def check_start_recording(self):
        global time_s, time_w, record_data, force_to_zero_flag
        if time.time() < time_s or time.time() < time_w:
            return
        if not record_data and s_pushed:
            print("\033[32m--- Starting recording ---\033[0m")
            record_data = True
            force_to_zero_flag = False

    def publish_sim_callback(self):
        """发布给仿真/控制器的关节状态（支持强制零位）"""
        global force_to_zero_flag, msg_pub_to_sim
        if not current_joint_states:
            return

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7', 'joint8']  # 请根据你的 URDF 实际关节名称修改
        msg.velocity = [0.0] * 8
        msg.effort = [0.0] * 8
        if force_to_zero_flag:
            msg_pub_to_sim = [0.0] * 8
        msg.position = msg_pub_to_sim

        self.pub_sim.publish(msg)


class Data_Recorder(Node):
    def __init__(self):
        super().__init__('Data_Recorder')
        self.Hz = 10
        self.timer = self.create_timer(1/self.Hz, self.timer_callback)
        self.recording_active = False
        self.episode_index = 0
        self.total_frames_count = 0
        self.base_dir = "/home/qwe/isaacsim_piper/isaacsim_piper/isaacsim_piper/my_data/"
        self.data_dir = os.path.join(self.base_dir, "data/chunk_000/")
        self.meta_dir = os.path.join(self.base_dir, "meta/")
        self.wrist_vid_dir = os.path.join(self.base_dir, "videos/chunk_000/observation.images.wrist/")
        self.top_vid_dir = os.path.join(self.base_dir, "videos/chunk_000/observation.images.top/")
        self.side_video_dir = os.path.join(self.base_dir, "videos/chunk_000/observation.images.side/")
        print(f"Creating directories in: {self.base_dir}")
        for d in [self.data_dir, self.wrist_vid_dir, self.top_vid_dir, self.side_video_dir, self.meta_dir]:
            os.makedirs(d, exist_ok=True)
        self.reset_buffers()
        self.save_task_jsonl("pick up and place bottle")

    def reset_buffers(self):
        self.df_list = []
        self.frame_index = 0
        self.time_stamp = 0.0
        self.wrist_frames = []
        self.top_frames = []
        self.side_frames = []

    def timer_callback(self):
        global record_data, current_joint_states, wrist_camera_image, top_camera_image, side_camera_image
        if record_data:
            if not self.recording_active:
                print(f'\033[32m--- Start Recording Episode {self.episode_index} ---\033[0m')
                self.reset_buffers()
                self.recording_active = True
            state = np.array(current_joint_states, dtype=np.float32)
            action = np.array(msg_pub_to_sim, dtype=np.float32)
            self.df_list.append({
                'observation.state': state,
                'action': action,
                'timestamp': np.float32(self.time_stamp),
                'frame_index': self.frame_index,
                'episode_index': self.episode_index,
                'index': self.frame_index,
                'task_index': 0
            })
            self.wrist_frames.append(copy.copy(wrist_camera_image))
            self.top_frames.append(copy.copy(top_camera_image))
            self.side_frames.append(copy.copy(side_camera_image))
            self.frame_index += 1
            self.time_stamp += 1/self.Hz
            print(f"Recording frame: {self.frame_index}", end='\r')
        else:
            if self.recording_active:
                self.save_episode()
                self.recording_active = False

    def save_episode(self):
        if not self.df_list:
            return
        print(f'\n\033[34mSaving Episode {self.episode_index} ({self.frame_index} frames)...\033[0m')
        prefix = f"episode_{self.episode_index:06d}"
        try:
            df = pd.DataFrame(self.df_list)
            table = pa.Table.from_pandas(df)
            pq.write_table(table, os.path.join(self.data_dir, f"{prefix}.parquet"))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_w = cv2.VideoWriter(os.path.join(self.wrist_vid_dir, f"{prefix}.mp4"), fourcc, self.Hz, (vid_W, vid_H))
            for f in self.wrist_frames: out_w.write(f)
            out_w.release()
            out_t = cv2.VideoWriter(os.path.join(self.top_vid_dir, f"{prefix}.mp4"), fourcc, self.Hz, (vid_W, vid_H))
            for f in self.top_frames: out_t.write(f)
            out_t.release()
            out_s = cv2.VideoWriter(os.path.join(self.side_video_dir, f"{prefix}.mp4"), fourcc, self.Hz, (vid_W, vid_H))
            for f in self.side_frames: out_s.write(f)
            out_s.release()
            self.total_frames_count += self.frame_index
            self.episode_index += 1
            print(f"\033[32mSuccessfully saved {prefix}\033[0m")
        except Exception as e:
            print(f"\033[31mError saving episode: {e}\033[0m")

    def save_task_jsonl(self, task_name):
        task_path = os.path.join(self.meta_dir, "tasks.jsonl")
        with open(task_path, 'w') as f:
            json.dump({"task_index": 0, "task": task_name}, f)
            f.write('\n')

    def generate_final_info(self):
        if self.recording_active:
            self.save_episode()
            self.recording_active = False
        print(f"\033[33mGenerating final info.json for {self.episode_index} episodes...\033[0m")
        state_dim = len(current_joint_states) if current_joint_states else 8
        info = {
            "codebase_version": "v2.1",
            "robot_type": "piper",
            "total_episodes": self.episode_index,
            "total_frames": self.total_frames_count,
            "total_tasks": 1,
            "total_videos": self.episode_index * 3,  # 改成 3 个相机
            "total_chunks": 1,
            "chunks_size": 1000,
            "fps": self.Hz,
            "splits": {
                "train": f"0:{self.episode_index}"
            },
            "data_path": "data/chunk_000/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk_000/observation.images.{camera_name}/episode_{episode_index:06d}.mp4",
            "features": {
                "observation.state": {
                    "dtype": "float32",
                    "shape": [state_dim],
                    "names": [f"joint_{i+1}" for i in range(state_dim)]
                },
                "action": {
                    "dtype": "float32",
                    "shape": [state_dim],
                    "names": [f"joint_{i+1}" for i in range(state_dim)]
                },
                "observation.images.top": self._vid_info(),
                "observation.images.wrist": self._vid_info(),
                "observation.images.side": self._vid_info(),
                "timestamp": {"dtype": "float32", "shape": [1], "names": None},
                "frame_index": {"dtype": "int64", "shape": [1], "names": None},
                "episode_index": {"dtype": "int64", "shape": [1], "names": None},
                "index": {"dtype": "int64", "shape": [1], "names": None},
                "task_index": {"dtype": "int64", "shape": [1], "names": None}
            }
        }
        with open(os.path.join(self.meta_dir, "info.json"), 'w') as f:
            json.dump(info, f, indent=4)
        print("\033[32minfo.json generated successfully.\033[0m")

    def _vid_info(self):
        return {
            "dtype": "video",
            "shape": [vid_H, vid_W, 3],
            "names": ["height", "width", "channels"],
            "info": {
                "video.fps": float(self.Hz),
                "video.height": vid_H,
                "video.width": vid_W,
                "video.channels": 3,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False
            }
        }


def keyboard_listener(sub_node: RobotSubscriber):
    global record_data, running, force_to_zero_flag, time_s, time_w, s_pushed

    print("\n-------------------------------------------")
    print("[s] Start new episode (force zero 3s → auto record)")
    print("[w] Force zero 5s during recording (still record)")
    print("[q] Stop current episode")
    print("[e] Exit & Save Info")
    print("-------------------------------------------\n")

    while running:
        try:
            user_input = input().lower().strip()
            now = time.time()

            if user_input == 's':
                if record_data:
                    print("已有 episode 正在录制，请先按 q 结束")
                    continue
                print("\033[33m--- 强制零位 3 秒后开始录制 ---\033[0m")
                time_s = now + ZERO_DURATION_S
                force_to_zero_flag = True
                s_pushed = True

            elif user_input == 'w':
                if not record_data:
                    print("当前未在录制")
                    continue
                print("\033[33m--- 强制零位 5 秒（继续录制） ---\033[0m")
                time_w = now + ZERO_DURATION_W
                force_to_zero_flag = True

            elif user_input == 'q':
                s_pushed = False
                if record_data:
                    record_data = False
                    force_to_zero_flag = False
                    print("\033[33m--- Stopping episode ---\033[0m")
                else:
                    print("当前未在录制")

            elif user_input == 'e':
                record_data = False
                running = False
                s_pushed = False
                print("\033[32m--- Exiting... ---\033[0m")

        except EOFError:
            running = False
            break


if __name__ == '__main__':
    rclpy.init()
    sub_node = RobotSubscriber()
    recorder_node = Data_Recorder()
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(sub_node)
    executor.add_node(recorder_node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    keyboard_listener(sub_node)
    print("Shutting down...")
    time.sleep(1.0)
    recorder_node.generate_final_info()
    rclpy.shutdown()