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

# ====================== 全局变量（扩展为双臂） ======================
record_data = False
running = True
vid_H = 336
vid_W = 336

# 图像缓冲（保持 3 个相机）
wrist_camera_image_l = np.zeros((vid_H, vid_W, 3), np.uint8)
wrist_camera_image_r = np.zeros((vid_H, vid_W, 3), np.uint8)
top_front_camera_image = np.zeros((vid_H, vid_W, 3), np.uint8)
top_back_camera_image = np.zeros((vid_H, vid_W, 3), np.uint8)

# 主臂（当前臂）状态与发布
current_joint_states_l = [0.0] * 8          # 主臂 8 维 (6关节 + 2夹爪)
msg_pub_to_sim_l = [0.0] * 8                # 主臂发布给仿真/控制器

# 第二臂（后缀 _r）
current_joint_states_r = [0.0] * 8
msg_pub_to_sim_r = [0.0] * 8

# 零位控制（共用）
time_s = 0.0
time_w = 0.0
ZERO_DURATION_S = 3.0
ZERO_DURATION_W = 5.0
force_to_zero_flag = False
s_pushed = False

class RobotSubscriber(Node):
    def __init__(self):
        super().__init__('robot_subscriber')

        # ==================== 主臂订阅（保持不变） ====================
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states_single_l', self.joint_callback_l, 10)
        self.joint_pika_sub = self.create_subscription(
            JointState, '/joint_states_l', self.joint_pika_callback_l, 10)
        self.gripper_sub = self.create_subscription(
            JointState, '/gripper_l/joint_state', self.gripper_callback_l, 10)
        # ==================== 第二臂订阅（新增 _1） ====================
        self.joint_sub_1 = self.create_subscription(
            JointState, '/joint_states_single_r', self.joint_callback_r, 10)
        self.joint_pika_sub_1 = self.create_subscription(
            JointState, '/joint_states_r', self.joint_pika_callback_r, 10)
        self.gripper_sub_1 = self.create_subscription(
            JointState, '/gripper_r/joint_state', self.gripper_callback_r, 10)
        # 相机订阅（保持不变）
        self.wrist_sub_l = self.create_subscription(Image, '/wrist_rgb_l', self.wrist_callback_l, 10)
        self.wrist_sub_r = self.create_subscription(Image, '/wrist_rgb_r', self.wrist_callback_r, 10)
        self.top_front_sub = self.create_subscription(Image, '/top_front_rgb', self.top_front_callback, 10)
        self.top_back_sub = self.create_subscription(Image, '/top_back_rgb', self.top_back_callback, 10)

        # 发布融合后的关节状态给仿真/控制器（目前只发布主臂，可按需扩展）
        self.pub_sim_l = self.create_publisher(JointState, '/joint_states_sim_l', 10)
        self.pub_sim_r = self.create_publisher(JointState, '/joint_states_sim_r', 10)
        self.create_timer(0.1, self.publish_sim_callback_l)
        self.create_timer(0.1, self.publish_sim_callback_r)

    # ====================== 主臂回调（基本不变） ======================
    def joint_callback_l(self, msg):
        global current_joint_states_l
        current_joint_states_l = list(msg.position)
        self.check_start_recording()

    def joint_pika_callback_l(self, msg: JointState):
        global msg_pub_to_sim_l
        if len(msg.position) < 6:
            self.get_logger().warn("arm joints < 6, ignoring")
            return
        if force_to_zero_flag:
            msg_pub_to_sim_l[:6] = [0.0] * 6
        else:
            msg_pub_to_sim_l[:6] = list(msg.position[:6])
        self.check_start_recording()

    def gripper_callback_l(self, msg):
        global msg_pub_to_sim_l
        if not msg.position:
            return
        if force_to_zero_flag:
            msg_pub_to_sim_l[6] = 0.0
            msg_pub_to_sim_l[7] = 0.0
        else:
            half = msg.position[0] / 2.0
            msg_pub_to_sim_l[6] = half
            msg_pub_to_sim_l[7] = -half
        self.check_start_recording()

    # ====================== 第二臂回调（新增） ======================
    def joint_callback_r(self, msg):
        global current_joint_states_r
        current_joint_states_r = list(msg.position)
        self.check_start_recording()

    def joint_pika_callback_r(self, msg: JointState):
        global msg_pub_to_sim_r
        if len(msg.position) < 6:
            self.get_logger().warn("arm_r joints < 6, ignoring")
            return
        if force_to_zero_flag:
            msg_pub_to_sim_r[:6] = [0.0] * 6
        else:
            msg_pub_to_sim_r[:6] = list(msg.position[:6])
        self.check_start_recording()

    def gripper_callback_r(self, msg):
        global msg_pub_to_sim_r
        if not msg.position:
            return
        if force_to_zero_flag:
            msg_pub_to_sim_r[6] = 0.0
            msg_pub_to_sim_r[7] = 0.0
        else:
            half = msg.position[0] / 2.0
            msg_pub_to_sim_r[6] = half
            msg_pub_to_sim_r[7] = -half
        self.check_start_recording()

    # ====================== 相机回调（不变） ======================
    def wrist_callback_l(self, msg):
        global wrist_camera_image_l
        try:
            wrist_camera_image_l = cv2.resize(bridge.imgmsg_to_cv2(msg, "bgr8"), (vid_W, vid_H))
        except: pass

    def wrist_callback_r(self, msg):
        global wrist_camera_image_r
        try:
            wrist_camera_image_r = cv2.resize(bridge.imgmsg_to_cv2(msg, "bgr8"), (vid_W, vid_H))
        except: pass

    def top_front_callback(self, msg):
        global top_front_camera_image
        try:
            top_front_camera_image = cv2.resize(bridge.imgmsg_to_cv2(msg, "bgr8"), (vid_W, vid_H))
        except: pass

    def top_back_callback(self, msg):
        global top_back_camera_image
        try:
            top_back_camera_image = cv2.resize(bridge.imgmsg_to_cv2(msg, "bgr8"), (vid_W, vid_H))
        except: pass

    def check_start_recording(self):
        global time_s, time_w, record_data, force_to_zero_flag
        if time.time() < time_s or time.time() < time_w:
            return
        if not record_data and s_pushed:
            print("\033[32m--- Starting recording ---\033[0m")
            record_data = True
            force_to_zero_flag = False

    def publish_sim_callback_l(self):
        """目前只发布主臂，可根据需要扩展为发布双臂"""
        global force_to_zero_flag, msg_pub_to_sim_l
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7', 'joint8']
        msg.velocity = [0.0] * 8
        msg.effort = [0.0] * 8
        msg.position = [0.0] * 8 if force_to_zero_flag else msg_pub_to_sim_l
        self.pub_sim_l.publish(msg)

    def publish_sim_callback_r(self):
        """发布第二臂状态"""
        global force_to_zero_flag, msg_pub_to_sim_r
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7', 'joint8']
        msg.velocity = [0.0] * 8
        msg.effort = [0.0] * 8
        msg.position = [0.0] * 8 if force_to_zero_flag else msg_pub_to_sim_r
        self.pub_sim_r.publish(msg)

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
        self.wrist_vid_dir_l = os.path.join(self.base_dir, "videos/chunk_000/observation.images.wrist_l/")
        self.wrist_vid_dir_r = os.path.join(self.base_dir, "videos/chunk_000/observation.images.wrist_r/")
        self.top_front_vid_dir = os.path.join(self.base_dir, "videos/chunk_000/observation.images.top_front/")
        self.top_back_vid_dir = os.path.join(self.base_dir, "videos/chunk_000/observation.images.top_back/")
        
        for d in [self.data_dir, self.wrist_vid_dir_l, self.wrist_vid_dir_r, self.top_front_vid_dir, self.top_back_vid_dir, self.meta_dir]:
            os.makedirs(d, exist_ok=True)
        self.reset_buffers()
        self.save_task_jsonl("pass cube")

    def reset_buffers(self):
        self.df_list = []
        self.frame_index = 0
        self.time_stamp = 0.0
        self.wrist_frames_l = []
        self.wrist_frames_r = []
        self.top_front_frames = []
        self.top_back_frames = []

    def timer_callback(self):
        global record_data, current_joint_states_l, current_joint_states_r
        global msg_pub_to_sim_l, msg_pub_to_sim_r

        if record_data:
            if not self.recording_active:
                print(f'\033[32m--- Start Recording Episode {self.episode_index} ---\033[0m')
                self.reset_buffers()
                self.recording_active = True

            # ====================== 双臂拼接 ======================
            state = np.concatenate([
                np.array(current_joint_states_l, dtype=np.float32),      # 主臂 8 维
                np.array(current_joint_states_r, dtype=np.float32)     # 第二臂 8 维
            ])

            action = np.concatenate([
                np.array(msg_pub_to_sim_l, dtype=np.float32),
                np.array(msg_pub_to_sim_r, dtype=np.float32)
            ])

            self.df_list.append({
                'observation.state': state,
                'action': action,
                'timestamp': np.float32(self.time_stamp),
                'frame_index': self.frame_index,
                'episode_index': self.episode_index,
                'index': self.frame_index,
                'task_index': 0
            })

            self.wrist_frames_l.append(copy.copy(wrist_camera_image_l))
            self.wrist_frames_r.append(copy.copy(wrist_camera_image_r))
            self.top_front_frames.append(copy.copy(top_front_camera_image))
            self.top_back_frames.append(copy.copy(top_back_camera_image))
            self.frame_index += 1
            self.time_stamp += 1/self.Hz
            print(f"Recording frame: {self.frame_index}", end='\r')
        else:
            if self.recording_active:
                self.save_episode()
                self.recording_active = False

    # ====================== save_episode / save_task_jsonl / generate_final_info ======================
    # 以下部分基本不变，仅调整 state_dim 为 16，并更新 names

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
            for cam, frames, dirname in [
                ("wrist_l", self.wrist_frames_l, self.wrist_vid_dir_l),
                ("wrist_r", self.wrist_frames_r, self.wrist_vid_dir_r),
                ("top_front", self.top_front_frames, self.top_front_vid_dir),
                ("top_back", self.top_back_frames, self.top_back_vid_dir)
                
            ]:
                out = cv2.VideoWriter(os.path.join(dirname, f"{prefix}.mp4"), fourcc, self.Hz, (vid_W, vid_H))
                for f in frames:
                    out.write(f)
                out.release()

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
        state_dim = 16  # 双臂：8 + 8

        info = {
            "codebase_version": "v2.1",
            "robot_type": "piper_dual",
            "total_episodes": self.episode_index,
            "total_frames": self.total_frames_count,
            "total_tasks": 1,
            "total_videos": self.episode_index * 4,  # 4 个相机
            "total_chunks": 1,
            "chunks_size": 1000,
            "fps": self.Hz,
            "splits": {"train": f"0:{self.episode_index}"},
            "data_path": "data/chunk_000/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk_000/observation.images.{camera_name}/episode_{episode_index:06d}.mp4",
            "features": {
                "observation.state": {
                    "dtype": "float32",
                    "shape": [state_dim],
                    "names": [f"joint_{i+1}" for i in range(16)]   # 可自定义更清晰的名字
                },
                "action": {
                    "dtype": "float32",
                    "shape": [state_dim],
                    "names": [f"joint_{i+1}" for i in range(16)]
                },
                "observation.images.top_front": self._vid_info(),
                "observation.images.wrist_l": self._vid_info(),
                "observation.images.wrist_r": self._vid_info(),
                "observation.images.top_back": self._vid_info(),
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


# ====================== 键盘监听（不变） ======================
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