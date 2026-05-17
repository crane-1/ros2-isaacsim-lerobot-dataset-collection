#!/usr/bin/env python3
import json
import ast
import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import String
# import cv_bridge
import cv2
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer
import message_filters
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.configuration_act import ACTConfig

class ACTROSNode(Node):
    def __init__(self):
        super().__init__('act_ros_node')

        # ==================== 配置 ====================
        self.checkpoint_dir = Path("/home/qwe/isaacsim_piper/isaacsim_piper/isaacsim_piper/lerobot_related/outputs/train/my_act_dual_3_views/checkpoints/100000/pretrained_model")
        self.stats_path = "/home/qwe/.cache/huggingface/lerobot/xc/my_data_new_dual/meta/stats.json"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.image_size = (336, 336)          # 模型期望的图像分辨率
        self.state_dim = 8                    # 根据你的测试成功，action/state dim=8
        self.chunk_size = 50                  # 必须和训练时一致！
        self.min_chunk_left = 5
        self.control_hz = 10
        self.action_queue = []
        self.joint_names_l = [
            "joint1", "joint2", "joint3", "joint4",
            "joint5", "joint6", "joint7", "joint8"
        ]
        self.joint_names_r = [
            "joint1", "joint2", "joint3", "joint4",
            "joint5", "joint6", "joint7", "joint8"
        ]
        self.joint_states = [0.0] * 16
        self.alpha = 0.5  # 平滑系数
        # 订阅话题
        # ==================== 时间同步订阅 ====================
        self.create_subscription(Image, '/top_front_rgb', lambda msg: self.image_callback(msg, 'observation.images.top_front'), 10)
        self.create_subscription(Image, '/wrist_rgb_r',   lambda msg: self.image_callback(msg, 'observation.images.wrist_r'), 10)
        self.create_subscription(Image, '/top_back_rgb',  lambda msg: self.image_callback(msg, 'observation.images.top_back'), 10)
        self.create_subscription(Image, '/wrist_rgb_l',   lambda msg: self.image_callback(msg, 'observation.images.wrist_l'), 10)

        self.create_subscription(JointState, '/joint_states_single_l', self.joint_state_l_callback, 10)
        self.create_subscription(JointState, '/joint_states_single_r', self.joint_state_r_callback, 10)

        # 发布动作
        self.pub_joint_l = self.create_publisher(
            JointState, '/joint_states_sim_l', self.control_hz)
        self.pub_joint_r = self.create_publisher(
            JointState, '/joint_states_sim_r', self.control_hz)
        # 观察缓冲
        self.obs = {
            'observation.images.top_front': None,
            'observation.images.wrist_l': None,
            'observation.images.wrist_r':None,
            'observation.images.top_back': None,
            'observation.state':          None,
        }

        # self.action_buffer = np.zeros((self.chunk_size, 16), dtype=np.float32)  # 用于 ensembling
        # self.buffer_weights = np.exp(-np.arange(self.chunk_size) * 0.1)  # 指数衰减权重，可调
        # self.buffer_weights /= self.buffer_weights.sum()
        # self.current_step_in_chunk = 0
        
        # 加载模型和 tokenizer
        self.load_model()

        # 控制循环（频率建议 5~10Hz，根据推理速度调整）
        self.timer = self.create_timer(1.0 / self.control_hz, self.control_loop)  

        self.get_logger().info("ACT ROS node initialized. Waiting for data...")
        with open(self.stats_path, 'r', encoding='utf-8') as f:
            raw_content = f.read().strip()
        try:
            # 用 ast.literal_eval 解析 Python 字面量
            stats_dict = ast.literal_eval(raw_content)
            
            # 现在可以安全取值
            action_stats = stats_dict["action"]
            state_stats = stats_dict["observation.state"]  # 这个 key 确实存在
            
            self.action_mean = torch.tensor(action_stats["mean"], dtype=torch.float32, device=self.device)
            self.action_std  = torch.tensor(action_stats["std"], dtype=torch.float32, device=self.device)
            
            self.state_mean = torch.tensor(state_stats["mean"], dtype=torch.float32, device=self.device)
            self.state_std  = torch.tensor(state_stats["std"], dtype=torch.float32, device=self.device)
        except Exception as e:
            self.get_logger().error(f"加载 stats 失败: {e}")
        self.get_logger().info("Loaded action stats for manual unnormalize.")
        
    def image_callback(self, msg: Image, key: str):
        try:
            # self.get_logger().info(f"✅ Received image on {key}")
            self.process_image(msg, key)
        except Exception as e:
            self.get_logger().error(f"Image callback error {key}: {e}")

    def joint_state_l_callback(self, msg: JointState):
        if len(msg.position) == self.state_dim:
            self.joint_states[0:8] = np.array(msg.position, dtype=np.float32)
            # self.get_logger().info("✅ Received left joint states")
            # self.joint_norm()
        else:
            self.get_logger().warn(f"Left joint dim mismatch: {len(msg.position)}")

    def joint_state_r_callback(self, msg: JointState):
        if len(msg.position) == self.state_dim:
            self.joint_states[8:16] = np.array(msg.position, dtype=np.float32)
            # self.get_logger().info("✅ Received right joint states")
            # self.joint_norm()
        else:
            self.get_logger().warn(f"Right joint dim mismatch: {len(msg.position)}")

    def process_image(self, msg: Image, key: str):
        try:
            cv_img = self.ros_image_to_cv2(msg)
            cv_img = cv_img.copy()
            tensor = torch.from_numpy(cv_img).permute(2, 0, 1).float() / 255.0
            tensor = torch.nn.functional.interpolate(
                tensor.unsqueeze(0), size=self.image_size, mode='bilinear', align_corners=False
            ).squeeze(0).to(self.device)
            self.obs[key] = tensor
            # self.get_logger().debug(f"✅ Processed image {key} shape: {tensor.shape}")
        except Exception as e:
            self.get_logger().warn(f"Image process error {key}: {e}")

    def ros_image_to_cv2(self, msg: Image) -> np.ndarray:
            """
            手动转换 ROS sensor_msgs/Image 到 OpenCV numpy array
            支持常见 encoding：rgb8, bgr8, bgra8, mono8, 16UC1 等
            """
            height = msg.height
            width = msg.width
            encoding = msg.encoding.lower()  # 统一小写比较
            data = np.frombuffer(msg.data, dtype=np.uint8)

            # 计算每像素字节数
            if 'mono8' in encoding or '8uc1' in encoding:
                channels = 1
            elif 'rgb' in encoding or 'bgr' in encoding:
                channels = 3
            elif 'bgra' in encoding or 'rgba' in encoding:
                channels = 4
            elif '16uc1' in encoding or 'mono16' in encoding:
                channels = 1
                data = data.view(np.uint16)
            else:
                raise ValueError(f"不支持的 encoding: {msg.encoding}")

            bytes_per_pixel = channels * (data.dtype.itemsize)

            # 检查是否有 padding（step != width * bytes_per_pixel）
            if msg.step == width * bytes_per_pixel:
                # 无 padding，直接 reshape
                img = data.reshape((height, width, channels) if channels > 1 else (height, width))
            else:
                # 有 padding，逐行拷贝
                img = np.zeros((height, width, channels) if channels > 1 else (height, width),
                            dtype=data.dtype)
                for i in range(height):
                    start = i * msg.step
                    end = start + width * bytes_per_pixel
                    row_data = data[start:end]
                    img[i] = row_data.reshape((width, channels) if channels > 1 else (width,))

            # 根据 encoding 做通道调整
            if encoding in ['rgb8', 'rgb']:
                # img = img[:, :, ::-1]
                pass
            elif encoding in ['bgr8', 'bgr']:
                pass  
            elif encoding in ['bgra8', 'bgra']:
                img = img[:, :, :3][:, :, ::-1]  # BGRA → RGB，去 alpha
            elif encoding in ['mono8', '8uc1', 'mono16', '16uc1']:
                # 灰度图，转成 3 通道方便显示
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                self.get_logger().warn(f"未知 encoding，已按默认处理: {encoding}")

            return img

    def load_model(self):
        self.get_logger().info(f"Loading ACT from: {self.checkpoint_dir}")
        try:
                self.policy = ACTPolicy.from_pretrained(
                    str(self.checkpoint_dir),
                    device_map=None,  # 或 "cuda"
                )
                if self.device == "cuda":
                    self.policy = self.policy.cuda()
                self.policy.eval()
                self.get_logger().info(f"ACT policy loaded from {self.checkpoint_dir}")
        except Exception as e:
                self.get_logger().error(f"Failed to load ACT: {e}")
                raise
        
    def joint_norm(self):
        # 手动归一化 state
        processed_position = (self.joint_states - self.state_mean.cpu().numpy()) / self.state_std.cpu().numpy()
        state = torch.from_numpy(processed_position).unsqueeze(0).to(self.device)
        self.obs['observation.state'] = state

    def control_loop(self):
        self.joint_norm()

        # 检查是否所有必要输入就绪
        required_keys = ['observation.images.top_front', 'observation.images.wrist_l',
                         'observation.images.top_back', 'observation.images.wrist_r',
                         'observation.state']
        if any(self.obs.get(k) is None for k in required_keys):
            return  # 等待数据

        # 准备 batch
        obs_batched = {}
        for k, v in self.obs.items():
            if v is not None:
                if v.dim() == 3:  # 图像 [C,H,W] → [1,C,H,W]
                    obs_batched[k] = v.unsqueeze(0)
                else:
                    obs_batched[k] = v
        if len(self.action_queue) < self.min_chunk_left:
            try:
                with torch.no_grad():
                    action = self.policy.select_action(obs_batched)
                if action.dim() == 3:           # [1, horizon, dim]
                    chunk = action[0]
                elif action.dim() == 2:         # [1, dim] → 视为 horizon=1
                    chunk = action[0:1]         # 保持 2维 [1, dim]
                else:
                    raise ValueError(f"Unexpected action dim: {action.dim()}")

                chunk_np = chunk.cpu().numpy()
                # overlap = 0
                # if len(self.action_queue) > 0:
                #     overlap = min(len(self.action_queue), len(chunk_np))
                #     for i in range(overlap):
                #         self.action_queue[i] = (self.action_queue[i] * self.alpha + chunk_np[i] * (1 - self.alpha))
                # self.action_queue.extend(chunk_np.tolist()[overlap:])
                self.action_queue.extend(chunk_np.tolist())
            except Exception as e:
                self.get_logger().error(f"Inference failed: {e}")
        if self.action_queue:
            act = np.array(self.action_queue.pop(0), dtype=np.float32)

            msg_l = JointState()
            msg_r = JointState()
            msg_l.header.stamp = self.get_clock().now().to_msg()
            msg_l.name = self.joint_names_l
            msg_r.header.stamp = self.get_clock().now().to_msg()
            msg_r.name = self.joint_names_r
            position = act  # numpy 数组
            position = position * self.action_std.cpu().numpy() + self.action_mean.cpu().numpy()  # 手动反归一化
            msg_l.position = position.tolist()[0:8]
            msg_r.position = position.tolist()[8:16]
            self.pub_joint_l.publish(msg_l)
            self.pub_joint_r.publish(msg_r)
        else:
            self.get_logger().warn("No action in queue, waiting...")
            

def main(args=None):
    rclpy.init(args=args)
    node = ACTROSNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()