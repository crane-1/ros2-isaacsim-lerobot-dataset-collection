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

# SmolVLA 导入（根据你的环境路径）
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

joint_lower = np.array([-150, 0, -150, -104, -70, -180, 0, -0.05])  # 示例下限
joint_upper = np.array([150, 180, 0, 104, 70, 180, 0.05, 0])

class SmolVLAROSNode(Node):
    def __init__(self):
        super().__init__('smolvla_ros_node')

        # ==================== 配置 ====================
        self.checkpoint_dir = Path("/home/qwe/isaacsim_piper/isaacsim_piper/isaacsim_piper/lerobot_related/outputs/train/my_smolvla_dual/checkpoints/040000/pretrained_model")
        self.stats_path = "/home/qwe/.cache/huggingface/lerobot/xc/my_data_new_dual/meta/stats.json"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.image_size = (336, 336)          # 模型期望的图像分辨率
        self.state_dim = 8                    # 根据你的测试成功，action/state dim=8
        self.max_lang_length = 32             # tokenizer max_length
        self.current_state = [0.0] * 2 * self.state_dim  # 初始化状态
        self.joint_names = [
            "joint1", "joint2", "joint3", "joint4",
            "joint5", "joint6", "joint7", "joint8"  # 根据你的机器人修改，8个自由度示例
        ]

        # 默认指令（如果没有收到语言话题）
        self.current_instruction = "pass cylinder to box"  # 或 ""，根据训练时的指令风格调整

        # 订阅话题
        self.sub_top    = self.create_subscription(
            Image, '/top_front_rgb', self.top_front_image_callback, 5)
        self.sub_wrist  = self.create_subscription(
            Image, '/wrist_rgb_l', self.wrist_l_image_callback, 5)
        self.sub_side   = self.create_subscription(
            Image, '/wrist_rgb_r', self.wrist_r_image_callback, 5)  
        self.sub_joint  = self.create_subscription(
            JointState, '/joint_states_single_l', self.joint_state_l_callback, 10)
        self.sub_joint  = self.create_subscription(
            JointState, '/joint_states_single_r', self.joint_state_r_callback, 10)
        # 发布动作
        self.pub_joint_l = self.create_publisher(
            JointState, '/joint_states_sim_l', 10)
        self.pub_joint_r = self.create_publisher(
            JointState, '/joint_states_sim_r', 10)
        # 观察缓冲
        self.obs = {
            'observation.images.camera1': None,
            'observation.images.camera2': None,
            'observation.images.camera3': None,
            'observation.state':          None,
        }

        # 加载模型和 tokenizer
        self.load_model_and_tokenizer()

        # 控制循环（频率建议 5~10Hz，根据推理速度调整）
        self.timer = self.create_timer(0.2, self.control_loop)  

        self.get_logger().info("SmolVLA ROS node initialized. Waiting for data...")
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

    def load_model_and_tokenizer(self):
        self.get_logger().info(f"Loading SmolVLA from: {self.checkpoint_dir}")
        try:
            self.policy = SmolVLAPolicy.from_pretrained(
                str(self.checkpoint_dir),
                dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                device_map=None,
                low_cpu_mem_usage=True,
            )
            if self.device == "cuda":
                self.policy = self.policy.cuda()
                torch.cuda.synchronize()
            self.policy.eval()
            self.get_logger().info("SmolVLA model loaded successfully on " + self.device)

            # tokenizer
            vlm_name = getattr(self.policy.config, 'vlm_model_name', "HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
            self.tokenizer = AutoTokenizer.from_pretrained(vlm_name)
            self.get_logger().info(f"Tokenizer loaded from: {vlm_name}")

        except Exception as e:
            self.get_logger().error(f"Model loading failed: {str(e)}")
            raise

    def top_front_image_callback(self, msg: Image):
        try:
            cv_img = self.ros_image_to_cv2(msg)
            # cv_img = cv_bridge.CvBridge().imgmsg_to_cv2(msg, desired_encoding='bgr8')  # 使用 cv_bridge 处理图像
            cv_img = cv_img.copy()  # 确保数据连续，避免后续处理问题
            tensor = torch.from_numpy(cv_img).permute(2, 0, 1).float() / 255.0
            tensor = torch.nn.functional.interpolate(
                tensor.unsqueeze(0), size=self.image_size, mode='bilinear', align_corners=False
            ).squeeze(0).to(self.device)
            self.obs['observation.images.camera1'] = tensor  # top → camera1
        except Exception as e:
            self.get_logger().warn(f"Top image processing error: {e}")

    def wrist_l_image_callback(self, msg: Image):
        try:
            cv_img = self.ros_image_to_cv2(msg)
            # cv_img = cv_bridge.CvBridge().imgmsg_to_cv2(msg, desired_encoding='bgr8')  # 使用 cv_bridge 处理图像
            cv_img = cv_img.copy()
            tensor = torch.from_numpy(cv_img).permute(2, 0, 1).float() / 255.0
            tensor = torch.nn.functional.interpolate(
                tensor.unsqueeze(0), size=self.image_size, mode='bilinear', align_corners=False
            ).squeeze(0).to(self.device)
            self.obs['observation.images.camera2'] = tensor  # wrist → camera2
        except Exception as e:
            self.get_logger().warn(f"Wrist image processing error: {e}")

    def wrist_r_image_callback(self, msg: Image):
        try:
            cv_img = self.ros_image_to_cv2(msg)
            # cv_img = cv_bridge.CvBridge().imgmsg_to_cv2(msg, desired_encoding='bgr8')  # 使用 cv_bridge 处理图像
            cv_img = cv_img.copy()
            tensor = torch.from_numpy(cv_img).permute(2, 0, 1).float() / 255.0
            tensor = torch.nn.functional.interpolate(
                tensor.unsqueeze(0), size=self.image_size, mode='bilinear', align_corners=False
            ).squeeze(0).to(self.device)
            self.obs['observation.images.camera3'] = tensor  # side → camera3
        except Exception as e:
            self.get_logger().warn(f"Side image processing error: {e}")

    def joint_state_l_callback(self, msg: JointState):
        if len(msg.position) != self.state_dim:
            self.get_logger().warn(f"Joint state dimension mismatch: got {len(msg.position)}, expect {self.state_dim}")
            return
        state = msg.position
        self.current_state[0:8] = state

    def joint_state_r_callback(self, msg: JointState):
        if len(msg.position) != self.state_dim:
            self.get_logger().warn(f"Joint state dimension mismatch: got {len(msg.position)}, expect {self.state_dim}")
            return
        state = msg.position
        self.current_state[8:16] = state

    def state_norm(self):
        processed_position = np.array(self.current_state, dtype=np.float32)
        processed_position = (processed_position - self.state_mean.cpu().numpy()) / self.state_std.cpu().numpy()
        state = torch.from_numpy(processed_position).unsqueeze(0).to(self.device)
        self.obs['observation.state'] = state

    def control_loop(self):
        # 检查是否所有必要输入就绪
        self.state_norm()  # 每次控制循环都更新状态输入
        required_keys = ['observation.images.camera1', 'observation.images.camera2',
                         'observation.images.camera3', 'observation.state']
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

        # 添加语言输入（每次都重新 tokenize，因为指令可能变化）
        try:
            tokens_dict = self.tokenizer(
                self.current_instruction,
                return_tensors="pt",
                padding="max_length",
                max_length=self.max_lang_length,
                truncation=True,
            )
            obs_batched['observation.language.tokens'] = tokens_dict["input_ids"].to(self.device)
            obs_batched['observation.language.attention_mask'] = tokens_dict["attention_mask"].to(self.device).bool()

            # 推理
            with torch.no_grad():
                action_dict = self.policy.select_action(obs_batched)
                action = action_dict['action'] if isinstance(action_dict, dict) else action_dict

            # 处理 action chunk（取第一步）
            if action.dim() == 3:  # [1, horizon, dim]
                action = action[:, 0, :]
            action_np = action.squeeze(0).cpu().numpy()

            # 手动反归一化 action
            action_np = action_np * self.action_std.cpu().numpy() + self.action_mean.cpu().numpy()
            position_ = action_np.tolist()
            joint_msg_l = JointState()
            joint_msg_l.header.stamp = self.get_clock().now().to_msg()
            joint_msg_l.name = self.joint_names
            joint_msg_l.position = position_[0:8]  # 左臂动作
            self.pub_joint_l.publish(joint_msg_l)
            joint_msg_r = JointState()
            joint_msg_r.header.stamp = self.get_clock().now().to_msg()
            joint_msg_r.name = self.joint_names
            joint_msg_r.position = position_[8:16]  # 右臂动作
            self.pub_joint_r.publish(joint_msg_r)
            self.get_logger().debug(f"Published action: {action_np.round(4)}")

        except Exception as e:
            self.get_logger().error(f"Inference failed: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    node = SmolVLAROSNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()