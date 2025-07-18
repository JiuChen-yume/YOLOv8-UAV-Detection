from ultralytics import YOLO
import torch
import os

# 强制设置CUDA环境变量
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # 使GPU 0和1可见

# 检查CUDA是否真的可用
print(f"CUDA是否可用: {torch.cuda.is_available()}")
print(f"CUDA版本: {torch.version.cuda if torch.cuda.is_available() else '不可用'}")
print(f"GPU数量: {torch.cuda.device_count() if torch.cuda.is_available() else 0}")
if torch.cuda.is_available():
    print(f"当前GPU: {torch.cuda.get_device_name(0)}")

# 加载预训练模型
model = YOLO('yolov8n.pt')

# 强制使用GPU设备1
device = 0  # 使用第二个GPU (索引1)
print(f"强制使用设备: cuda:{device}")

# 使用选定的设备训练模型
model.train(
    data='data.yaml',  # 数据集配置文件路径
    epochs=100,        # 训练轮数
    imgsz=640,         # 图像大小
    batch=16,          # 批次大小
    name='yolov8n_custom',  # 实验名称
    device=device      # 强制使用GPU设备1
)