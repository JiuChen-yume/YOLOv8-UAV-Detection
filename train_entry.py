# train_entry.py
import torch
torch.use_deterministic_algorithms(False)
from ultralytics import YOLO


MODEL_WT = 'yolov8n.pt'  # 可换成你自己的模型 yaml 或预训练权重
DATA_YAML = 'data.yaml'

print(f'Using model: {MODEL_WT}')


if __name__ == '__main__':
    model=YOLO(MODEL_WT)
    if hasattr(model, 'model') and hasattr(model.model, 'yaml'):
        yaml = model.model.yaml
        print(f"\n当前模型类型参数:")
        print(f"模型结构文件: {yaml.get('model', '未指定')}")
        print(f"depth_multiple: {yaml.get('depth_multiple', 'N/A')}")
        print(f"width_multiple: {yaml.get('width_multiple', 'N/A')}")
        print(f"类别数量 (nc): {yaml.get('nc', 'N/A')}")
    else:
        print("无法读取模型结构信息，请确认模型是否正确加载。")

    print('开始训练 ...')
    model.train(
        data=DATA_YAML,
        epochs=100,
        imgsz=640,
        batch=12,
        device='cuda',  # 改为 'cuda' 更快
        optimizer='SGD',
        momentum=0.937,
        lr0=0.01,
        cos_lr=True,
        close_mosaic=10,
        warmup_epochs=3.0,
        copy_paste=0.3,
        mosaic=1.0,
        mixup=0.2,

        project='runs_custom',
        name='train_entry'
    )
