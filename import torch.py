import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientdet import EfficientDetBackbone
import numpy as np
import cv2
import os
import math
from utils.cython_nms import nms

class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.depth = d_model // num_heads
        
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        
        self.dense = nn.Linear(d_model, d_model)
        
    def split_heads(self, x, batch_size):
        """将最后一个维度分成 (num_heads, depth)"""
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, q, k, v, mask=None):
        batch_size = q.shape[0]
        
        q = self.wq(q)  # (batch_size, seq_len_q, d_model)
        k = self.wk(k)  # (batch_size, seq_len_k, d_model)
        v = self.wv(v)  # (batch_size, seq_len_v, d_model)
        
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
        
        # 缩放点积注意力
        scaled_attention, attention_weights = self.scaled_dot_product_attention(
            q, k, v, mask)
        
        # (batch_size, seq_len_q, num_heads, depth)
        scaled_attention = scaled_attention.permute(0, 2, 1, 3)
        
        # (batch_size, seq_len_q, d_model)
        concat_attention = scaled_attention.reshape(batch_size, -1, self.d_model)
        
        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
        
        return output, attention_weights
    
    def scaled_dot_product_attention(self, q, k, v, mask=None):
        """计算注意力权重"""
        matmul_qk = torch.matmul(q, k.transpose(-2, -1))  # (..., seq_len_q, seq_len_k)
        
        # 缩放 matmul_qk
        dk = torch.tensor(k.shape[-1], dtype=torch.float32)
        scaled_attention_logits = matmul_qk / torch.sqrt(dk)
        
        # 添加 mask
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        
        # softmax 归一化权重
        attention_weights = F.softmax(scaled_attention_logits, dim=-1)
        
        output = torch.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)
        
        return output, attention_weights

class TransformerEncoder(nn.Module):
    """简化版Transformer编码器"""
    def __init__(self, d_model, num_heads, num_layers):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads)
            for _ in range(num_layers)
        ])
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class SpatialTransformer(nn.Module):
    """空间Transformer模块"""
    def __init__(self, in_channels, num_heads=8, num_layers=2):
        super(SpatialTransformer, self).__init__()
        self.in_channels = in_channels
        
        # 投影层
        self.proj_in = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        # Transformer编码器
        self.transformer = TransformerEncoder(
            d_model=in_channels,
            num_heads=num_heads,
            num_layers=num_layers
        )
        
        # 输出投影
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
    def forward(self, x):
        # x: [B, C, H, W]
        b, c, h, w = x.shape
        
        # 投影
        x_proj = self.proj_in(x)
        
        # 重塑为序列
        x_flat = x_proj.flatten(2).permute(2, 0, 1)  # [H*W, B, C]
        
        # 应用Transformer
        x_transformed = self.transformer(x_flat)
        
        # 重塑回特征图
        x_restored = x_transformed.permute(1, 2, 0).view(b, c, h, w)
        
        # 输出投影
        x_out = self.proj_out(x_restored)
        
        # 残差连接
        return x + x_out

# Fast R-CNN 风格的 RoI 池化层
class RoIPool(nn.Module):
    def __init__(self, output_size, spatial_scale):
        super(RoIPool, self).__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.roi_pool = nn.AdaptiveMaxPool2d(output_size)
        
    def forward(self, features, rois):
        """
        Args:
            features: 特征图 [B, C, H, W]
            rois: 感兴趣区域 [N, 5] (batch_idx, x1, y1, x2, y2)
        """
        roi_features = []
        for roi in rois:
            batch_idx = int(roi[0])
            x1, y1, x2, y2 = roi[1:].mul(self.spatial_scale).round().int()
            
            # 确保坐标有效
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(features.size(3) - 1, x2)
            y2 = min(features.size(2) - 1, y2)
            
            # 提取RoI
            roi_feature = features[batch_idx, :, y1:y2+1, x1:x2+1]
            
            # 池化到固定大小
            roi_feature = self.roi_pool(roi_feature).unsqueeze(0)
            roi_features.append(roi_feature)
            
        return torch.cat(roi_features, 0)

# Fast R-CNN 头部网络
class FastRCNNHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(FastRCNNHead, self).__init__()
        self.fc1 = nn.Linear(in_channels * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        
        # 分类器
        self.cls_score = nn.Linear(1024, num_classes)
        # 边界框回归器
        self.bbox_pred = nn.Linear(1024, num_classes * 4)
        
    def forward(self, x):
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        cls_score = self.cls_score(x)
        bbox_pred = self.bbox_pred(x)
        
        return cls_score, bbox_pred

# 区域提议网络 (RPN)
class RegionProposalNetwork(nn.Module):
    def __init__(self, in_channels, mid_channels=256, anchor_scales=[8, 16, 32]):
        super(RegionProposalNetwork, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        
        # 分类层 (前景/背景)
        self.cls_layer = nn.Conv2d(mid_channels, len(anchor_scales) * 2, kernel_size=1)
        
        # 边界框回归层
        self.bbox_layer = nn.Conv2d(mid_channels, len(anchor_scales) * 4, kernel_size=1)
        
        # 初始化权重
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self, x):
        # 共享特征
        h = F.relu(self.conv1(x))
        
        # 分类分数
        rpn_cls_scores = self.cls_layer(h)
        
        # 边界框回归
        rpn_bbox_preds = self.bbox_layer(h)
        
        return rpn_cls_scores, rpn_bbox_preds

# EfficientDet + Transformer + Fast R-CNN 融合模型
class EfficientDetTransformerFastRCNN(nn.Module):
    def __init__(self, num_classes=80, phi=0, pretrained=False, transformer_layers=2, transformer_heads=8):
        super(EfficientDetTransformerFastRCNN, self).__init__()
        
        # EfficientDet 骨干网络
        self.efficientdet = EfficientDetBackbone(num_classes=num_classes, phi=phi, pretrained=pretrained)
        
        # 获取特征维度
        self.fpn_num_filters = [64, 88, 112, 160, 224, 288, 384, 384]
        self.feature_dim = self.fpn_num_filters[phi]
        
        # 空间Transformer增强
        self.transformers = nn.ModuleList([
            SpatialTransformer(
                in_channels=self.feature_dim,
                num_heads=transformer_heads,
                num_layers=transformer_layers
            )
            for _ in range(5)  # 5个特征层 (P3-P7)
        ])
        
        # 区域提议网络 (RPN)
        self.rpn = RegionProposalNetwork(
            in_channels=self.feature_dim,
            mid_channels=256,
            anchor_scales=[8, 16, 32]
        )
        
        # RoI池化
        self.roi_pool = RoIPool(output_size=7, spatial_scale=1/16)
        
        # Fast R-CNN 头部
        self.fast_rcnn_head = FastRCNNHead(
            in_channels=self.feature_dim,
            num_classes=num_classes + 1  # +1 for background
        )
        
    def forward(self, images, proposals=None):
        # 从EfficientDet获取特征
        _, p3, p4, p5 = self.efficientdet.backbone_net(images)
        features = (p3, p4, p5)
        features = self.efficientdet.bifpn(features)
        
        # 应用Transformer增强特征
        enhanced_features = []
        for i, feature in enumerate(features):
            enhanced = self.transformers[i](feature)
            enhanced_features.append(enhanced)
        
        # 使用P4特征进行RPN (可以根据需要调整)
        rpn_feature = enhanced_features[1]  # P4
        rpn_cls_scores, rpn_bbox_preds = self.rpn(rpn_feature)
        
        # 如果没有提供proposals，则使用RPN生成
        if proposals is None:
            # 这里应该有代码将RPN输出转换为proposals
            # 简化起见，我们假设proposals已经生成
            batch_size = images.size(0)
            proposals = torch.rand(100, 5)  # 假设每张图像100个proposals
            proposals[:, 0] = torch.randint(0, batch_size, (100,))  # batch索引
        
        # RoI池化
        roi_features = self.roi_pool(enhanced_features[1], proposals)
        
        # Fast R-CNN头部
        cls_scores, bbox_preds = self.fast_rcnn_head(roi_features)
        
        if self.training:
            # 训练模式返回所有中间结果
            return enhanced_features, rpn_cls_scores, rpn_bbox_preds, cls_scores, bbox_preds
        else:
            # 推理模式只返回最终结果
            return cls_scores, bbox_preds, proposals

# 推理函数
def inference(model, image_path, device, conf_thresh=0.5, nms_thresh=0.3):
    # 加载图像
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 预处理
    input_image = cv2.resize(image, (512, 512))
    input_tensor = torch.from_numpy(input_image).permute(2, 0, 1).float() / 255.0
    input_tensor = input_tensor.unsqueeze(0).to(device)
    
    # 推理
    model.eval()
    with torch.no_grad():
        cls_scores, bbox_preds, proposals = model(input_tensor)
    
    # 获取类别和分数
    scores = F.softmax(cls_scores, dim=1)
    max_scores, pred_classes = scores.max(dim=1)
    
    # 过滤低置信度预测
    keep = max_scores > conf_thresh
    scores = max_scores[keep]
    pred_classes = pred_classes[keep]
    bbox_preds = bbox_preds[keep]
    proposals = proposals[keep]
    
    # 应用边界框回归
    # 这里应该有代码将bbox_preds应用到proposals上
    # 简化起见，我们直接使用proposals
    boxes = proposals[:, 1:5]
    
    # 应用NMS
    keep = nms(torch.cat([boxes, scores.unsqueeze(1)], dim=1).cpu().numpy(), nms_thresh)
    
    boxes = boxes[keep]
    scores = scores[keep]
    pred_classes = pred_classes[keep]
    
    # 转换回原始图像尺寸
    orig_h, orig_w = image.shape[:2]
    scale_x = orig_w / 512
    scale_y = orig_h / 512
    
    boxes[:, 0] *= scale_x
    boxes[:, 1] *= scale_y
    boxes[:, 2] *= scale_x
    boxes[:, 3] *= scale_y
    
    return boxes, scores, pred_classes

# 可视化函数
def visualize_detections(image, boxes, scores, class_ids, class_names, save_path=None):
    # 复制图像以避免修改原始图像
    image_copy = image.copy()
    
    # 绘制每个检测框
    for box, score, class_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = box.int().cpu().numpy()
        class_name = class_names[class_id.item()]
        
        # 绘制框
        cv2.rectangle(image_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 绘制标签
        label = f"{class_name}: {score:.2f}"
        cv2.putText(image_copy, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # 保存或显示图像
    if save_path:
        cv2.imwrite(save_path, cv2.cvtColor(image_copy, cv2.COLOR_RGB2BGR))
    
    return image_copy

# 示例用法
if __name__ == "__main__":
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建模型
    model = EfficientDetTransformerFastRCNN(
        num_classes=80,  # COCO 数据集有 80 个类别
        phi=0,  # 使用 EfficientDet-D0
        pretrained=True,
        transformer_layers=2,
        transformer_heads=8
    ).to(device)
    
    # 示例: 推理单张图像
    image_path = "c:\\Users\\lmq\\Desktop\\test_image.jpg"
    boxes, scores, pred_classes = inference(model, image_path, device)
    
    # 可视化结果
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # COCO 类别名称 (简化)
    class_names = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train"]
    
    result_image = visualize_detections(
        image, boxes, scores, pred_classes, class_names, 
        save_path="c:\\Users\\lmq\\Desktop\\result.jpg"
    )
    
    print("推理完成，结果已保存到 result.jpg")