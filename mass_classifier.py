import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.abspath(".")) 
from models.iris import Iris

class MASSClassifier(nn.Module):
    def __init__(self, num_classes=2, freeze_encoder=True):
        super().__init__()
        
        # 1. 實例化 Iris (MASS) 模型
        self.encoder_model = Iris(
            in_ch=1,
            channels=[32, 64, 128, 256, 512],
            scale=[2, 2, 2, 2],
            kernel_size=[3, 3, 3, 3, 3],
            block='BasicBlock',
            num_block=[2, 2, 2, 2],
            pool=True,
            norm='in',
            tn=73,
            num_prior_stage=3,
            ema_moment=0.99
        )
        
        # 2. 載入 MASS 預訓練權重
        checkpoint = torch.load('mass_base.pth', map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)
        self.encoder_model.load_state_dict(state_dict, strict=False)
        
        # 3. 凍結 Encoder 權重 (Linear Probing 核心)
        if freeze_encoder:
            for param in self.encoder_model.parameters():
                param.requires_grad = False
            print("🔒 Encoder 權重已成功凍結！只訓練分類頭 (Linear Probe)")
            
        # 4. 全局平均池化與分類頭 (512 維特徵 -> 2 類)
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # 萃取 5 個尺度的特徵圖，取最深層 Bottleneck (512 通道)
        features = self.encoder_model.encoder(x)
        bottleneck = features[0] # 維度: [B, 512, 6, 8, 12]
        
        # 全局池化成向量: [B, 512, 1, 1, 1] -> [B, 512]
        pooled = self.global_pool(bottleneck).flatten(1)
        
        # 分類頭輸出 logits: [B, num_classes]
        logits = self.head(pooled)
        return logits

if __name__ == "__main__":
    print("測試 MASSClassifier 模型結構...")
    dummy_input = torch.randn(2, 1, 96, 128, 192) # Batch = 2
    model = MASSClassifier(num_classes=2)
    out = model(dummy_input)
    print(f"✅ 模型輸出 shape: {out.shape} (成功對應 2 個分類類別)")