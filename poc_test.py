import torch
import sys
import os

# 確保 Python 抓得到 models 資料夾
sys.path.append(os.path.abspath(".")) 
from models.iris import Iris

def run_encoder(model, dummy_input):
    """執行特徵萃取的獨立函數"""
    model.eval()
    with torch.no_grad():
        features = model.encoder(dummy_input)
    return features

def main():
    # 根據 YAML 發現的真實大小：96x128x192
    print("1. 建立測試用的 3D 假影像 (Batch=1, Channel=1, 96x128x192)...")
    dummy_input = torch.randn(1, 1, 96, 128, 192)

    print("2. 實例化模型 (使用 YAML 中的精確參數)...")
    # 將 YAML 裡面的 model 設定完整搬過來
    model = Iris(
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

    print("3. 載入 MASS 預訓練權重...")
    try:
        checkpoint = torch.load('mass_base.pth', map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)
        # 由於參數完全正確，載入過程應該會非常順利
        model.load_state_dict(state_dict, strict=False)
        print("✅ 權重初步載入成功！")
    except FileNotFoundError:
        print("❌ 找不到 mass_base.pth，請確認檔案是否在同一層目錄下。")
        return

    print("4. 單獨使用 Encoder 進行特徵萃取 (Forward Pass)...")
    features = run_encoder(model, dummy_input)

    print("5. 觀察輸出的特徵維度 (動態偵測)：")
    # 針對不同的回傳格式動態解析
    if isinstance(features, (list, tuple)):
        print(f"  - 模型回傳了 {len(features)} 個不同尺度的特徵圖：")
        for i, feat in enumerate(features):
            print(f"    * 第 {i} 層維度: {feat.shape}")
    elif isinstance(features, dict):
        print("  - 模型回傳了 Dict 格式的特徵：")
        for key, feat in features.items():
            print(f"    * [{key}] 維度: {feat.shape}")
    else:
        print(f"  - 模型回傳單一特徵，維度: {features.shape}")

    print("🚀 PoC 腳本測試完畢！")

if __name__ == "__main__":
    main()