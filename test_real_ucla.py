import torch
import sys
import os

# 確保 Python 抓得到 models 資料夾
sys.path.append(os.path.abspath(".")) 
from models.iris import Iris
from monai_pipeline import get_preprocessing_pipeline

def main():
    # 使用剛剛檢視到的真實 UCLA 受試者影像路徑
    real_mri_path = "./data/ucla/sub-10448/anat/sub-10448_T1w.nii.gz"

    if not os.path.exists(real_mri_path):
        print(f"❌ 找不到檔案：{real_mri_path}")
        return

    print(f"1. 讀取並預處理真實 UCLA 腦部 MRI: {real_mri_path}...")
    pipeline = get_preprocessing_pipeline()
    
    # 執行 MONAI 預處理 (載入 NIfTI、轉 RAS、Resample、Crop/Pad 至 96x128x192)
    data_dict = pipeline({"image": real_mri_path})
    processed_tensor = data_dict["image"].unsqueeze(0) # 加上 Batch 維度 -> [1, 1, 96, 128, 192]
    print(f"✅ MONAI 預處理成功！張量維度已調整為: {processed_tensor.shape}")

    print("\n2. 載入 MASS 預訓練模型...")
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
    
    checkpoint = torch.load('mass_base.pth', map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    print("\n3. 將真實 MRI 送入 MASS 特徵萃取器 (Encoder)...")
    with torch.no_grad():
        features = model.encoder(processed_tensor)

    print("\n🎉 恭喜！成功從真實 UCLA 受試者腦部 MRI 中萃取出 5 個尺度的特徵圖：")
    for i, feat in enumerate(features):
        print(f"  * 第 {i} 層特徵維度: {feat.shape}")

    print("\n🚀 端到端真實數據測試成功完成！你的環境與數據流已全線通暢！")

if __name__ == "__main__":
    main()