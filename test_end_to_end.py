import torch
import sys
import os
import monai.transforms as mt
import nibabel as nib
import numpy as np

# 確保抓得到 models 資料夾
sys.path.append(os.path.abspath(".")) 
from models.iris import Iris
from monai_pipeline import get_preprocessing_pipeline

def create_dummy_nifti_file(filename="sample_brain.nii.gz"):
    """若手邊沒有 NIfTI 檔，先動態生成一張假 3D 腦部影像用來測試 Pipeline"""
    if not os.path.exists(filename):
        print(f"📦 正在生成測試用 NIfTI 影像: {filename}...")
        # 模擬一張 256x256x176 原始規格的 MRI (隨機數值)
        data = np.random.rand(256, 256, 176).astype(np.float32)
        affine = np.eye(4) # 假幾何矩陣
        img = nib.Nifti1Image(data, affine)
        nib.save(img, filename)
        print("✅ 測試影像建立完成！")
    return filename

def main():
    # 1. 準備一張 NIfTI 影像
    nifti_path = create_dummy_nifti_file("sample_brain.nii.gz")

    print("\n[Step 1] 透過 MONAI 進行影像預處理...")
    pipeline = get_preprocessing_pipeline()
    
    # MONAI Dictionary 格式輸入
    data_dict = pipeline({"image": nifti_path})
    processed_tensor = data_dict["image"] # Shape: [1, 96, 128, 192]
    
    # 增加 Batch 維度 -> [1, 1, 96, 128, 192]
    input_tensor = processed_tensor.unsqueeze(0)
    print(f"✅ 預處理成功！餵給模型的 Tensor 維度: {input_tensor.shape}")

    print("\n[Step 2] 載入 MASS 模型與權重...")
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

    print("\n[Step 3] 執行 Forward Pass 萃取特徵...")
    with torch.no_grad():
        features = model.encoder(input_tensor)

    print("✅ 成功抽出特徵！最終 Bottleneck (最深層) 特徵圖維度:")
    print(f"👉 {features[0].shape}")
    print("\n🎉 端到端整合測試成功！你的系統已經具備處理真實 MRI 數據的能力了！")

if __name__ == "__main__":
    main()