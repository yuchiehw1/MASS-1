import monai.transforms as mt
import torch

def get_preprocessing_pipeline():
    """
    建置符合 MASS 模型需求的 MONAI 轉換流程
    """
    transforms = mt.Compose([
        # 1. 讀取影像 (支援 NIfTI 格式)
        mt.LoadImaged(keys=["image"]),
        
        # 2. 確保有 Channel 維度 (變成 C, D, H, W)
        mt.EnsureChannelFirstd(keys=["image"]),
        
        # 3. 統一方向為 RAS (Right, Anterior, Superior)
        mt.Orientationd(keys=["image"], axcodes="RAS"),
        
        # 4. 統一解析度為 1.5mm 等向性
        mt.Spacingd(keys=["image"], pixdim=(1.5, 1.5, 1.5), mode="bilinear"),
        
        # 5. 強度正規化 (將 MRI 數值縮放到 0~1 或標準常態分佈，這對神經網路很重要)
        mt.NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        
        # 6. 裁切或填充至官方 YAML 指定的輸入大小
        mt.ResizeWithPadOrCropd(keys=["image"], spatial_size=[96, 128, 192]),
        
        # 7. 轉換為 PyTorch Tensor
        mt.EnsureTyped(keys=["image"], dtype=torch.float32)
    ])
    
    return transforms

def main():
    print("初始化 MONAI 預處理 Pipeline...")
    pipeline = get_preprocessing_pipeline()
    
    print("\n✅ Pipeline 建置成功！")
    print("接下來我們可以拿一張真實的 NIfTI (.nii.gz) MRI 影像來測試了。")

if __name__ == "__main__":
    main()