import os
import glob
import pandas as pd
import torch
from torch.utils.data import Dataset
from monai_pipeline import get_preprocessing_pipeline

class UCLABrainDataset(Dataset):
    def __init__(self, tsv_path="./data/ucla/participants.tsv", data_dir="./data/ucla", target_classes=None):
        """
        UCLA 腦部 MRI Dataset
        :param target_classes: 想訓練的類別列表，預設為 ['CONTROL', 'SCHZ'] (健康對照 vs 思覺失調症)
        """
        if target_classes is None:
            target_classes = ['CONTROL', 'SCHZ']
            
        self.data_dir = data_dir
        self.pipeline = get_preprocessing_pipeline()
        
        # 1. 讀取標籤檔並篩選目標類別
        df = pd.read_csv(tsv_path, sep='\t')
        df = df[df['diagnosis'].isin(target_classes)].copy()
        
        # 2. 建立了類別映射文字 -> 數字標籤 (如: CONTROL -> 0, SCHZ -> 1)
        self.label_map = {cls_name: i for i, cls_name in enumerate(target_classes)}
        df['label'] = df['diagnosis'].map(self.label_map)
        
        # 3. 搜尋所有 T1w 檔案並進行 ID 比對
        t1_files = glob.glob(os.path.join(data_dir, "sub-*/anat/*T1w.nii.gz"))
        if not t1_files:
            t1_files = glob.glob(os.path.join(data_dir, "*T1w.nii.gz"))
            
        path_dict = {}
        for path in t1_files:
            # 從檔名解析出 sub-XXXXX
            sub_id = os.path.basename(path).split('_')[0]
            path_dict[sub_id] = path
            
        # 4. 只保留「同時有 MRI 檔案與標籤」的受試者
        self.samples = []
        for _, row in df.iterrows():
            sub_id = row['participant_id']
            if sub_id in path_dict:
                self.samples.append({
                    'image_path': path_dict[sub_id],
                    'label': row['label'],
                    'sub_id': sub_id
                })
                
        print(f"✅ 成功載入 UCLA Dataset！包含 {len(self.samples)} 筆有效樣本 (目標類別: {self.label_map})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 執行 MONAI 預處理
        data_dict = self.pipeline({"image": sample['image_path']})
        image_tensor = data_dict["image"] # [1, 96, 128, 192]
        
        label_tensor = torch.tensor(sample['label'], dtype=torch.long)
        
        return image_tensor, label_tensor, sample['sub_id']

# 快速測試用
if __name__ == "__main__":
    dataset = UCLABrainDataset()
    img, lbl, sub_id = dataset[0]
    print(f"單筆測試成功：受試者 {sub_id} | 影像維度 {img.shape} | 標籤數字 {lbl.item()}")