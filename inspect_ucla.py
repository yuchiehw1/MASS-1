import pandas as pd
import glob
import os

def main():
    tsv_path = "./data/ucla/participants.tsv"
    
    if not os.path.exists(tsv_path):
        print(f"❌ 找不到 {tsv_path}，請確認檔案位置。")
        return

    # 1. 讀取 UCLA 標籤檔
    df = pd.read_csv(tsv_path, sep='\t')
    print("📊 UCLA 資料集診斷類別分佈 (diagnosis)：")
    print(df['diagnosis'].value_counts())
    print("\n前 5 筆受試者資料範例：")
    print(df[['participant_id', 'diagnosis', 'age', 'gender']].head())

    # 2. 搜尋下載好的 T1w MRI 檔案
    # OpenNeuro BIDS 結構通常是: ./data/ucla/sub-XXXXX/anat/sub-XXXXX_T1w.nii.gz
    t1_files = glob.glob("./data/ucla/sub-*/anat/*T1w.nii.gz")
    if not t1_files:
        # 備用搜尋：如果都在根目錄
        t1_files = glob.glob("./data/ucla/*T1w.nii.gz")

    print(f"\n📦 成功抓取到的 T1w 影像數量: {len(t1_files)}")
    if len(t1_files) > 0:
        print(f"第一個受試者影像路徑: {t1_files[0]}")

if __name__ == "__main__":
    main()