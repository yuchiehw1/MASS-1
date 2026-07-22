import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1" # 自動啟用 CPU 補位

import torch
import torch.nn as nn

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import time

from ucla_dataset import UCLABrainDataset
from mass_classifier import MASSClassifier

def main():
    # 1. 偵測硬體加速 (Apple Silicon MPS / CUDA / CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"🖥️  使用運算裝置: {device}")

    # 2. 載入 Dataset 並切分 Train / Val (80% / 20%)
    full_dataset = UCLABrainDataset()
    val_size = int(len(full_dataset) * 0.2)
    train_size = len(full_dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    
    # 建立 PyTorch DataLoader (針對 3D MRI 記憶體大小，建議 Batch Size 設 4)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)

    print(f"📦 數據集準備完畢：訓練集 {len(train_dataset)} 筆 | 驗證集 {len(val_dataset)} 筆")

    # 3. 初始化模型、損失函數、優化器
    model = MASSClassifier(num_classes=2, freeze_encoder=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=1e-3, weight_decay=1e-4)

    # 4. 開始訓練迴圈 (訓練 5 個 Epochs 測試)
    epochs = 5
    print("\n🎯 開始訓練下游思覺失調症 (SCHZ) 分類器...")
    
    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        for imgs, labels, _ in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * imgs.size(0)
            preds = torch.argmax(outputs, dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            
        train_acc = train_correct / train_total
        avg_train_loss = train_loss / train_total

        # 驗證階段
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels, _ in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * imgs.size(0)
                preds = torch.argmax(outputs, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                
        val_acc = val_correct / val_total
        avg_val_loss = val_loss / val_total
        elapsed = time.time() - start_time

        print(f"Epoch [{epoch+1}/{epochs}] ({elapsed:.1f}s) | "
              f"Train Loss: {avg_train_loss:.4f} - Train Acc: {train_acc*100:.1f}% | "
              f"Val Loss: {avg_val_loss:.4f} - Val Acc: {val_acc*100:.1f}%")

    print("\n🎉 訓練完成！分類器運作正常！")

if __name__ == "__main__":
    main()