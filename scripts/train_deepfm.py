#!/usr/bin/env python
"""
Скрипт для обучения модели DeepFM+SVD++.

Использование:
    python scripts/train_deepfm.py
"""

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim

# Добавление корневой директории в путь
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loaders import load_all_data
from src.data.dataset import ITMDataset
from src.models.deepfm_svdpp import DeepFMSVDPlusPlus
from src.training.config import DEEPFM_CONFIG
from src.utils.helpers import set_seed


def main():
    """Основная функция обучения DeepFM+SVD++."""
    print("="*60)
    print("ОБУЧЕНИЕ МОДЕЛИ DeepFM+SVD++")
    print("="*60)
    
    # Установка seed
    set_seed(42)
    
    # Определение устройства
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nИспользуется устройство: {device}")
    
    # Загрузка данных
    print("\nЗагрузка данных...")
    if os.path.exists('data/processed/ratings_processed.csv'):
        import pandas as pd
        ratings = pd.read_csv('data/processed/ratings_processed.csv')
        users = pd.read_csv('data/processed/users_processed.csv')
        items = pd.read_csv('data/processed/items_processed.csv')
    else:
        from src.data.loaders import download_kaggle_dataset
        data_path = download_kaggle_dataset("irecsys/itmrec", "data/raw")
        data = load_all_data(data_path)
        ratings = data['ratings']
        users = data['users']
        items = data['items']
    
    # Создание датасета
    print("\nСоздание ITMDataset...")
    dataset = ITMDataset(ratings, users, items)
    
    # Разделение на train/val/test
    train_loader, val_loader, test_loader = dataset.train_test_split(
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        batch_size=DEEPFM_CONFIG['batch_size'],
        shuffle=True,
        random_state=42
    )
    
    # Создание модели
    print("\nСоздание модели DeepFM+SVD++...")
    model = DeepFMSVDPlusPlus(
        n_users=dataset.n_users,
        n_items=dataset.n_items,
        n_classes=dataset.n_classes,
        n_semesters=dataset.n_semesters,
        n_lockdowns=dataset.n_lockdowns,
        device=device,
        embedding_dim=DEEPFM_CONFIG['embedding_dim'],
        hidden_dims=DEEPFM_CONFIG['hidden_dims'],
        dropout_rate=DEEPFM_CONFIG['dropout']
    )
    
    print(f"  Параметров: {sum(p.numel() for p in model.parameters()):,}")
    
    # Оптимизатор и функция потерь
    optimizer = optim.Adam(
        model.parameters(),
        lr=DEEPFM_CONFIG['lr'],
        weight_decay=DEEPFM_CONFIG.get('weight_decay', 1e-5)
    )
    criterion = nn.MSELoss()
    
    # Обучение
    n_epochs = DEEPFM_CONFIG['n_epochs']
    best_val_loss = float('inf')
    
    print(f"\nОбучение на {n_epochs} эпох...")
    for epoch in range(n_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for features, targets in train_loader:
            for key in features:
                features[key] = features[key].to(device)
            targets = targets.to(device)
            
            outputs = model(
                features['user_id'].squeeze(),
                features['item_id'].squeeze(),
                features['class'].squeeze(),
                features['semester'].squeeze(),
                features['lockdown'].squeeze()
            )
            
            loss = (criterion(outputs['rating'], targets[:, 0]) +
                   criterion(outputs['app'], targets[:, 1]) +
                   criterion(outputs['data'], targets[:, 2]) +
                   criterion(outputs['ease'], targets[:, 3])) / 4.0
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, targets in val_loader:
                for key in features:
                    features[key] = features[key].to(device)
                targets = targets.to(device)
                
                outputs = model(
                    features['user_id'].squeeze(),
                    features['item_id'].squeeze(),
                    features['class'].squeeze(),
                    features['semester'].squeeze(),
                    features['lockdown'].squeeze()
                )
                
                loss = (criterion(outputs['rating'], targets[:, 0]) +
                       criterion(outputs['app'], targets[:, 1]) +
                       criterion(outputs['data'], targets[:, 2]) +
                       criterion(outputs['ease'], targets[:, 3])) / 4.0
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{n_epochs}: "
              f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Сохранение лучшей модели
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs('data/models', exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'user_encoder': dataset.user_encoder,
                'item_encoder': dataset.item_encoder,
                'class_encoder': dataset.class_encoder,
                'semester_encoder': dataset.semester_encoder,
                'lockdown_encoder': dataset.lockdown_encoder,
                'rating_scaler': dataset.rating_scaler
            }, 'data/models/deepfm_svdplusplus_best.pth')
            print(f"  ✓ Сохранена лучшая модель (val_loss: {best_val_loss:.4f})")
    
    print("\n" + "="*60)
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print("="*60)
    print(f"Лучшая validation loss: {best_val_loss:.4f}")
    print(f"Модель сохранена: data/models/deepfm_svdplusplus_best.pth")


if __name__ == "__main__":
    main()
