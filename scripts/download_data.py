#!/usr/bin/env python
"""
Скрипт для загрузки данных с Kaggle.

Использование:
    python scripts/download_data.py
"""

import sys
import os

# Добавление корневой директории в путь
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loaders import download_kaggle_dataset, load_all_data


def main():
    """Основная функция загрузки данных."""
    print("="*60)
    print("ЗАГРУЗКА ДАННЫХ С KAGGLE")
    print("="*60)
    
    # Загрузка датасета
    dataset_name = "irecsys/itmrec"
    output_dir = "data/raw"
    
    print(f"\nЗагрузка датасета: {dataset_name}")
    print(f"Директория сохранения: {output_dir}")
    
    try:
        data_path = download_kaggle_dataset(dataset_name, output_dir)
        
        # Загрузка всех данных для проверки
        print("\nПроверка загруженных данных...")
        data = load_all_data(data_path, load_group_ratings=False)
        
        print("\n" + "="*60)
        print("ДАННЫЕ УСПЕШНО ЗАГРУЖЕНЫ!")
        print("="*60)
        print(f"\nРасположение: {data_path}")
        print(f"  - ratings: {data['ratings'].shape}")
        print(f"  - users: {data['users'].shape}")
        print(f"  - items: {data['items'].shape}")
        
    except Exception as e:
        print(f"\nОШИБКА при загрузке данных: {e}")
        print("\nУбедитесь, что:")
        print("  1. Установлен kagglehub: pip install kagglehub")
        print("  2. Настроен Kaggle API (kaggle.json в ~/.kaggle/)")
        sys.exit(1)


if __name__ == "__main__":
    main()
