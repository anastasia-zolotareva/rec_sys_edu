"""
Модуль работы с данными.

Включает:
- Загрузку данных с Kaggle
- Предобработку данных
- PyTorch Dataset для ITM-REC
"""

try:
    from .dataset import ITMDataset
    from .loaders import (
        download_kaggle_dataset,
        load_ratings,
        load_users,
        load_items,
        load_all_data
    )
    from .preprocessing import (
        fill_missing_values,
        encode_categorical,
        normalize_ratings,
        validate_data
    )
    
    __all__ = [
        'ITMDataset',
        'download_kaggle_dataset',
        'load_ratings',
        'load_users',
        'load_items',
        'load_all_data',
        'fill_missing_values',
        'encode_categorical',
        'normalize_ratings',
        'validate_data',
    ]
except ImportError:
    # Модули еще не реализованы
    __all__ = []
