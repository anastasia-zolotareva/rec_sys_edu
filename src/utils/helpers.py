"""
Вспомогательные функции.
"""

import torch
import numpy as np
import random
from typing import Optional
import os


def set_seed(seed: int = 42):
    """
    Установка seed для воспроизводимости.
    
    Args:
        seed: Значение seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_model(model: torch.nn.Module, filepath: str, metadata: Optional[dict] = None):
    """
    Сохранение модели.
    
    Args:
        model: PyTorch модель
        filepath: Путь для сохранения
        metadata: Дополнительные метаданные для сохранения
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_class': model.__class__.__name__,
    }
    
    if metadata:
        checkpoint.update(metadata)
    
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    torch.save(checkpoint, filepath)
    print(f"Модель сохранена: {filepath}")


def load_model(model: torch.nn.Module, filepath: str, device: Optional[torch.device] = None) -> dict:
    """
    Загрузка модели.
    
    Args:
        model: PyTorch модель для загрузки весов
        filepath: Путь к файлу модели
        device: Устройство для загрузки
    
    Returns:
        Словарь с метаданными из чекпоинта
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Модель загружена: {filepath}")
    
    return checkpoint
