#!/usr/bin/env python
"""
Скрипт для обучения DQN агента.

Использование:
    python scripts/train_dqn.py [--episodes N] [--eval-interval M]
"""

import sys
import os
import argparse
import torch

# Добавление корневой директории в путь
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loaders import load_all_data
from src.data.dataset import ITMDataset
from src.models.deepfm_svdpp import DeepFMSVDPlusPlus
from src.models.dueling_dqn import DuelingDQN
from src.environment.educational_env import EducationalEnvironment
from src.training.replay_buffer import PrioritizedReplayBuffer
from src.training.trainer import DQNTrainer
from src.training.config import TRAIN_CONFIG, DEEPFM_CONFIG, REPLAY_BUFFER_CONFIG
from src.evaluation.experiment_runner import ExperimentRunner
from src.utils.helpers import set_seed


def main():
    """Основная функция обучения DQN агента."""
    parser = argparse.ArgumentParser(description='Обучение DQN агента')
    parser.add_argument('--episodes', type=int, default=200,
                       help='Количество эпизодов обучения (по умолчанию: 200)')
    parser.add_argument('--eval-interval', type=int, default=20,
                       help='Интервал оценки (по умолчанию: 20)')
    parser.add_argument('--load-model', type=str, default=None,
                       help='Путь к предобученной модели DeepFM (опционально)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("ОБУЧЕНИЕ DQN АГЕНТА")
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
    dataset = ITMDataset(ratings, users, items)
    
    # Загрузка или создание модели DeepFM+SVD++
    print("\nЗагрузка модели DeepFM+SVD++...")
    model_path = args.load_model or 'data/models/deepfm_svdplusplus_best.pth'
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
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
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print(f"  ✓ Модель загружена из {model_path}")
    else:
        print(f"  ✗ Модель не найдена: {model_path}")
        print("  Сначала запустите: python scripts/train_deepfm.py")
        sys.exit(1)
    
    # Создание среды
    print("\nСоздание среды...")
    env = EducationalEnvironment(ratings, users, items, model, dataset)
    
    # Создание DQN агента
    print("\nСоздание DQN агента...")
    state_dim = 65
    action_dim = dataset.n_items
    
    dqn_agent = DuelingDQN(
        state_dim,
        action_dim,
        hidden_dims=[256, 128, 64],
        device=device
    )
    
    # Создание буфера и тренера
    replay_buffer = PrioritizedReplayBuffer(
        capacity=REPLAY_BUFFER_CONFIG['capacity'],
        alpha=REPLAY_BUFFER_CONFIG['alpha'],
        beta=REPLAY_BUFFER_CONFIG['beta']
    )
    
    trainer = DQNTrainer(env, dqn_agent, replay_buffer, TRAIN_CONFIG)
    
    # Загрузка существующего чекпоинта (если есть)
    checkpoint_path = 'data/models/dqn_agent_checkpoint.pth'
    if os.path.exists(checkpoint_path):
        print(f"\nЗагрузка существующего чекпоинта: {checkpoint_path}")
        trainer.load_checkpoint(checkpoint_path)
    
    # Запуск обучения
    print("\n" + "="*60)
    print("ЗАПУСК ОБУЧЕНИЯ")
    print("="*60)
    print(f"Эпизодов: {args.episodes}")
    print(f"Интервал оценки: {args.eval_interval}")
    
    experiment_runner = ExperimentRunner(trainer, env)
    training_results = experiment_runner.run_training_experiment(
        n_episodes=args.episodes,
        eval_interval=args.eval_interval
    )
    
    # Сохранение модели
    os.makedirs('data/models', exist_ok=True)
    trainer.save_checkpoint(checkpoint_path)
    
    # Визуализация прогресса
    print("\nВизуализация прогресса обучения...")
    experiment_runner.plot_training_progress()
    
    print("\n" + "="*60)
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print("="*60)
    print(f"Модель сохранена: {checkpoint_path}")
    print(f"Финальный epsilon: {trainer.epsilon:.3f}")
    print(f"Средняя награда (последние 10 эпизодов): "
          f"{sum(training_results['training_rewards'][-10:]) / 10:.3f}")


if __name__ == "__main__":
    main()
