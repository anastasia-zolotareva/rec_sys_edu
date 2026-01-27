#!/usr/bin/env python
"""
Скрипт для полной оценки системы рекомендаций.

Использование:
    python scripts/evaluate_system.py [--test-users N] [--recommendations-per-user K]
"""

import sys
import os
import argparse
import json
import torch
import pandas as pd

# Добавление корневой директории в путь
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loaders import load_all_data
from src.data.dataset import ITMDataset
from src.models.deepfm_svdpp import DeepFMSVDPlusPlus
from src.models.dueling_dqn import DuelingDQN
from src.environment.educational_env import EducationalEnvironment
from src.training.config import DEEPFM_CONFIG
from src.evaluation.comparative_tester import ComparativeTester
from src.evaluation.long_term_evaluator import LongTermEvaluator
from src.utils.helpers import set_seed


def main():
    """Основная функция оценки системы."""
    parser = argparse.ArgumentParser(description='Оценка системы рекомендаций')
    parser.add_argument('--test-users', type=int, default=20,
                       help='Количество пользователей для тестирования (по умолчанию: 20)')
    parser.add_argument('--recommendations-per-user', type=int, default=10,
                       help='Количество рекомендаций на пользователя (по умолчанию: 10)')
    parser.add_argument('--trajectory-length', type=int, default=100,
                       help='Длина траектории для долгосрочной оценки (по умолчанию: 100)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("ПОЛНАЯ ОЦЕНКА СИСТЕМЫ РЕКОМЕНДАЦИЙ")
    print("="*60)
    
    # Установка seed
    set_seed(42)
    
    # Определение устройства
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nИспользуется устройство: {device}")
    
    # Загрузка данных
    print("\nЗагрузка данных...")
    if os.path.exists('data/processed/ratings_processed.csv'):
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
    
    # Загрузка моделей
    print("\nЗагрузка моделей...")
    
    # DeepFM+SVD++
    model_path = 'data/models/deepfm_svdplusplus_best.pth'
    if not os.path.exists(model_path):
        print(f"  ✗ Модель не найдена: {model_path}")
        print("  Сначала запустите: python scripts/train_deepfm.py")
        sys.exit(1)
    
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
    print("  ✓ DeepFM+SVD++ загружена")
    
    # DQN Agent
    dqn_path = 'data/models/dqn_agent_checkpoint.pth'
    if not os.path.exists(dqn_path):
        print(f"  ✗ DQN агент не найден: {dqn_path}")
        print("  Сначала запустите: python scripts/train_dqn.py")
        sys.exit(1)
    
    state_dim = 65
    action_dim = dataset.n_items
    dqn_agent = DuelingDQN(state_dim, action_dim, hidden_dims=[256, 128, 64], device=device)
    dqn_checkpoint = torch.load(dqn_path, map_location=device, weights_only=False)
    dqn_agent.load_state_dict(dqn_checkpoint['agent_state_dict'])
    print("  ✓ DQN агент загружен")
    
    # Создание среды
    print("\nСоздание среды...")
    env = EducationalEnvironment(ratings, users, items, model, dataset)
    
    # 1. Сравнительное тестирование
    print("\n" + "="*60)
    print("1. СРАВНИТЕЛЬНОЕ ТЕСТИРОВАНИЕ")
    print("="*60)
    
    tester = ComparativeTester(env, model, dqn_agent, dataset)
    comparative_results = tester.run_comparative_test(
        test_users=args.test_users,
        recommendations_per_user=args.recommendations_per_user
    )
    
    summary_df = tester.visualize_results(comparative_results)
    
    # 2. Долгосрочная оценка
    print("\n" + "="*60)
    print("2. ДОЛГОСРОЧНАЯ ОЦЕНКА")
    print("="*60)
    
    # Базовые модели
    def random_recommender(user_id, context, k=1):
        import numpy as np
        return [np.random.randint(0, dataset.n_items) for _ in range(k)]
    
    def popularity_recommender(user_id, context, k=1):
        item_popularity = dataset.ratings['ItemID_encoded'].value_counts()
        popular_items = item_popularity.index[:k].tolist()
        return popular_items
    
    def static_deepfm_recommender(user_id, context, k=1):
        item_ids = torch.arange(dataset.n_items).long().to(device)
        user_ids = torch.LongTensor([user_id] * dataset.n_items).to(device)
        class_ids = torch.LongTensor([context['class']] * dataset.n_items).to(device)
        semester_ids = torch.LongTensor([context['semester']] * dataset.n_items).to(device)
        lockdown_ids = torch.LongTensor([context['lockdown']] * dataset.n_items).to(device)
        
        with torch.no_grad():
            predictions = model(user_ids, item_ids, class_ids, semester_ids, lockdown_ids)
            scores = (0.5 * predictions['rating'] + 
                     0.3 * predictions['app'] + 
                     0.15 * predictions['data'] + 
                     0.05 * predictions['ease'])
            top_indices = torch.topk(scores, k).indices.cpu().numpy()
            return top_indices.tolist()
    
    long_term_evaluator = LongTermEvaluator(
        env=env,
        dqn_agent=dqn_agent,
        baseline_models={
            'Random': random_recommender,
            'Popularity': popularity_recommender,
            'DeepFM-SVD++ (Static)': static_deepfm_recommender
        }
    )
    
    long_term_results = long_term_evaluator.run_long_term_experiment(
        n_users=args.test_users,
        trajectory_length=args.trajectory_length
    )
    
    long_term_evaluator.visualize_long_term_results(long_term_results)
    
    # Сохранение результатов
    print("\n" + "="*60)
    print("СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
    print("="*60)
    
    results_to_save = {
        'comparative_test': {
            k: {
                'mean': {col: float(v['mean'][col]) for col in v['mean'].index},
                'std': {col: float(v['std'][col]) for col in v['std'].index}
            }
            for k, v in comparative_results.items()
        },
        'long_term_evaluation': {
            k: {
                'mean': {col: float(v['mean'][col]) for col in v['mean'].index},
                'std': {col: float(v['std'][col]) for col in v['std'].index}
            }
            for k, v in long_term_results.items()
        }
    }
    
    os.makedirs('results', exist_ok=True)
    results_path = 'results/evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results_to_save, f, indent=2)
    
    print(f"\nРезультаты сохранены: {results_path}")
    
    # Итоговый отчет
    print("\n" + "="*60)
    print("ИТОГОВЫЙ ОТЧЕТ")
    print("="*60)
    
    print("\nСравнительное тестирование:")
    for model_name, results in comparative_results.items():
        mean = results['mean']
        print(f"  {model_name}:")
        print(f"    Precision@10: {mean['Precision@10']:.3f}")
        print(f"    Recall@10: {mean['Recall@10']:.3f}")
        print(f"    F1@10: {mean['F1@10']:.3f}")
    
    print("\nДолгосрочная оценка:")
    for model_name, results in long_term_results.items():
        mean = results['mean']
        print(f"  {model_name}:")
        print(f"    CDR: {mean['CDR']:.3f}")
        print(f"    Retention Rate: {mean['Retention_Rate']:.3f}")
        print(f"    Learning Slope: {mean['Learning_Slope']:.3f}")


if __name__ == "__main__":
    main()
