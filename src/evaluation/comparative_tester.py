"""
Сравнительное тестирование различных методов рекомендаций.

ComparativeTester сравнивает различные методы рекомендаций:
- Random (случайные рекомендации)
- Popularity (популярные предметы)
- DeepFM-SVD++ (статическая модель)
- DQN-Enhanced (наш агент)
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
from sklearn.metrics import precision_score, recall_score, f1_score

from .metrics import (
    calculate_precision_at_k,
    calculate_recall_at_k,
    calculate_f1_at_k,
    calculate_coverage,
    calculate_diversity,
    calculate_novelty
)


class ComparativeTester:
    """
    Класс для сравнительного тестирования всех моделей.
    
    Тестирует различные методы рекомендаций и сравнивает их
    по метрикам: Precision@K, Recall@K, F1@K, Coverage, Diversity, Novelty.
    """
    
    def __init__(self, env, deepfm_model, dqn_agent, dataset):
        """
        Инициализация тестера.
        
        Args:
            env: EducationalEnvironment
            deepfm_model: Обученная модель DeepFM+SVD++
            dqn_agent: Обученный DQN агент
            dataset: ITMDataset
        """
        self.env = env
        self.deepfm_model = deepfm_model
        self.dqn_agent = dqn_agent
        self.dataset = dataset
        self.n_items = dataset.n_items
        
        # Базовые модели
        self.baseline_models = {
            'Random': self.random_recommender,
            'Popularity': self.popularity_recommender,
            'DeepFM-SVD++ (Static)': self.static_deepfm_recommender,
            'DQN-Enhanced': self.dqn_recommender
        }
        
        # Статистика популярности предметов
        self._compute_item_popularity()
    
    def _compute_item_popularity(self):
        """Вычисление популярности предметов."""
        # Используем закодированные ID из dataset
        item_counts = self.dataset.ratings['ItemID_encoded'].value_counts()
        self.item_popularity = item_counts.to_dict()
        
        # Нормализация популярности
        max_pop = max(self.item_popularity.values()) if self.item_popularity else 1
        self.item_popularity_norm = {
            item: count / max_pop for item, count in self.item_popularity.items()
        }
    
    def random_recommender(self, user_id: int, context: Dict, k: int = 10) -> List[int]:
        """Случайный рекомендатель."""
        return np.random.choice(self.n_items, k, replace=False).tolist()
    
    def popularity_recommender(self, user_id: int, context: Dict, k: int = 10) -> List[int]:
        """Рекомендатель на основе популярности."""
        # Сортируем предметы по популярности
        sorted_items = sorted(
            self.item_popularity.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [int(item[0]) for item in sorted_items[:k]]
    
    def static_deepfm_recommender(self, user_id: int, context: Dict, k: int = 10) -> List[int]:
        """Статичный DeepFM-SVD++ рекомендатель."""
        device = self.deepfm_model.device
        
        # Создание тензоров для всех предметов
        item_ids = torch.arange(self.n_items).long().to(device)
        user_ids = torch.LongTensor([user_id] * self.n_items).to(device)
        class_ids = torch.LongTensor([context['class']] * self.n_items).to(device)
        semester_ids = torch.LongTensor([context['semester']] * self.n_items).to(device)
        lockdown_ids = torch.LongTensor([context['lockdown']] * self.n_items).to(device)
        
        with torch.no_grad():
            # Получение предсказаний для всех предметов
            predictions = self.deepfm_model(
                user_ids, item_ids, class_ids, semester_ids, lockdown_ids
            )
            
            # Комбинирование мультикритериальных оценок
            scores = (
                0.5 * predictions['rating'] +
                0.3 * predictions['app'] +
                0.15 * predictions['data'] +
                0.05 * predictions['ease']
            )
        
        # Выбор топ-k предметов
        top_indices = torch.topk(scores, k).indices.cpu().numpy()
        return top_indices.tolist()
    
    def dqn_recommender(self, user_id: int, context: Dict, k: int = 10) -> List[int]:
        """DQN-улучшенный рекомендатель."""
        state = self.env.reset(user_id=user_id, context=context)
        
        recommended_items = []
        
        for _ in range(k):
            action = self.dqn_agent.get_action(state, epsilon=0.01)
            recommended_items.append(action)
            
            # Обновление состояния (без реального взаимодействия)
            next_state, _, done, _ = self.env.step(action)
            state = next_state
            
            if done:
                break
        
        return recommended_items
    
    def calculate_metrics(
        self,
        recommendations: List[int],
        ground_truth: List[int],
        k: int = 10
    ) -> Dict[str, float]:
        """
        Расчет всех метрик качества.
        
        Args:
            recommendations: Список рекомендованных предметов
            ground_truth: Список релевантных предметов
            k: Количество рекомендаций для оценки
        
        Returns:
            Словарь с метриками
        """
        metrics = {}
        
        # Precision@k, Recall@k, F1@k
        metrics['Precision@10'] = calculate_precision_at_k(
            recommendations, ground_truth, k, self.n_items
        )
        metrics['Recall@10'] = calculate_recall_at_k(
            recommendations, ground_truth, k, self.n_items
        )
        metrics['F1@10'] = calculate_f1_at_k(
            recommendations, ground_truth, k, self.n_items
        )
        
        # Coverage (покрытие каталога)
        metrics['Coverage'] = calculate_coverage(recommendations, self.n_items)
        
        # Novelty (новизна - обратная популярность)
        metrics['Novelty'] = calculate_novelty(
            recommendations, self.item_popularity_norm, k
        )
        
        # Diversity (разнообразие рекомендаций)
        metrics['Diversity'] = calculate_diversity(recommendations)
        
        return metrics
    
    def run_comparative_test(
        self,
        test_users: int = 20,
        recommendations_per_user: int = 10
    ) -> Dict[str, Dict[str, Any]]:
        """
        Запуск сравнительного тестирования.
        
        Args:
            test_users: Количество пользователей для тестирования
            recommendations_per_user: Количество рекомендаций на пользователя
        
        Returns:
            Словарь с агрегированными результатами для каждой модели
        """
        results = {model_name: [] for model_name in self.baseline_models.keys()}
        
        print("=" * 60)
        print("СРАВНИТЕЛЬНОЕ ТЕСТИРОВАНИЕ МОДЕЛЕЙ")
        print("=" * 60)
        
        # Получаем уникальных пользователей из датасета
        unique_users = self.dataset.ratings['UserID_encoded'].unique()
        test_user_ids = np.random.choice(unique_users, min(test_users, len(unique_users)), replace=False)
        
        for i, user_id in enumerate(test_user_ids):
            print(f"\nПользователь {i+1}/{len(test_user_ids)}")
            
            # Случайный контекст
            context = {
                'class': np.random.randint(0, 3),
                'semester': np.random.randint(0, 2),
                'lockdown': np.random.randint(0, 3)
            }
            
            # "Правда" - предметы, которые пользователь оценил высоко (рейтинг > 3)
            user_ratings = self.dataset.ratings[
                self.dataset.ratings['UserID_encoded'] == user_id
            ]
            ground_truth = user_ratings[
                user_ratings['Rating'] > 3
            ]['ItemID_encoded'].tolist()[:20]  # Берем до 20 высокооцененных
            
            if len(ground_truth) < 5:
                continue  # Пропускаем пользователей с недостаточной историей
            
            for model_name, model_func in self.baseline_models.items():
                try:
                    # Получение рекомендаций
                    recommendations = model_func(user_id, context, k=recommendations_per_user)
                    
                    # Расчет метрик
                    metrics = self.calculate_metrics(recommendations, ground_truth, k=10)
                    
                    # Добавление кумулятивного вознаграждения для RL моделей
                    if model_name == 'DQN-Enhanced':
                        cumulative_reward = self._evaluate_dqn_trajectory(user_id, context)
                        metrics['Cumulative_Reward'] = cumulative_reward
                    else:
                        metrics['Cumulative_Reward'] = self._simulate_reward(
                            recommendations, user_id, context
                        )
                    
                    results[model_name].append(metrics)
                    
                    print(f"  {model_name}: Precision={metrics['Precision@10']:.3f}, "
                          f"Reward={metrics['Cumulative_Reward']:.3f}")
                    
                except Exception as e:
                    print(f"  Ошибка в {model_name}: {e}")
                    continue
        
        # Агрегация результатов
        aggregated_results = {}
        for model_name, metrics_list in results.items():
            if metrics_list:
                df = pd.DataFrame(metrics_list)
                aggregated_results[model_name] = {
                    'mean': df.mean(),
                    'std': df.std(),
                    'data': df
                }
        
        return aggregated_results
    
    def _evaluate_dqn_trajectory(self, user_id: int, context: Dict, max_steps: int = 10) -> float:
        """Оценка траектории DQN агента."""
        state = self.env.reset(user_id=user_id, context=context)
        cumulative_reward = 0
        
        for step in range(max_steps):
            action = self.dqn_agent.get_action(state, epsilon=0.01)
            next_state, reward, done, _ = self.env.step(action)
            cumulative_reward += reward
            state = next_state
            
            if done:
                break
        
        return cumulative_reward
    
    def _simulate_reward(self, recommendations: List[int], user_id: int, context: Dict) -> float:
        """Симуляция вознаграждения для не-RL моделей."""
        total_reward = 0
        
        for item in recommendations[:10]:
            # Используем средний рейтинг пользователя для предмета
            user_ratings = self.dataset.ratings[
                (self.dataset.ratings['UserID_encoded'] == user_id) &
                (self.dataset.ratings['ItemID_encoded'] == item)
            ]
            
            if not user_ratings.empty:
                reward = user_ratings['Rating'].mean() / 5.0  # Нормализация к [0,1]
            else:
                # Используем среднее по всем пользователям
                all_ratings = self.dataset.ratings[
                    self.dataset.ratings['ItemID_encoded'] == item
                ]
                reward = all_ratings['Rating'].mean() / 5.0 if not all_ratings.empty else 0.5
            
            total_reward += reward
        
        return total_reward / min(len(recommendations), 10)
    
    def visualize_results(self, aggregated_results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Визуализация результатов тестирования.
        
        Args:
            aggregated_results: Агрегированные результаты от run_comparative_test
        
        Returns:
            DataFrame с сводной таблицей результатов
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        # Подготовка данных для визуализации
        models = list(aggregated_results.keys())
        
        # 1. Precision@10
        precisions = [aggregated_results[m]['mean']['Precision@10'] for m in models]
        precisions_std = [aggregated_results[m]['std']['Precision@10'] for m in models]
        
        axes[0].bar(models, precisions, yerr=precisions_std, capsize=5, alpha=0.7)
        axes[0].set_title('Precision@10')
        axes[0].set_ylabel('Precision')
        axes[0].tick_params(axis='x', rotation=45)
        
        # 2. Recall@10
        recalls = [aggregated_results[m]['mean']['Recall@10'] for m in models]
        recalls_std = [aggregated_results[m]['std']['Recall@10'] for m in models]
        
        axes[1].bar(models, recalls, yerr=recalls_std, capsize=5, alpha=0.7)
        axes[1].set_title('Recall@10')
        axes[1].set_ylabel('Recall')
        axes[1].tick_params(axis='x', rotation=45)
        
        # 3. F1@10
        f1_scores = [aggregated_results[m]['mean']['F1@10'] for m in models]
        f1_std = [aggregated_results[m]['std']['F1@10'] for m in models]
        
        axes[2].bar(models, f1_scores, yerr=f1_std, capsize=5, alpha=0.7)
        axes[2].set_title('F1@10')
        axes[2].set_ylabel('F1 Score')
        axes[2].tick_params(axis='x', rotation=45)
        
        # 4. Cumulative Reward
        rewards = [aggregated_results[m]['mean']['Cumulative_Reward'] for m in models]
        rewards_std = [aggregated_results[m]['std']['Cumulative_Reward'] for m in models]
        
        axes[3].bar(models, rewards, yerr=rewards_std, capsize=5, alpha=0.7)
        axes[3].set_title('Cumulative Reward')
        axes[3].set_ylabel('Reward')
        axes[3].tick_params(axis='x', rotation=45)
        
        # 5. Coverage
        coverages = [aggregated_results[m]['mean']['Coverage'] for m in models]
        coverages_std = [aggregated_results[m]['std']['Coverage'] for m in models]
        
        axes[4].bar(models, coverages, yerr=coverages_std, capsize=5, alpha=0.7)
        axes[4].set_title('Coverage (Покрытие каталога)')
        axes[4].set_ylabel('Coverage')
        axes[4].tick_params(axis='x', rotation=45)
        
        # 6. Diversity
        diversities = [aggregated_results[m]['mean']['Diversity'] for m in models]
        diversities_std = [aggregated_results[m]['std']['Diversity'] for m in models]
        
        axes[5].bar(models, diversities, yerr=diversities_std, capsize=5, alpha=0.7)
        axes[5].set_title('Diversity (Разнообразие)')
        axes[5].set_ylabel('Diversity')
        axes[5].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        # Создание сводной таблицы
        summary_data = []
        for model_name in models:
            mean_metrics = aggregated_results[model_name]['mean']
            std_metrics = aggregated_results[model_name]['std']
            
            row = {
                'Model': model_name,
                'Precision@10': f"{mean_metrics['Precision@10']:.3f} ± {std_metrics['Precision@10']:.3f}",
                'Recall@10': f"{mean_metrics['Recall@10']:.3f} ± {std_metrics['Recall@10']:.3f}",
                'F1@10': f"{mean_metrics['F1@10']:.3f} ± {std_metrics['F1@10']:.3f}",
                'Cumulative_Reward': f"{mean_metrics['Cumulative_Reward']:.3f} ± {std_metrics['Cumulative_Reward']:.3f}",
                'Coverage': f"{mean_metrics['Coverage']:.3f} ± {std_metrics['Coverage']:.3f}",
                'Diversity': f"{mean_metrics['Diversity']:.3f} ± {std_metrics['Diversity']:.3f}"
            }
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        
        print("\n" + "=" * 100)
        print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
        print("=" * 100)
        print(summary_df.to_string(index=False))
        
        return summary_df
