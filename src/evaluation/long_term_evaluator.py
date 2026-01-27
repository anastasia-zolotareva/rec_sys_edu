"""
Долгосрочная оценка полезности рекомендаций.

LongTermEvaluator оценивает долгосрочную эффективность рекомендаций
на длинных траекториях взаимодействия пользователей.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Callable
from scipy import stats

from .metrics import calculate_cumulative_discounted_reward, calculate_retention_rate


class LongTermEvaluator:
    """
    Оценка долгосрочной полезности рекомендаций.
    
    Оценивает метрики на длинных траекториях:
    - Cumulative Discounted Reward (CDR)
    - Retention Rate (удержание пользователей)
    - Learning Slope (тренд улучшения)
    - Coverage Progress (прогресс покрытия)
    """
    
    def __init__(self, env, dqn_agent, baseline_models: Dict[str, Callable]):
        """
        Инициализация оценщика.
        
        Args:
            env: EducationalEnvironment
            dqn_agent: Обученный DQN агент
            baseline_models: Словарь {name: recommender_function} базовых моделей
        """
        self.env = env
        self.dqn_agent = dqn_agent
        self.baseline_models = baseline_models
    
    def run_long_term_experiment(
        self,
        n_users: int = 30,
        trajectory_length: int = 100
    ) -> Dict[str, Dict[str, Any]]:
        """
        Запуск долгосрочного эксперимента.
        
        Args:
            n_users: Количество пользователей для тестирования
            trajectory_length: Длина траектории взаимодействия
        
        Returns:
            Словарь с результатами для каждой модели
        """
        results = {}
        
        # Получаем уникальных пользователей
        unique_users = self.env.ratings['UserID_encoded'].unique()
        test_users = np.random.choice(
            unique_users,
            min(n_users, len(unique_users)),
            replace=False
        )
        
        for model_name, model in [('DQN', self.dqn_agent)] + list(self.baseline_models.items()):
            print(f"\nТестирование {model_name}...")
            
            user_results = []
            for user_idx in test_users:
                # Сброс среды для нового пользователя
                state = self.env.reset(user_id=int(user_idx))
                
                rewards = []
                recommended_items = []
                relevant_items = []
                
                for step in range(trajectory_length):
                    # Получение рекомендации
                    if model_name == 'DQN':
                        action = model.get_action(state, epsilon=0.01)
                    else:
                        # Для базовых моделей используем функцию рекомендателя
                        context = self.env.current_context
                        recommendations = model(int(user_idx), context, k=1)
                        action = recommendations[0] if recommendations else np.random.randint(0, self.env.dataset.n_items)
                    
                    recommended_items.append(action)
                    
                    # Шаг в среде
                    next_state, reward, done, info = self.env.step(action)
                    
                    rewards.append(reward)
                    
                    # Сбор ground truth
                    if reward > 0.7:
                        relevant_items.append(action)
                    
                    state = next_state
                    if done:
                        break
                
                # Расчет метрик для этого пользователя
                user_metrics = self._calculate_user_metrics(
                    recommended_items, relevant_items, rewards, trajectory_length
                )
                user_results.append(user_metrics)
            
            # Агрегация по всем пользователям
            results[model_name] = self._aggregate_results(user_results)
        
        return results
    
    def _calculate_user_metrics(
        self,
        recommendations: List[int],
        relevant_items: List[int],
        rewards: List[float],
        trajectory_length: int
    ) -> Dict[str, float]:
        """
        Расчет всех метрик для одного пользователя.
        
        Args:
            recommendations: Список рекомендованных предметов
            relevant_items: Список релевантных предметов
            rewards: Список наград
            trajectory_length: Длина траектории
        
        Returns:
            Словарь с метриками
        """
        metrics = {}
        
        # 1. Традиционные метрики (на всей траектории)
        k = min(90, len(recommendations))
        true_positives = len(set(recommendations[:k]) & set(relevant_items))
        
        metrics['Precision@10'] = true_positives / k if k > 0 else 0
        metrics['Recall@10'] = true_positives / len(relevant_items) if relevant_items else 0
        metrics['F1@10'] = 2 * metrics['Precision@10'] * metrics['Recall@10'] / (
            metrics['Precision@10'] + metrics['Recall@10'] + 1e-8
        )
        
        # 2. Долгосрочные метрики
        # Cumulative Discounted Reward (CDR)
        metrics['CDR'] = calculate_cumulative_discounted_reward(rewards, gamma=0.99)
        
        # Retention Rate
        metrics['Retention_Rate'] = calculate_retention_rate(rewards, threshold=0.5)
        
        # Learning Progress (тренд улучшения наград)
        if len(rewards) >= 10:
            segments = np.array_split(rewards, 5)
            segment_means = [np.mean(seg) for seg in segments]
            x = range(len(segment_means))
            slope, _ = np.polyfit(x, segment_means, 1)
            metrics['Learning_Slope'] = slope
        else:
            metrics['Learning_Slope'] = 0
        
        # Coverage прогресс
        coverage_progress = []
        for i in range(1, len(recommendations) + 1):
            coverage_progress.append(
                len(set(recommendations[:i])) / min(i, self.env.dataset.n_items)
            )
        
        metrics['Coverage_Progress'] = coverage_progress[-1] if coverage_progress else 0
        metrics['Final_Coverage'] = len(set(recommendations)) / self.env.dataset.n_items
        
        metrics['rewards_progress'] = rewards
        
        return metrics
    
    def _aggregate_results(self, user_results: List[Dict[str, float]]) -> Dict[str, Any]:
        """
        Агрегация результатов по пользователям.
        
        Args:
            user_results: Список метрик для каждого пользователя
        
        Returns:
            Словарь с агрегированными метриками
        """
        df = pd.DataFrame(user_results)
        
        cols = [
            'CDR', 'Precision@10', 'Recall@10', 'F1@10',
            'Retention_Rate', 'Learning_Slope',
            'Coverage_Progress', 'Final_Coverage'
        ]
        
        return {
            'mean': df[cols].mean(),
            'std': df[cols].std(),
            'data': df
        }
    
    def visualize_long_term_results(self, results: Dict[str, Dict[str, Any]]):
        """
        Визуализация долгосрочных результатов.
        
        Args:
            results: Результаты от run_long_term_experiment
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        models = list(results.keys())
        
        # 1. Cumulative Discounted Reward
        cdrs = [results[m]['mean']['CDR'] for m in models]
        cdrs_std = [results[m]['std']['CDR'] for m in models]
        
        axes[0, 0].bar(models, cdrs, yerr=cdrs_std, capsize=5, alpha=0.7)
        axes[0, 0].set_title('Cumulative Discounted Reward')
        axes[0, 0].set_ylabel('CDR')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Precision@10
        precisions = [results[m]['mean']['Precision@10'] for m in models]
        precisions_std = [results[m]['std']['Precision@10'] for m in models]
        
        axes[0, 1].bar(models, precisions, yerr=precisions_std, capsize=5, alpha=0.7)
        axes[0, 1].set_title('Precision@10 (долгосрочная)')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Retention Rate
        retentions = [results[m]['mean']['Retention_Rate'] for m in models]
        retentions_std = [results[m]['std']['Retention_Rate'] for m in models]
        
        axes[0, 2].bar(models, retentions, yerr=retentions_std, capsize=5, alpha=0.7)
        axes[0, 2].set_title('Retention Rate (удержание)')
        axes[0, 2].set_ylabel('Retention Rate')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # 4. Learning Slope
        slopes = [results[m]['mean']['Learning_Slope'] for m in models]
        slopes_std = [results[m]['std']['Learning_Slope'] for m in models]
        
        axes[1, 0].bar(models, slopes, yerr=slopes_std, capsize=5, alpha=0.7)
        axes[1, 0].set_title('Learning Slope (тренд улучшения)')
        axes[1, 0].set_ylabel('Slope')
        axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.3)
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 5. Final Coverage
        coverages = [results[m]['mean']['Final_Coverage'] for m in models]
        coverages_std = [results[m]['std']['Final_Coverage'] for m in models]
        
        axes[1, 1].bar(models, coverages, yerr=coverages_std, capsize=5, alpha=0.7)
        axes[1, 1].set_title('Final Coverage (покрытие)')
        axes[1, 1].set_ylabel('Coverage')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # 6. Статистическая значимость различий
        # Сравнение DQN с лучшим базовым методом
        baseline_names = [m for m in models if m != 'DQN']
        if baseline_names:
            best_baseline = max(baseline_names, key=lambda x: results[x]['mean']['CDR'])
            
            dqn_data = results['DQN']['data']['CDR']
            baseline_data = results[best_baseline]['data']['CDR']
            
            # T-test для статистической значимости
            t_stat, p_value = stats.ttest_ind(dqn_data, baseline_data, equal_var=False)
            
            axes[1, 2].text(0.1, 0.5, f'DQN vs {best_baseline}\n', fontsize=12)
            axes[1, 2].text(0.1, 0.4, f't-statistic: {t_stat:.3f}', fontsize=10)
            axes[1, 2].text(0.1, 0.3, f'p-value: {p_value:.4f}', fontsize=10)
            axes[1, 2].text(0.1, 0.2, f'Significant: {p_value < 0.05}', fontsize=10,
                           color='green' if p_value < 0.05 else 'red')
        axes[1, 2].set_title('Статистическая значимость')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Дополнительный график: прогресс наград во времени
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for model_name in models:
            # Берем случайного пользователя для визуализации
            user_data = results[model_name]['data'].iloc[0]
            
            if 'rewards_progress' in user_data:
                rewards_progress = user_data['rewards_progress']
                ax.plot(rewards_progress, label=model_name, alpha=0.7)
        
        ax.set_xlabel('Шаг рекомендации')
        ax.set_ylabel('Награда')
        ax.set_title('Прогресс наград во времени (один пользователь)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
