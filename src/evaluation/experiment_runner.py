"""
Запуск базовых экспериментов обучения.

ExperimentRunner управляет процессом обучения и оценки агента,
включая сравнение с базовыми методами и ablation study.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional
from tqdm import tqdm


class ExperimentRunner:
    """
    Запуск экспериментов по валидации.
    
    Управляет:
    - Обучением агента
    - Сравнением с базовыми методами
    - Ablation study
    - Визуализацией результатов
    """
    
    def __init__(self, trainer, env):
        """
        Инициализация runner.
        
        Args:
            trainer: DQNTrainer для обучения агента
            env: EducationalEnvironment для тестирования
        """
        self.trainer = trainer
        self.env = env
        self.results = {
            'training_rewards': [],
            'training_losses': [],
            'evaluation_scores': [],
            'epsilon_values': []
        }
    
    def run_training_experiment(
        self,
        n_episodes: int = 100,
        eval_interval: int = 10,
        max_steps_per_episode: Optional[int] = None,
    ) -> Dict[str, List]:
        """
        Эксперимент по обучению.
        
        Args:
            n_episodes: Количество эпизодов обучения
            eval_interval: Интервал оценки (каждые N эпизодов)
            max_steps_per_episode: Максимум шагов в обучающем эпизоде.
        
        Returns:
            Словарь с результатами обучения
        """
        print(f"Запуск обучения на {n_episodes} эпизодов...")
        
        for episode in tqdm(range(n_episodes)):
            # Обучение на одном эпизоде
            episode_reward, avg_loss = self.trainer.train_episode(
                max_steps=max_steps_per_episode
            )
            
            # Сохранение результатов
            self.results['training_rewards'].append(episode_reward)
            self.results['training_losses'].append(avg_loss)
            self.results['epsilon_values'].append(self.trainer.epsilon)
            
            # Периодическая оценка
            if (episode + 1) % eval_interval == 0:
                eval_results = self.trainer.evaluate(n_episodes=3)
                self.results['evaluation_scores'].append({
                    'episode': episode + 1,
                    'mean_reward': eval_results['mean_reward'],
                    'std_reward': eval_results['std_reward']
                })
                
                print(f"\nЭпизод {episode + 1}:")
                print(f"  Награда обучения: {episode_reward:.3f}")
                print(f"  Средняя оценка: {eval_results['mean_reward']:.3f}")
                print(f"  Epsilon: {self.trainer.epsilon:.3f}")
        
        print("\nОбучение завершено!")
        return self.results
    
    def compare_with_baselines(self) -> Dict[str, Dict[str, float]]:
        """
        Сравнение с базовыми методами.
        
        Returns:
            Словарь с результатами сравнения
        """
        print("\nСравнение с базовыми методами...")
        
        baselines = {
            'random': self._run_random_policy(),
            'popularity': self._run_popularity_policy()
        }
        
        # Оценка нашего агента
        our_agent_score = self.trainer.evaluate(n_episodes=10)
        
        comparison = {
            'our_agent': our_agent_score,
            'random': baselines['random'],
            'popularity': baselines['popularity']
        }
        
        # Визуализация сравнения
        self._plot_comparison(comparison)
        
        return comparison
    
    def _run_random_policy(self) -> Dict[str, float]:
        """Случайная политика."""
        rewards = []
        for _ in range(10):
            state = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action = np.random.randint(0, self.env.dataset.n_items)
                next_state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                state = next_state
            
            rewards.append(episode_reward)
        
        return {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards)
        }
    
    def _run_popularity_policy(self) -> Dict[str, float]:
        """Политика популярности (рекомендует самые популярные темы)."""
        # Анализ популярности тем
        item_popularity = self.env.ratings['ItemID_encoded'].value_counts()
        popular_items = item_popularity.index[:10].tolist()
        
        rewards = []
        for _ in range(10):
            state = self.env.reset()
            episode_reward = 0
            done = False
            step = 0
            
            while not done:
                # Циклическая рекомендация популярных тем
                if popular_items:
                    action = int(popular_items[step % len(popular_items)])
                else:
                    action = np.random.randint(0, self.env.dataset.n_items)
                next_state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                state = next_state
                step += 1
            
            rewards.append(episode_reward)
        
        return {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards)
        }
    
    def _plot_comparison(self, comparison: Dict[str, Dict[str, float]]):
        """Визуализация сравнения методов."""
        methods = list(comparison.keys())
        means = [comparison[m]['mean_reward'] for m in methods]
        stds = [comparison[m].get('std_reward', 0) for m in methods]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(methods, means, yerr=stds, capsize=10, alpha=0.7)
        plt.ylabel('Средняя награда за эпизод')
        plt.title('Сравнение с базовыми методами')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Добавление значений на столбцы
        for bar, mean in zip(bars, means):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{mean:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    def plot_training_progress(self):
        """Визуализация прогресса обучения."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # 1. Награды за эпизод
        axes[0, 0].plot(self.results['training_rewards'], alpha=0.7)
        axes[0, 0].set_title('Награды за эпизод обучения')
        axes[0, 0].set_xlabel('Эпизод')
        axes[0, 0].set_ylabel('Награда')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Скользящее среднее
        window = 10
        if len(self.results['training_rewards']) >= window:
            rolling_mean = pd.Series(self.results['training_rewards']).rolling(window).mean()
            axes[0, 0].plot(rolling_mean, 'r-', linewidth=2, label=f'Среднее ({window})')
            axes[0, 0].legend()
        
        # 2. Потери
        axes[0, 1].plot(self.results['training_losses'], alpha=0.7)
        axes[0, 1].set_title('Потери при обучении')
        axes[0, 1].set_xlabel('Эпизод')
        axes[0, 1].set_ylabel('Потеря')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Epsilon
        axes[1, 0].plot(self.results['epsilon_values'], 'g-', alpha=0.7)
        axes[1, 0].set_title('Динамика epsilon')
        axes[1, 0].set_xlabel('Эпизод')
        axes[1, 0].set_ylabel('Epsilon')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Оценка
        if self.results['evaluation_scores']:
            episodes = [score['episode'] for score in self.results['evaluation_scores']]
            eval_rewards = [score['mean_reward'] for score in self.results['evaluation_scores']]
            eval_stds = [score['std_reward'] for score in self.results['evaluation_scores']]
            
            axes[1, 1].errorbar(episodes, eval_rewards, yerr=eval_stds, 
                               fmt='o-', capsize=5, alpha=0.7)
            axes[1, 1].set_title('Оценка во время обучения')
            axes[1, 1].set_xlabel('Эпизод')
            axes[1, 1].set_ylabel('Средняя награда')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
