"""
Функции визуализации.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any


def plot_distributions(
    data: pd.DataFrame,
    columns: List[str],
    figsize: tuple = (14, 10)
):
    """
    Визуализация распределений рейтингов.
    
    Args:
        data: DataFrame с данными
        columns: Список колонок для визуализации
        figsize: Размер фигуры
    """
    n_cols = len(columns)
    n_rows = (n_cols + 1) // 2
    
    fig, axes = plt.subplots(n_rows, 2, figsize=figsize)
    if n_cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, col in enumerate(columns):
        ax = axes[idx]
        
        # Гистограмма с кривой плотности вероятности
        sns.histplot(data=data, x=col, bins=5, kde=True, ax=ax,
                     stat='percent', edgecolor='black')
        
        # Статистики
        mean_val = data[col].mean()
        median_val = data[col].median()
        mode_val = data[col].mode()[0] if not data[col].mode().empty else mean_val
        
        # Линии для основных статистик
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                   label=f'Среднее: {mean_val:.2f}')
        ax.axvline(median_val, color='green', linestyle='--', linewidth=2,
                   label=f'Медиана: {median_val:.2f}')
        ax.axvline(mode_val, color='blue', linestyle='--', linewidth=2,
                   label=f'Мода: {mode_val}')
        
        ax.set_title(col, fontsize=14, fontweight='bold')
        ax.set_xlabel('Оценка (1-5)', fontsize=12)
        ax.set_ylabel('Процент оценок (%)', fontsize=12)
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    # Скрываем лишние subplot'ы
    for idx in range(len(columns), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()


def plot_correlations(
    data: pd.DataFrame,
    columns: List[str],
    figsize: tuple = (10, 8)
):
    """
    Визуализация корреляционной матрицы.
    
    Args:
        data: DataFrame с данными
        columns: Список колонок для анализа
        figsize: Размер фигуры
    """
    correlation_matrix = data[columns].corr(method='pearson')
    
    plt.figure(figsize=figsize)
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm',
                center=0, square=True, linewidths=1,
                fmt='.3f', cbar_kws={'label': 'Коэффициент корреляции'})
    
    plt.title('Матрица корреляций Пирсона между критериями оценок',
              fontsize=16, fontweight='bold', pad=20)
    plt.xticks(fontsize=12, rotation=45)
    plt.yticks(fontsize=12, rotation=0)
    plt.tight_layout()
    plt.show()


def plot_training_progress(
    training_rewards: List[float],
    training_losses: Optional[List[float]] = None,
    epsilon_values: Optional[List[float]] = None,
    evaluation_scores: Optional[List[Dict[str, Any]]] = None,
    figsize: tuple = (12, 8)
):
    """
    Визуализация прогресса обучения.
    
    Args:
        training_rewards: Список наград за эпизоды
        training_losses: Список потерь за эпизоды
        epsilon_values: Список значений epsilon
        evaluation_scores: Список результатов оценки
        figsize: Размер фигуры
    """
    n_plots = 1
    if training_losses:
        n_plots += 1
    if epsilon_values:
        n_plots += 1
    if evaluation_scores:
        n_plots += 1
    
    n_rows = (n_plots + 1) // 2
    n_cols = 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    plot_idx = 0
    
    # 1. Награды за эпизод обучения
    axes[plot_idx].plot(training_rewards, alpha=0.7)
    axes[plot_idx].set_title('Награды за эпизод обучения')
    axes[plot_idx].set_xlabel('Эпизод')
    axes[plot_idx].set_ylabel('Награда')
    axes[plot_idx].grid(True, alpha=0.3)
    
    # Скользящее среднее
    window = 10
    if len(training_rewards) >= window:
        rolling_mean = pd.Series(training_rewards).rolling(window).mean()
        axes[plot_idx].plot(rolling_mean, 'r-', linewidth=2, label=f'Среднее ({window})')
        axes[plot_idx].legend()
    plot_idx += 1
    
    # 2. Потери при обучении
    if training_losses and plot_idx < len(axes):
        axes[plot_idx].plot(training_losses, alpha=0.7)
        axes[plot_idx].set_title('Потери при обучении')
        axes[plot_idx].set_xlabel('Эпизод')
        axes[plot_idx].set_ylabel('Потеря')
        axes[plot_idx].grid(True, alpha=0.3)
        plot_idx += 1
    
    # 3. Коэффициент epsilon для поиска-использования компромисса
    if epsilon_values and plot_idx < len(axes):
        axes[plot_idx].plot(epsilon_values, 'g-', alpha=0.7)
        axes[plot_idx].set_title('Динамика коэффициента epsilon')
        axes[plot_idx].set_xlabel('Эпизод')
        axes[plot_idx].set_ylabel('Epsilon')
        axes[plot_idx].grid(True, alpha=0.3)
        plot_idx += 1
    
    # 4. Оценка модели
    if evaluation_scores and plot_idx < len(axes):
        episodes = [score['episode'] for score in evaluation_scores]
        eval_rewards = [score['mean_reward'] for score in evaluation_scores]
        eval_stds = [score.get('std_reward', 0) for score in evaluation_scores]
        
        axes[plot_idx].errorbar(episodes, eval_rewards, yerr=eval_stds,
                               fmt='o-', capsize=5, alpha=0.7)
        axes[plot_idx].set_title('Оценка модели во время обучения')
        axes[plot_idx].set_xlabel('Эпизод')
        axes[plot_idx].set_ylabel('Средняя награда')
        axes[plot_idx].grid(True, alpha=0.3)
        plot_idx += 1
    
    # Скрываем лишние subplot'ы
    for idx in range(plot_idx, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()
