"""
Тренер для обучения DQN агента.

DQNTrainer управляет процессом обучения DQN агента в среде,
используя Prioritized Experience Replay и target network.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Any

from ..models.dueling_dqn import DuelingDQN


class DQNTrainer:
    """
    Тренер для обучения DQN агента.
    
    Управляет:
    - Обучением на эпизодах
    - Управлением target network (мягкое и полное обновление)
    - Epsilon-greedy стратегией
    - Gradient clipping
    - Сохранением/загрузкой моделей
    
    Attributes:
        env: Образовательная среда
        agent: DQN агент для обучения
        buffer: Буфер воспроизведения опыта
        target_network: Target network для стабильного обучения
        device: Устройство для вычислений
        gamma: Коэффициент дисконтирования
        lr: Learning rate
        tau: Коэффициент мягкого обновления
        target_update_freq: Частота полного обновления target network
        batch_size: Размер батча
        epsilon: Текущее значение epsilon
        epsilon_start: Начальное значение epsilon
        epsilon_end: Конечное значение epsilon
        epsilon_decay: Скорость затухания epsilon
        optimizer: Оптимизатор
        loss_fn: Функция потерь
        step_count: Счетчик шагов обучения
        episode_rewards: Список наград за эпизоды
        losses: Список потерь
    """
    
    def __init__(
        self,
        env,
        agent: DuelingDQN,
        buffer,
        config: Dict[str, Any]
    ):
        """
        Инициализация тренера.
        
        Args:
            env: Образовательная среда (EducationalEnvironment)
            agent: DQN агент для обучения
            buffer: Буфер воспроизведения опыта (PrioritizedReplayBuffer)
            config: Словарь с конфигурацией обучения
        """
        self.env = env
        self.agent = agent
        self.buffer = buffer
        
        # Device
        self.device = agent.device
        
        # Инициализация target network с тем же устройством
        self.target_network = DuelingDQN(
            agent.state_dim,
            agent.action_dim,
            hidden_dims=getattr(agent, "hidden_dims", [256, 128, 64]),
            device=self.device
        )
        self.target_network.load_state_dict(agent.state_dict())
        self.target_network.to(self.device)
        self.target_network.eval()  # Target network always in evaluation mode
        
        # Конфигурация
        self.gamma = config.get('gamma', 0.99)
        self.lr = config.get('lr', 0.001)
        self.tau = config.get('tau', 0.01)  # Для мягкого обновления
        self.target_update_freq = config.get('target_update_freq', 100)
        self.batch_size = config.get('batch_size', 64)
        self.max_steps_per_episode = int(config.get('max_steps_per_episode', 100))
        self.epsilon_start = config.get('epsilon_start', 1.0)
        self.epsilon_end = config.get('epsilon_end', 0.01)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.use_action_mask = bool(config.get('use_action_mask', True))
        
        # Оптимизатор и функция потерь
        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=self.lr)
        self.loss_fn = nn.SmoothL1Loss(reduction='none')  # Huber loss с весами PER
        
        # Трекеры
        self.epsilon = self.epsilon_start
        self.step_count = 0
        self.episode_rewards = []
        self.losses = []
        
        # Перемещаем agent на device
        self.agent.to(self.device)
        
        print(f"Инициализация тренера на {self.device}")
        print(f"  Gamma: {self.gamma}")
        print(f"  Learning rate: {self.lr}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Epsilon: {self.epsilon_start} -> {self.epsilon_end}")
    
    def update_epsilon(self) -> float:
        """
        Обновление epsilon для ε-жадной стратегии.
        
        Returns:
            Новое значение epsilon
        """
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        return self.epsilon
    
    def soft_update_target(self):
        """Мягкое обновление target network."""
        for target_param, param in zip(
            self.target_network.parameters(),
            self.agent.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )
    
    def hard_update_target(self):
        """Полное обновление target network."""
        self.target_network.load_state_dict(self.agent.state_dict())
    
    def compute_td_targets(
        self,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        next_action_masks: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Вычисление целевых значений (TD targets).
        
        Поддерживает action_mask для OULAD и других датасетов,
        где допустимое множество действий зависит от состояния.
        
        Args:
            rewards: Награды [batch_size]
            next_states: Следующие состояния [batch_size, state_dim]
            dones: Флаги завершения [batch_size]
            next_action_masks: Маски допустимых действий [batch_size, action_dim] (опционально)
        
        Returns:
            TD targets [batch_size]
        """
        with torch.no_grad():
            # Q-значения от target network
            next_q_values = self.target_network(next_states)  # [batch_size, action_dim]
            
            # Применяем маску, если предоставлена (для OULAD и др.)
            if next_action_masks is not None:
                # Маски: 1 = допустимо, 0 = запрещено
                # Замещаем запрещенные действия на -inf для исключения из max
                masked_q = next_q_values.clone()
                masked_q[next_action_masks == 0] = -float('inf')
                next_q_max = masked_q.max(1)[0]
                
                # Проверка: если все действия запрещены (всё -inf), используем 0
                next_q_max = torch.where(
                    torch.isinf(next_q_max),
                    torch.zeros_like(next_q_max),
                    next_q_max
                )
            else:
                # Стандартный режим без маски (для ITM-Rec)
                next_q_max = next_q_values.max(1)[0]
            
            # TD targets: r + γ * max Q(s', a') * (1 - done)
            td_targets = rewards + self.gamma * next_q_max * (1 - dones)
        
        return td_targets
    
    def train_step(self, next_action_masks: torch.Tensor = None) -> Optional[float]:
        """
        Один шаг обучения.
        
        Args:
            next_action_masks: Опциональные маски для next_states (для OULAD)
        
        Returns:
            Значение потери или None если буфер недостаточно заполнен
        """
        if len(self.buffer) < self.batch_size:
            return None
        
        # Выборка из буфера
        batch = self.buffer.sample(self.batch_size)
        if batch is None:
            return None
        
        states, actions, rewards, next_states, dones, next_action_masks, indices, weights = batch
        
        # Перемещение на device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        weights = weights.to(self.device)
        if next_action_masks is not None:
            next_action_masks = next_action_masks.to(self.device)
        
        # Текущие Q-значения
        current_q_values = self.agent(states)
        current_q = current_q_values.gather(1, actions.unsqueeze(1)).squeeze()
        
        # Целевые Q-значения (с поддержкой action_mask)
        td_targets = self.compute_td_targets(
            rewards=rewards,
            next_states=next_states,
            dones=dones,
            next_action_masks=next_action_masks  # Если передана, используется для маскирования
        )
        
        # Ошибка TD
        td_errors = td_targets - current_q
        
        # Обновление приоритетов в буфере
        self.buffer.update_priorities(indices, td_errors.cpu().detach().numpy())
        
        # Потеря с учетом importance sampling weights
        per_sample_loss = self.loss_fn(current_q, td_targets)
        loss = (weights * per_sample_loss).mean()
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.agent.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Обновление target network
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.hard_update_target()
        else:
            self.soft_update_target()
        
        return loss.item()
    
    def train_episode(self, max_steps: Optional[int] = None) -> Tuple[float, float]:
        """
        Обучение на одном эпизоде.
        
        Args:
            max_steps: Максимальное количество шагов в эпизоде.
                Если не задан, используется ``config['max_steps_per_episode']``.
        
        Returns:
            Tuple (episode_reward, avg_loss):
            - episode_reward: Суммарная награда за эпизод
            - avg_loss: Средняя потеря за эпизод
        """
        state = self.env.reset()
        episode_reward = 0
        episode_losses = []
        max_steps = int(max_steps or self.max_steps_per_episode)

        for step in range(max_steps):
            # Маска допустимых действий
            mask = None
            if self.use_action_mask and hasattr(self.env, 'get_action_mask'):
                try:
                    mask = self.env.get_action_mask()
                except Exception:
                    mask = None
            
            # Выбор действия
            action = self.agent.get_action(state, self.epsilon, action_mask=mask)
            
            # Шаг в среде
            next_state, reward, done, info = self.env.step(action)
            
            next_mask = None
            if self.use_action_mask and not done and hasattr(self.env, 'get_action_mask'):
                try:
                    next_mask = self.env.get_action_mask()
                except Exception:
                    next_mask = None

            # Сохранение в буфер
            self.buffer.push(
                state,
                action,
                reward,
                next_state,
                done,
                next_action_mask=next_mask,
            )
            
            # Шаг обучения
            loss = self.train_step()
            if loss is not None:
                episode_losses.append(loss)
            
            # Обновление состояния и наград
            state = next_state if not done else self.env.reset()
            episode_reward += reward
            
            # Обновление epsilon
            self.update_epsilon()
            
            if done:
                break
        
        avg_loss = np.mean(episode_losses) if episode_losses else 0
        
        # Сохранение метрик
        self.episode_rewards.append(episode_reward)
        if episode_losses:
            self.losses.append(avg_loss)
        
        return episode_reward, avg_loss
    
    def evaluate(self, n_episodes: int = 10) -> Dict[str, float]:
        """
        Оценка агента.
        
        Args:
            n_episodes: Количество эпизодов для оценки
        
        Returns:
            Словарь с метриками:
            {
                'mean_reward': Средняя награда,
                'std_reward': Стандартное отклонение наград,
                'min_reward': Минимальная награда,
                'max_reward': Максимальная награда
            }
        """
        # Переключаем в режим оценки
        self.agent.eval()
        
        total_rewards = []
        
        for episode in range(n_episodes):
            state = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                mask = None
                if self.use_action_mask and hasattr(self.env, 'get_action_mask'):
                    try:
                        mask = self.env.get_action_mask()
                    except Exception:
                        mask = None
                action = self.agent.get_action(state, epsilon=0.01, action_mask=mask)
                next_state, reward, done, _ = self.env.step(action)
                
                state = next_state
                episode_reward += reward
            
            total_rewards.append(episode_reward)
        
        # Возвращаем в режим обучения
        self.agent.train()
        
        return {
            'mean_reward': np.mean(total_rewards),
            'std_reward': np.std(total_rewards),
            'min_reward': np.min(total_rewards),
            'max_reward': np.max(total_rewards)
        }
    
    def save_checkpoint(self, filepath: str):
        """
        Сохранение чекпоинта тренера.
        
        Args:
            filepath: Путь для сохранения
        """
        checkpoint = {
            'agent_state_dict': self.agent.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'episode_rewards': self.episode_rewards,
            'losses': self.losses,
        }
        torch.save(checkpoint, filepath)
        print(f"Чекпоинт сохранен: {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """
        Загрузка чекпоинта тренера.
        
        Args:
            filepath: Путь к чекпоинту
        """
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.agent.load_state_dict(checkpoint['agent_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_start)
        self.step_count = checkpoint.get('step_count', 0)
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        self.losses = checkpoint.get('losses', [])
        print(f"Чекпоинт загружен: {filepath}")
