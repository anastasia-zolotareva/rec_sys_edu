"""
Конфигурации для обучения моделей.
"""

# Конфигурация обучения DQN
TRAIN_CONFIG = {
    'gamma': 0.99,                    # Коэффициент дисконтирования
    'lr': 0.001,                      # Learning rate
    'tau': 0.01,                      # Коэффициент мягкого обновления target network
    'target_update_freq': 100,        # Частота полного обновления target network
    'batch_size': 64,                 # Размер батча
    'epsilon_start': 1.0,            # Начальное значение epsilon
    'epsilon_end': 0.01,             # Конечное значение epsilon
    'epsilon_decay': 0.995,          # Скорость затухания epsilon
}

# Конфигурация обучения DeepFM+SVD++
DEEPFM_CONFIG = {
    'embedding_dim': 32,             # Размерность эмбеддингов
    'hidden_dims': [128, 64],        # Размерности скрытых слоев Deep части
    'dropout': 0.2,                   # Коэффициент dropout
    'n_epochs': 50,                   # Количество эпох
    'batch_size': 256,                # Размер батча
    'lr': 0.001,                      # Learning rate
    'weight_decay': 1e-5,            # Weight decay для регуляризации
}

# Конфигурация Prioritized Replay Buffer
REPLAY_BUFFER_CONFIG = {
    'capacity': 10000,                # Емкость буфера
    'alpha': 0.6,                     # Степень приоритизации (0 - равномерная выборка)
    'beta': 0.4,                      # Степень коррекции смещения
    'beta_increment': 0.001,          # Шаг увеличения beta
}

# Конфигурация Dueling DQN архитектуры
DUELING_DQN_CONFIG = {
    'state_dim': 65,                  # Размерность состояния
    'action_dim': 70,                 # Размерность пространства действий
    'hidden_dims': [256, 128, 64],    # Размерности скрытых слоев
}
