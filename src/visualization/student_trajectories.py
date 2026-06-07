"""
Визуализация образовательных траекторий для конкретных студентов.

Модуль предоставляет инструменты для:
- Выбора репрезентативных студентов из датасета
- Симуляции индивидуальных траекторий рекомендаций
- Форматирования и визуализации результатов для презентаций
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from ..data.schemas import DatasetBundle
from ..environment.oulad_env import OULADEnvironment
from ..environment.educational_env import EducationalEnvironment
from ..models.dueling_dqn import DuelingDQN
from ..models.deepfm_svdpp import DeepFMSVDPlusPlus

logger = logging.getLogger("rec_sys_edu")


class StudentTrajectoryAnalyzer:
    """Анализатор образовательных траекторий для студентов."""

    def __init__(
        self,
        bundle: DatasetBundle,
        deepfm_model: DeepFMSVDPlusPlus,
        dqn_model: DuelingDQN,
        env: OULADEnvironment | EducationalEnvironment,
        device: torch.device = torch.device("cpu"),
        config: Optional[Mapping[str, Any]] = None,
    ):
        """
        Инициализация анализатора.

        Parameters
        ----------
        bundle : DatasetBundle
            Подготовленный датасет
        deepfm_model : DeepFMSVDPlusPlus
            Обученная модель для предсказания критериев
        dqn_model : DuelingDQN
            Обученный DQN агент
        env : OULADEnvironment | EducationalEnvironment
            Окружение для симуляции траекторий
        device : torch.device
            Устройство (CPU/GPU)
        config : Optional[Mapping[str, Any]]
            Конфигурация окружения
        """
        self.bundle = bundle
        self.deepfm_model = deepfm_model
        self.dqn_model = dqn_model
        self.env = env
        self.device = device
        self.config = dict(config or {})
        self.dataset_type = bundle.dataset_type

    def select_representative_students(
        self,
        n_students: int = 5,
        stratify_by: str = "proxy_values",
    ) -> List[Dict[str, Any]]:
        """
        Выбирает репрезентативных студентов с разными характеристиками.

        Parameters
        ----------
        n_students : int
            Количество студентов для выбора
        stratify_by : str
            Критерий стратификации ('proxy_values', 'random')

        Returns
        -------
        List[Dict[str, Any]]
            Список со словарями информации о студентах
        """
        if self.dataset_type == "oulad":
            return self._select_oulad_students(n_students, stratify_by)
        else:  # itmrec
            return self._select_itmrec_students(n_students, stratify_by)

    def _select_oulad_students(
        self,
        n_students: int,
        stratify_by: str,
    ) -> List[Dict[str, Any]]:
        """Выбор репрезентативных студентов для OULAD."""
        student_features = self.bundle.metadata.get("student_features")
        if student_features is None or len(student_features) == 0:
            logger.warning("student_features не найден, выбираем случайных студентов")
            student_features = self.bundle.users.copy()

        # Используем прокси-критерии для стратификации
        if stratify_by == "proxy_values" and "Mastery" in student_features.columns:
            features_copy = student_features.copy()
            features_copy["score"] = (
                0.3 * features_copy.get("Mastery", 0.5)
                + 0.3 * features_copy.get("Engagement", 0.5)
                + 0.2 * features_copy.get("SelfRegulation", 0.5)
                + 0.2 * features_copy.get("Outcome", 0.5)
            )
        else:
            # Случайный выбор
            features_copy = student_features.copy()
            features_copy["score"] = np.random.rand(len(features_copy))

        # Стратифицированный выбор (разные уровни производительности)
        n_bins = min(n_students, 5)
        features_copy["bin"] = pd.qcut(features_copy["score"], q=n_bins, labels=False, duplicates="drop")
        
        selected = []
        for bin_val in sorted(features_copy["bin"].unique()):
            bin_students = features_copy[features_copy["bin"] == bin_val]
            if len(bin_students) > 0:
                # Выбираем студента ближе к центру бина
                idx = bin_students["score"].argmax() if bin_val == bin_students["bin"].max() else bin_students["score"].idxmin()
                student_row = student_features.loc[idx]
                selected.append({
                    "student_idx": idx,
                    "user_id": int(student_row.get("id_student", idx)),
                    "user_id_encoded": int(student_row.get("UserID_encoded", idx)),
                    "module": str(student_row.get("code_module", "Unknown")),
                    "presentation": str(student_row.get("code_presentation", "Unknown")),
                    "gender": str(student_row.get("gender", "Unknown")),
                    "age_band": str(student_row.get("age_band", "Unknown")),
                    "mastery": float(student_row.get("Mastery", 0.5)),
                    "engagement": float(student_row.get("Engagement", 0.5)),
                    "selfregulation": float(student_row.get("SelfRegulation", 0.5)),
                    "outcome": float(student_row.get("Outcome", 0.5)),
                    "final_result": str(student_row.get("final_result", "Unknown")),
                })
        
        return selected[:n_students]

    def _select_itmrec_students(
        self,
        n_students: int,
        stratify_by: str,
    ) -> List[Dict[str, Any]]:
        """Выбор репрезентативных студентов для ITM-Rec."""
        users = self.bundle.users.copy()
        ratings = self.bundle.ratings.copy()

        # Вычисляем среднюю оценку по пользователю
        avg_ratings = ratings.groupby("UserID_encoded")[self.bundle.target_columns].mean()
        users = users.merge(avg_ratings, left_on="UserID_encoded", right_index=True, how="left")

        if stratify_by == "proxy_values":
            users["score"] = users[self.bundle.target_columns].mean(axis=1)
        else:
            users["score"] = np.random.rand(len(users))

        # Стратифицированный выбор
        n_bins = min(n_students, 5)
        users["bin"] = pd.qcut(users["score"], q=n_bins, labels=False, duplicates="drop")

        selected = []
        for bin_val in sorted(users["bin"].unique()):
            bin_users = users[users["bin"] == bin_val]
            if len(bin_users) > 0:
                idx = bin_users.index[len(bin_users) // 2]  # Середина бина
                user_row = users.loc[idx]
                selected.append({
                    "student_idx": idx,
                    "user_id": int(user_row.get("UserID", idx)),
                    "user_id_encoded": int(user_row.get("UserID_encoded", idx)),
                    "class": str(user_row.get("Class", "Unknown")),
                    "semester": str(user_row.get("Semester", "Unknown")),
                    "rating_avg": float(user_row.get("Rating", 0.5)),
                    "app_avg": float(user_row.get("App", 0.5)),
                    "data_avg": float(user_row.get("Data", 0.5)),
                    "ease_avg": float(user_row.get("Ease", 0.5)),
                })

        return selected[:n_students]

    def simulate_student_trajectory(
        self,
        student_info: Dict[str, Any],
        n_steps: int = 10,
    ) -> Dict[str, Any]:
        """
        Симулирует траекторию одного студента.

        Parameters
        ----------
        student_info : Dict[str, Any]
            Информация о студенте (из select_representative_students)
        n_steps : int
            Количество шагов в траектории

        Returns
        -------
        Dict[str, Any]
            Словарь с траекторией и метриками на каждом шаге
        """
        user_id_encoded = student_info["user_id_encoded"]
        
        # Инициализируем окружение
        state = self.env.reset(user_id=user_id_encoded)
        
        trajectory = {
            "student_info": student_info,
            "steps": [],
            "cumulative_reward": 0.0,
        }
        
        for step_idx in range(n_steps):
            # Получаем action mask если доступен
            action_mask = None
            if hasattr(self.env, "get_action_mask"):
                action_mask = self.env.get_action_mask()

            # Выбираем действие через DQN (используем метод get_action с numpy array)
            action = self.dqn_model.get_action(state, action_mask=action_mask, epsilon=0.0)

            # Выполняем шаг в окружении
            next_state, reward, done, info = self.env.step(action)

            # Собираем информацию о шаге
            step_data = {
                "step": step_idx + 1,
                "action": action,
                "reward": float(reward),
                "cumulative_reward": float(trajectory["cumulative_reward"] + reward),
            }

            # Добавляем специфичную информацию для OULAD
            if self.dataset_type == "oulad" and hasattr(self.env, "current_proxy"):
                step_data.update({
                    "mastery": float(self.env.current_proxy.get("mastery", 0)),
                    "engagement": float(self.env.current_proxy.get("engagement", 0)),
                    "selfregulation": float(self.env.current_proxy.get("selfregulation", 0)),
                    "outcome": float(self.env.current_proxy.get("outcome", 0)),
                })

            # Пытаемся получить информацию о рекомендуемом предмете/действии
            if "item_name" in info:
                step_data["item_name"] = info["item_name"]
            elif action < len(self.bundle.items):
                item_row = self.bundle.items.iloc[action]
                if "item_id" in item_row.index:
                    step_data["item_name"] = str(item_row["item_id"])

            trajectory["steps"].append(step_data)
            trajectory["cumulative_reward"] += reward

            state = next_state

            if done:
                logger.info(f"Траектория завершена на шаге {step_idx + 1}")
                break

        return trajectory

    def simulate_multiple_trajectories(
        self,
        students: List[Dict[str, Any]],
        n_steps: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Симулирует траектории для нескольких студентов.

        Parameters
        ----------
        students : List[Dict[str, Any]]
            Список информации о студентах
        n_steps : int
            Количество шагов в каждой траектории

        Returns
        -------
        List[Dict[str, Any]]
            Список траекторий
        """
        trajectories = []
        for student in students:
            logger.info(f"Симулирую траекторию для студента {student.get('user_id_encoded')}")
            traj = self.simulate_student_trajectory(student, n_steps=n_steps)
            trajectories.append(traj)
        return trajectories

    def trajectories_to_dataframe(
        self,
        trajectories: List[Dict[str, Any]],
    ) -> pd.DataFrame:
        """
        Преобразует траектории в DataFrame для экспорта/анализа.

        Parameters
        ----------
        trajectories : List[Dict[str, Any]]
            Список траекторий из simulate_student_trajectory

        Returns
        -------
        pd.DataFrame
            Объединенный DataFrame с данными траекторий
        """
        rows = []
        for traj in trajectories:
            student_info = traj["student_info"]
            for step in traj["steps"]:
                row = {
                    "student_id": student_info.get("user_id"),
                    "student_id_encoded": student_info.get("user_id_encoded"),
                }
                row.update(student_info)
                row.update(step)
                rows.append(row)
        
        return pd.DataFrame(rows)

    def plot_student_profile_card(
        self,
        student_info: Dict[str, Any],
        trajectory: Dict[str, Any],
        save_path: Optional[Path] = None,
        show: bool = False,
    ) -> Path | None:
        """
        Рисует карточку профиля студента и его траекторию.

        Parameters
        ----------
        student_info : Dict[str, Any]
            Информация о студенте
        trajectory : Dict[str, Any]
            Симулированная траектория
        save_path : Optional[Path]
            Путь для сохранения
        show : bool
            Показать ли график

        Returns
        -------
        Path | None
            Путь к сохраненному файлу
        """
        fig = plt.figure(figsize=(14, 8))
        gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

        # Профиль студента (текст)
        ax_profile = fig.add_subplot(gs[0, 0])
        ax_profile.axis("off")
        
        profile_text = "ПРОФИЛЬ СТУДЕНТА\n" + "=" * 40 + "\n"
        profile_text += f"ID: {student_info.get('user_id_encoded')}\n"
        
        if self.dataset_type == "oulad":
            profile_text += f"Модуль: {student_info.get('module')}\n"
            profile_text += f"Период: {student_info.get('presentation')}\n"
            profile_text += f"Пол: {student_info.get('gender')}\n"
            profile_text += f"Возраст: {student_info.get('age_band')}\n"
            profile_text += f"Результат: {student_info.get('final_result')}\n"
        else:
            profile_text += f"Класс: {student_info.get('class')}\n"
            profile_text += f"Семестр: {student_info.get('semester')}\n"

        ax_profile.text(0.1, 0.5, profile_text, fontsize=10, family="monospace",
                       verticalalignment="center", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

        # Исходные метрики студента (столбцовая диаграмма)
        ax_metrics = fig.add_subplot(gs[0, 1])
        
        if self.dataset_type == "oulad":
            metrics = {
                "Mastery": student_info.get("mastery", 0),
                "Engagement": student_info.get("engagement", 0),
                "SelfReg": student_info.get("selfregulation", 0),
                "Outcome": student_info.get("outcome", 0),
            }
        else:
            metrics = {
                "Rating": student_info.get("rating_avg", 0),
                "App": student_info.get("app_avg", 0),
                "Data": student_info.get("data_avg", 0),
                "Ease": student_info.get("ease_avg", 0),
            }

        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        bars = ax_metrics.bar(range(len(metrics)), list(metrics.values()), color=colors, alpha=0.7)
        ax_metrics.set_xticks(range(len(metrics)))
        ax_metrics.set_xticklabels(list(metrics.keys()), rotation=45, ha="right")
        ax_metrics.set_ylabel("Значение")
        ax_metrics.set_title("Исходные характеристики")
        ax_metrics.set_ylim(0, 1.0)
        ax_metrics.grid(True, alpha=0.3, axis="y")
        
        # Добавляем значения на столбцы
        for bar in bars:
            height = bar.get_height()
            ax_metrics.text(bar.get_x() + bar.get_width() / 2., height,
                           f"{height:.2f}", ha="center", va="bottom", fontsize=9)

        # Награда по шагам
        ax_reward = fig.add_subplot(gs[1, 0])
        steps = [s["step"] for s in trajectory["steps"]]
        rewards = [s["reward"] for s in trajectory["steps"]]
        cum_rewards = [s["cumulative_reward"] for s in trajectory["steps"]]
        
        ax_reward.plot(steps, rewards, marker="o", label="Награда на шаге", alpha=0.7)
        ax_reward.set_xlabel("Шаг траектории")
        ax_reward.set_ylabel("Награда")
        ax_reward.set_title("Распределение наград")
        ax_reward.grid(True, alpha=0.3)
        ax_reward.legend()

        # Кумулятивная награда
        ax_cum = fig.add_subplot(gs[1, 1])
        ax_cum.plot(steps, cum_rewards, marker="s", color="green", label="Кумулятивная награда", linewidth=2)
        ax_cum.fill_between(steps, cum_rewards, alpha=0.3, color="green")
        ax_cum.set_xlabel("Шаг траектории")
        ax_cum.set_ylabel("Кумулятивная награда")
        ax_cum.set_title("Накопленная награда")
        ax_cum.grid(True, alpha=0.3)
        ax_cum.legend()

        plt.suptitle(f"Траектория студента {student_info.get('user_id_encoded')}", fontsize=14, fontweight="bold")

        path = None
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.tight_layout()
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            path = save_path
            logger.info(f"График сохранен в {save_path}")

        if show:
            plt.show()
        plt.close(fig)
        return path

    def plot_metrics_evolution(
        self,
        trajectories: List[Dict[str, Any]],
        save_path: Optional[Path] = None,
        show: bool = False,
    ) -> Path | None:
        """
        Рисует динамику метрик студентов за время траекторий.

        Parameters
        ----------
        trajectories : List[Dict[str, Any]]
            Список траекторий
        save_path : Optional[Path]
            Путь для сохранения
        show : bool
            Показать ли график

        Returns
        -------
        Path | None
            Путь к сохраненному файлу
        """
        if self.dataset_type != "oulad":
            logger.warning("plot_metrics_evolution поддерживается только для OULAD")
            return None

        # Проверяем наличие метрик
        has_metrics = all(
            "mastery" in traj["steps"][0] if traj["steps"] else False
            for traj in trajectories
        )

        if not has_metrics:
            logger.warning("Метрики в траекториях не найдены")
            return None

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        
        metric_names = ["mastery", "engagement", "selfregulation", "outcome"]
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

        for idx, metric_name in enumerate(metric_names):
            ax = axes[idx]
            
            for traj_idx, traj in enumerate(trajectories):
                steps = [s["step"] for s in traj["steps"]]
                values = [s.get(metric_name, 0) for s in traj["steps"]]
                student_id = traj["student_info"].get("user_id_encoded")
                ax.plot(steps, values, marker="o", label=f"Студент {student_id}", alpha=0.7)
            
            ax.set_xlabel("Шаг траектории")
            ax.set_ylabel("Значение")
            ax.set_title(f"Динамика {metric_name.capitalize()}")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            ax.set_ylim(0, 1.0)

        plt.suptitle("Эволюция метрик студентов за время траекторий", fontsize=14, fontweight="bold")
        fig.tight_layout()

        path = None
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            path = save_path
            logger.info(f"График сохранен в {save_path}")

        if show:
            plt.show()
        plt.close(fig)
        return path

    def export_summary_table(
        self,
        trajectories: List[Dict[str, Any]],
        save_path: Optional[Path] = None,
    ) -> pd.DataFrame:
        """
        Экспортирует сводную таблицу траекторий.

        Parameters
        ----------
        trajectories : List[Dict[str, Any]]
            Список траекторий
        save_path : Optional[Path]
            Путь для сохранения CSV

        Returns
        -------
        pd.DataFrame
            Сводная таблица
        """
        summary_rows = []
        
        for traj in trajectories:
            student = traj["student_info"]
            steps = traj["steps"]
            
            row = {
                "student_id_encoded": student.get("user_id_encoded"),
                "n_steps": len(steps),
                "total_reward": traj["cumulative_reward"],
                "avg_reward_per_step": traj["cumulative_reward"] / len(steps) if steps else 0,
            }

            if self.dataset_type == "oulad":
                # Начальные значения
                row.update({
                    "mastery_init": student.get("mastery", 0),
                    "engagement_init": student.get("engagement", 0),
                    "selfregulation_init": student.get("selfregulation", 0),
                    "outcome_init": student.get("outcome", 0),
                })
                # Финальные значения
                if steps:
                    last_step = steps[-1]
                    row.update({
                        "mastery_final": last_step.get("mastery", 0),
                        "engagement_final": last_step.get("engagement", 0),
                        "selfregulation_final": last_step.get("selfregulation", 0),
                        "outcome_final": last_step.get("outcome", 0),
                    })
                    # Изменения
                    row.update({
                        "delta_mastery": last_step.get("mastery", 0) - student.get("mastery", 0),
                        "delta_engagement": last_step.get("engagement", 0) - student.get("engagement", 0),
                        "delta_selfregulation": last_step.get("selfregulation", 0) - student.get("selfregulation", 0),
                        "delta_outcome": last_step.get("outcome", 0) - student.get("outcome", 0),
                    })

            summary_rows.append(row)

        summary_df = pd.DataFrame(summary_rows)

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            summary_df.to_csv(save_path, index=False, encoding="utf-8-sig")
            logger.info(f"Сводная таблица сохранена в {save_path}")

        return summary_df
