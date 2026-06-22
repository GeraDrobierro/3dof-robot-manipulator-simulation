"""
Модуль визуализации для трёхзвенного робота-манипулятора.

Реализует два вида отображения:
    1. 3D вид — полная модель в пространстве
    2. Вид сверху — проекция на плоскость Oxy на высоте z_0

Использует: matplotlib, numpy
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation


# ============================================================
# 1. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

def _get_axis_limits(points, margin=0.5):
    """
    Вычисляет пределы осей для равномерного масштабирования.

    Параметры:
        points (np.ndarray): массив точек (N, 3) или (N, 2)
        margin (float): отступ от крайних точек

    Возвращает:
        tuple: (min_val, max_val) для всех осей
    """
    if points.shape[1] == 3:
        # 3D точки
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        all_points = np.concatenate([x, y, z])
    else:
        # 2D точки
        x = points[:, 0]
        y = points[:, 1]
        all_points = np.concatenate([x, y])

    min_val = np.min(all_points) - margin
    max_val = np.max(all_points) + margin

    return min_val, max_val


# ============================================================
# 2. 3D ВИД МАНИПУЛЯТОРА
# ============================================================

def draw_3d(robot, ax=None, show_projection=True, show_axes=True,
            joint_labels=True, title="3D модель манипулятора"):
    """
    Рисует 3D модель манипулятора.

    Параметры:
        robot: экземпляр ManipulatorModel (или объект с методом get_joint_positions)
        ax: оси для рисования (если None, создаются новые)
        show_projection (bool): показывать проекцию схвата на плоскость
        show_axes (bool): показывать подписи осей
        joint_labels (bool): показывать подписи сочленений
        title (str): заголовок графика

    Возвращает:
        ax: оси с нарисованной моделью
    """
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

    # Получаем координаты сочленений
    joints = robot.get_joint_positions()
    points = np.array(joints)

    # Рисуем звенья (линии)
    ax.plot(points[:, 0], points[:, 1], points[:, 2],
            'b-o', linewidth=3, markersize=8,
            label='Звенья', zorder=3)

    # Рисуем сочленения (красные точки)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               color='red', s=80, zorder=5, label='Сочленения')

    # Подписываем точки
    if joint_labels:
        labels = ['O (основание)', 'A (звено 1)', 'B (звено 2)', 'C (схват)']
        for i, (x, y, z) in enumerate(points):
            ax.text(x, y, z, f'  {labels[i]}', fontsize=10, zorder=6)

    # Рисуем проекцию схвата на плоскость (пунктирная линия)
    if show_projection:
        p_end = points[-1]
        # Вертикальная линия от схвата до плоскости
        ax.plot([p_end[0], p_end[0]], [p_end[1], p_end[1]], [0, p_end[2]],
                'k--', linewidth=1, alpha=0.5, label='Проекция')

        # Точка проекции на плоскости
        ax.scatter(p_end[0], p_end[1], 0, color='green', s=60,
                   marker='s', zorder=4, label='Проекция схвата')

        # Горизонтальная линия от основания до проекции
        ax.plot([0, p_end[0]], [0, p_end[1]], [0, 0],
                'g--', linewidth=1, alpha=0.3)

    # Настройка осей
    if show_axes:
        ax.set_xlabel('X (м)')
        ax.set_ylabel('Y (м)')
        ax.set_zlabel('Z (м)')

    ax.set_title(title)
    ax.legend(loc='upper right')

    # Делаем равный масштаб по всем осям
    min_val, max_val = _get_axis_limits(points, margin=0.3)
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_zlim(0, max_val + 0.3)

    # Настройка угла обзора
    ax.view_init(elev=25, azim=-60)

    return ax


# ============================================================
# 3. ВИД СВЕРХУ (ПРОЕКЦИЯ НА ПЛОСКОСТЬ Oxy)
# ============================================================

def draw_top_view(robot, ax=None, show_workspace=True,
                  show_joint_labels=True, title="Вид сверху (проекция на Oxy)"):
    """
    Рисует вид сверху манипулятора (проекция на плоскость Oxy).

    Параметры:
        robot: экземпляр ManipulatorModel
        ax: оси для рисования (если None, создаются новые)
        show_workspace (bool): показывать границы рабочей зоны
        show_joint_labels (bool): показывать подписи сочленений
        title (str): заголовок графика

    Возвращает:
        ax: оси с нарисованной моделью
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    # Получаем координаты сочленений
    joints = robot.get_joint_positions()
    points = np.array(joints)

    # Берём только x и y координаты (проекция)
    points_xy = points[:, :2]

    # Рисуем звенья
    ax.plot(points_xy[:, 0], points_xy[:, 1],
            'b-o', linewidth=3, markersize=8, label='Звенья')

    # Рисуем сочленения
    ax.scatter(points_xy[:, 0], points_xy[:, 1],
               color='red', s=80, zorder=5, label='Сочленения')

    # Подписываем точки
    if show_joint_labels:
        labels = ['O', 'A', 'B', 'C']
        for i, (x, y) in enumerate(points_xy):
            ax.text(x, y, f'  {labels[i]}', fontsize=12, fontweight='bold')

    # Показываем границы рабочей зоны
    if show_workspace:
        from kinematics import get_workspace_boundary
        inner, outer = get_workspace_boundary(robot.L1, robot.L2)

        ax.plot(outer[:, 0], outer[:, 1], 'r--',
                linewidth=1.5, alpha=0.7, label='Макс. достижимость')
        ax.plot(inner[:, 0], inner[:, 1], 'r--',
                linewidth=1.5, alpha=0.7, label='Мин. достижимость')

    # Настройка осей
    ax.set_xlabel('X (м)')
    ax.set_ylabel('Y (м)')
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

    # Делаем равный масштаб
    min_val, max_val = _get_axis_limits(points_xy, margin=0.3)
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)

    return ax


# ============================================================
# 4. СОВМЕСТНЫЙ ГРАФИК (3D + ВИД СВЕРХУ)
# ============================================================

def plot_both_views(robot, figsize=(14, 6)):
    """
    Создаёт два графика рядом: 3D вид и вид сверху.

    Параметры:
        robot: экземпляр ManipulatorModel
        figsize (tuple): размер фигуры

    Возвращает:
        tuple: (fig, ax1, ax2)
    """
    fig = plt.figure(figsize=figsize)

    # 3D вид
    ax1 = fig.add_subplot(121, projection='3d')
    draw_3d(robot, ax1)

    # Вид сверху
    ax2 = fig.add_subplot(122)
    draw_top_view(robot, ax2)

    plt.tight_layout()

    return fig, ax1, ax2


# ============================================================
# 5. АНИМАЦИЯ ДВИЖЕНИЯ
# ============================================================

def animate_manipulator(robot, trajectory, interval=100,
                        save_path=None, show=True):
    """
    Создаёт анимацию движения манипулятора по траектории.

    Параметры:
        robot: экземпляр ManipulatorModel
        trajectory (list): список состояний [(z, psi, xi), ...]
        interval (int): интервал между кадрами (мс)
        save_path (str): путь для сохранения (опционально)
        show (bool): показывать анимацию

    Возвращает:
        animation: объект FuncAnimation
    """
    fig = plt.figure(figsize=(12, 6))

    # 3D вид
    ax1 = fig.add_subplot(121, projection='3d')
    # Вид сверху
    ax2 = fig.add_subplot(122)

    def update(frame):
        # Очищаем оси
        ax1.clear()
        ax2.clear()

        # Устанавливаем состояние робота
        z, psi, xi = trajectory[frame]
        robot.set_state(z, psi, xi)

        # Рисуем 3D вид
        draw_3d(robot, ax1, title=f'3D вид (кадр {frame + 1}/{len(trajectory)})')

        # Рисуем вид сверху
        draw_top_view(robot, ax2, title=f'Вид сверху (кадр {frame + 1}/{len(trajectory)})')

        return ax1, ax2

    anim = FuncAnimation(fig, update, frames=len(trajectory),
                         interval=interval, blit=False)

    if save_path:
        anim.save(save_path, writer='pillow', fps=20)
        print(f"Анимация сохранена в {save_path}")

    if show:
        plt.show()

    return anim


# ============================================================
# 6. ПОСТРОЕНИЕ РАБОЧЕЙ ЗОНЫ
# ============================================================

def plot_workspace(robot, resolution=100, figsize=(8, 8)):
    """
    Строит рабочую зону манипулятора на плоскости Oxy.

    Параметры:
        robot: экземпляр ManipulatorModel
        resolution (int): разрешение сетки
        figsize (tuple): размер фигуры

    Возвращает:
        tuple: (fig, ax)
    """
    from kinematics import forward_kinematics, get_workspace_boundary

    fig, ax = plt.subplots(figsize=figsize)

    # Генерируем точки рабочей зоны
    psi_values = np.linspace(-np.pi, np.pi, resolution)
    xi_values = np.linspace(-np.pi, np.pi, resolution)

    points_x = []
    points_y = []

    for psi in psi_values:
        for xi in xi_values:
            x, y = forward_kinematics(robot.L1, robot.L2, psi, xi)
            points_x.append(x)
            points_y.append(y)

    # Рисуем точки
    ax.scatter(points_x, points_y, s=1, alpha=0.3, c='blue', label='Рабочая зона')

    # Рисуем границы
    inner, outer = get_workspace_boundary(robot.L1, robot.L2)
    ax.plot(outer[:, 0], outer[:, 1], 'r-', linewidth=2, label='Макс. граница')
    ax.plot(inner[:, 0], inner[:, 1], 'r-', linewidth=2, label='Мин. граница')

    # Рисуем пример конфигурации
    x, y, _ = robot.forward_kinematics()
    ax.scatter(x, y, color='green', s=100, marker='*', label='Текущий схват')

    # Настройка
    ax.set_xlabel('X (м)')
    ax.set_ylabel('Y (м)')
    ax.set_title('Рабочая зона манипулятора в плоскости Oxy')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Ограничения
    margin = 0.3
    ax.set_xlim(-robot.L1 - robot.L2 - margin, robot.L1 + robot.L2 + margin)
    ax.set_ylim(-robot.L1 - robot.L2 - margin, robot.L1 + robot.L2 + margin)

    return fig, ax