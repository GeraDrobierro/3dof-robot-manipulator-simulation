"""
Главный файл. Демонстрирует полный цикл работы с роботом:
    1. Создание робота с параметрами
    2. Задание целевой точки
    3. Расчёт кинематики и динамики
    4. Планирование траектории
    5. Визуализация движения
"""

import numpy as np
import matplotlib.pyplot as plt
from robot import Robot
from visualization import (
    draw_3d,
    draw_top_view,
    plot_both_views,
    animate_manipulator,
    plot_workspace
)


def main():
    # ============================================================
    # 1. СОЗДАНИЕ РОБОТА
    # ============================================================

    print("\n" + "-" * 50)
    print("1. СОЗДАНИЕ РОБОТА")
    print("-" * 50)

    # Создаём робота с фиксированными параметрами <<<
    robot = Robot(
        L1=1.0, L2=0.8,  # длины звеньев (м)
        m1=1.0, m2=0.5, m3=0.3,  # массы звеньев (кг)
        I_z2=0.1, I_z3=0.05,  # моменты инерции (кг·м²)
        z_min=0.0, z_max=1.5,  # ограничения высоты (м)
        psi_min=-np.pi, psi_max=np.pi,  # ограничения углов (рад)
        g=9.81  # ускорение свободного падения (м/с²)
    )

    print("Робот создан с параметрами:")
    robot.print_info()

    # ============================================================
    # 2. УСТАНОВКА НАЧАЛЬНОГО СОСТОЯНИЯ
    # ============================================================

    print("\n" + "-" * 50)
    print("2. УСТАНОВКА НАЧАЛЬНОГО СОСТОЯНИЯ")
    print("-" * 50)

    robot.set_state(z=0.3, psi=0.2, xi=0.1)
    print("Начальное состояние:")
    print(robot)

    # Проверяем ограничения
    ok, messages = robot.check_constraints()
    if ok:
        print("Все ограничения соблюдены")
    else:
        print("Нарушены ограничения:")
        for msg in messages:
            print(f"   - {msg}")

    # ============================================================
    # 3. ЗАДАНИЕ ЦЕЛЕВОЙ ТОЧКИ
    # ============================================================

    print("\n" + "-" * 50)
    print("3. ЗАДАНИЕ ЦЕЛЕВОЙ ТОЧКИ")
    print("-" * 50)

    # Задаём целевую точку <<<
    target_x = 1.2
    target_y = 0.8
    target_z = 0.5

    print(f"Целевая точка: ({target_x:.3f}, {target_y:.3f}, {target_z:.3f}) м")

    # Решаем обратную задачу кинематики
    print("\nРешение обратной задачи кинематики...")
    result = robot.inverse_kinematics(target_x, target_y, target_z)

    if result['success']:
        print(" Моделирование корректно, найдены углы:")
        print(f"   z = {result['z']:.3f} м")
        print(f"   ψ = {result['psi']:.3f} рад ({np.degrees(result['psi']):.1f}°)")
        print(f"   ξ = {result['xi']:.3f} рад ({np.degrees(result['xi']):.1f}°)")
        print(f"   Сообщение: {result['message']}")
    else:
        print(f" Не удалось достичь цели")
        print(f"   {result['message']}")
        return

    # Проверяем, что робот теперь в целевой позиции
    print("\nТекущее состояние робота:")
    print(robot)

    # Проверяем, что схват действительно в целевой точке
    x_actual, y_actual, z_actual = robot.forward_kinematics()
    print(f"\nПроверка:")
    print(f"  Цель:     ({target_x:.4f}, {target_y:.4f}, {target_z:.4f})")
    print(f"  Схват:    ({x_actual:.4f}, {y_actual:.4f}, {z_actual:.4f})")
    print(
        f"  Ошибка:   ({abs(target_x - x_actual):.2e}, {abs(target_y - y_actual):.2e}, {abs(target_z - z_actual):.2e})")

    # ============================================================
    # 4. РАСЧЁТ ДИНАМИКИ
    # ============================================================

    print("\n" + "-" * 50)
    print("4. РАСЧЁТ ДИНАМИКИ")
    print("-" * 50)

    # Задаём желаемые ускорения для движения к цели <<<
    ddz = 0.2  # ускорение по высоте (м/с²)
    ddpsi = 0.5  # угловое ускорение ψ (рад/с²)
    ddxi = 0.3  # угловое ускорение ξ (рад/с²)

    print(f"Заданные ускорения:")
    print(f"  ddz  = {ddz:.2f} м/с²")
    print(f"  ddψ  = {ddpsi:.2f} рад/с²")
    print(f"  ddξ  = {ddxi:.2f} рад/с²")

    # Вычисляем требуемые усилия
    dynamics = robot.compute_dynamics(ddz, ddpsi, ddxi)

    print(f"\nТребуемые усилия в приводах:")
    print(f"  F_z  = {dynamics['F_z']:.3f} Н  (сила по высоте)")
    print(f"  M_ψ  = {dynamics['M_psi']:.3f} Н·м (момент для ψ)")
    print(f"  M_ξ  = {dynamics['M_xi']:.3f} Н·м (момент для ξ)")

    if dynamics['constraints_ok']:
        print(" Все ограничения соблюдены")
    else:
        print(" Нарушены ограничения:")
        for msg in dynamics['messages']:
            print(f"   - {msg}")

    # Статические усилия (для удержания позиции)
    static = robot.compute_static_torques()
    print(f"\nСтатические усилия (для удержания позиции):")
    print(f"  F_z  = {static['F_z']:.3f} Н")
    print(f"  M_ψ  = {static['M_psi']:.3f} Н·м")
    print(f"  M_ξ  = {static['M_xi']:.3f} Н·м")

    # ============================================================
    # 5. ПЛАНИРОВАНИЕ ТРАЕКТОРИИ
    # ============================================================

    print("\n" + "-" * 50)
    print("5. ПЛАНИРОВАНИЕ ТРАЕКТОРИИ")
    print("-" * 50)

    # Сбрасываем робота в начальное положение
    robot.set_state(z=0.3, psi=0.0, xi=0.0)
    print(f"Начальное состояние: z={robot.z:.2f}, ψ={robot.psi:.2f}, ξ={robot.xi:.2f}")
    print(f"Целевое состояние:   z={0.5:.2f}, ψ={np.pi / 4:.2f}, ξ={np.pi / 6:.2f}")

    # Планируем траекторию <<<
    n_steps = 30
    trajectory = robot.plan_trajectory(
        target_z=0.5,
        target_psi=np.pi / 4,
        target_xi=np.pi / 6,
        n_steps=n_steps
    )

    print(f"\nСоздана траектория из {len(trajectory)} шагов")

    # Выводим несколько точек траектории
    print("\nТочки траектории (каждые 5 шагов):")
    for i in range(0, len(trajectory), 5):
        z, psi, xi = trajectory[i]
        print(f"  Шаг {i:2d}: z={z:.3f}, ψ={psi:.3f}, ξ={xi:.3f}")

    # ============================================================
    # 6. ВИЗУАЛИЗАЦИЯ
    # ============================================================

    print("\n" + "-" * 50)
    print("6. ВИЗУАЛИЗАЦИЯ")
    print("-" * 50)

    # Устанавливаем состояние для визуализации
    robot.set_state(z=0.5, psi=np.pi / 4, xi=np.pi / 6)

    # 6.1. 3D вид и вид сверху
    print("\n6.1. Отображение 3D вида и вида сверху...")
    print("     (Закройте окно графика, чтобы продолжить)")
    plot_both_views(robot)
    plt.show()

    # 6.2. Рабочая зона
    print("\n6.2. Построение рабочей зоны...")
    print("     (Закройте окно графика, чтобы продолжить)")
    plot_workspace(robot, resolution=50)
    plt.show()

    # 6.3. Анимация движения
    print("\n6.3. Анимация движения по траектории...")
    print("     (Закройте окно анимации, чтобы продолжить)")
    print("     Создаём траекторию для анимации...")

    # Создаём красивую траекторию для анимации
    anim_trajectory = []
    n_frames = 50
    for t in np.linspace(0, 2 * np.pi, n_frames):
        z = 0.3 + 0.2 * (1 + np.sin(t)) / 2
        psi = 0.2 + 0.6 * (1 + np.sin(t)) / 2
        xi = 0.1 + 0.5 * (1 + np.sin(t / 1.5)) / 2
        anim_trajectory.append((z, psi, xi))

    print(f"   Создана анимация из {len(anim_trajectory)} кадров")
    print("   Запуск анимации...")
    animate_manipulator(robot, anim_trajectory, interval=50)

    # ============================================================
    # 7. ИТОГИ
    # ============================================================

    print("\n" + "=" * 70)
    print("РЕЗУЛЬТАТЫ МОДЕЛИРОВАНИЯ")
    print("=" * 70)

    print("\n1. Кинематика:")
    print(f"   - Длины звеньев: L1 = {robot.L1} м, L2 = {robot.L2} м")
    print(f"   - Начальное состояние: z={0.3:.2f}, ψ={0.2:.2f}, ξ={0.1:.2f}")
    print(f"   - Целевая точка: ({target_x:.2f}, {target_y:.2f}, {target_z:.2f}) м")
    print(f"   - Найденные углы: ψ={result['psi']:.3f}, ξ={result['xi']:.3f}")
    print(f"   - Ошибка позиционирования: {abs(target_x - x_actual):.2e} м")

    print("\n2. Динамика:")
    print(f"   - Массы: m1={robot.m1} кг, m2={robot.m2} кг, m3={robot.m3} кг")
    print(f"   - Моменты инерции: I_z2={robot.I_z2} кг·м², I_z3={robot.I_z3} кг·м²")
    print(f"   - Требуемая сила F_z = {dynamics['F_z']:.3f} Н")
    print(f"   - Требуемый момент M_ψ = {dynamics['M_psi']:.3f} Н·м")
    print(f"   - Требуемый момент M_ξ = {dynamics['M_xi']:.3f} Н·м")

    print("\n3. Ограничения:")
    if dynamics['constraints_ok']:
        print("   Все динамические ограничения соблюдены")
    else:
        print("   Обнаружены нарушения ограничений")

    print("\n4. Визуализация:")
    print("   3D модель построена")
    print("   Вид сверху построен")
    print("   Рабочая зона построена")
    print("   Анимация движения создана")

    print("\n" + "=" * 70)
    print("МОДЕЛИРОВАНИЕ ЗАВЕРШЕНО")
    print("=" * 70)


if __name__ == "__main__":
    main()