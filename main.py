"""
Главный файл. Демонстрирует полный цикл работы с роботом.
Пользователь задаёт целевую точку (x, y, z)
"""

import numpy as np
import matplotlib.pyplot as plt
from robot import Robot
from visualization import draw_3d, draw_top_view, animate_manipulator, plot_workspace


def main():
    # ============================================================
    # 1. СОЗДАНИЕ РОБОТА
    # ============================================================
    robot = Robot(
        L1=1.0, L2=0.8,
        m1=1.0, m2=0.5, m3=0.3,
        I_z2=0.1, I_z3=0.05,
        z_min=0.0, z_max=1.5,
        psi_min=-np.pi, psi_max=np.pi,
        g=9.81
    )

    print("Робот создан")
    robot.print_info()

    # ============================================================
    # 2. УСТАНОВКА НАЧАЛЬНОГО СОСТОЯНИЯ
    # ============================================================
    robot.set_state(z=0.3, psi=0.2, xi=0.1)
    print("\nНачальное состояние:")
    print(robot)

    # ============================================================
    # 3. ПОЛЬЗОВАТЕЛЬ ЗАДАЁТ ТОЧКУ
    # ============================================================
    print("\n" + "-" * 50)
    print("ПОЛЬЗОВАТЕЛЬ ЗАДАЁТ ЦЕЛЕВУЮ ТОЧКУ")
    print("-" * 50)

    # Целевая точка (соответствует углам psi=45°, xi=30°)
    target_x = 0.9142
    target_y = 1.4798
    target_z = 0.5

    print(f"Целевая точка: ({target_x:.4f}, {target_y:.4f}, {target_z:.2f}) м")

    # ============================================================
    # 4. ОБРАТНАЯ КИНЕМАТИКА (АВТОМАТИЧЕСКИ)
    # ============================================================
    print("\n--- ВЫЧИСЛЕНИЕ УГЛОВ (ОБРАТНАЯ КИНЕМАТИКА) ---")
    result = robot.inverse_kinematics(target_x, target_y, target_z)

    if not result['success']:
        print(f"Ошибка: {result['message']}")
        return

    target_psi = result['psi']
    target_xi = result['xi']

    print(f"Найденные углы:")
    print(f"  ψ = {target_psi:.4f} рад ({np.degrees(target_psi):.1f}°)")
    print(f"  ξ = {target_xi:.4f} рад ({np.degrees(target_xi):.1f}°)")

    # Проверяем точность
    x_actual, y_actual, z_actual = robot.forward_kinematics()
    print(f"\nОшибка позиционирования:")
    print(f"  Δx = {abs(target_x - x_actual):.2e} м")
    print(f"  Δy = {abs(target_y - y_actual):.2e} м")
    print(f"  Δz = {abs(target_z - z_actual):.2e} м")

    # ============================================================
    # 5. ПЛАНИРОВАНИЕ ТРАЕКТОРИИ
    # ============================================================
    print("\n--- ПЛАНИРОВАНИЕ ТРАЕКТОРИИ ---")

    # Сбрасываем в начальное состояние для анимации
    robot.set_state(z=0.3, psi=0.0, xi=0.0)

    n_steps = 100
    trajectory = robot.plan_trajectory(
        target_z=target_z,
        target_psi=target_psi,
        target_xi=target_xi,
        n_steps=n_steps
    )

    print(f"Создана траектория из {len(trajectory)} шагов")
    print(f"От: z={0.3:.2f}, ψ={0.0:.2f}, ξ={0.0:.2f}")
    print(f"До:  z={target_z:.2f}, ψ={target_psi:.4f}, ξ={target_xi:.4f}")

    # ============================================================
    # 6. ДИНАМИКА
    # ============================================================
    print("\n--- РАСЧЁТ ДИНАМИКИ ---")

    dynamics = robot.compute_dynamics(ddz=0.2, ddpsi=0.5, ddxi=0.3)

    print(f"Требуемые усилия:")
    print(f"  F_z  = {dynamics['F_z']:.3f} Н")
    print(f"  M_ψ  = {dynamics['M_psi']:.3f} Н·м")
    print(f"  M_ξ  = {dynamics['M_xi']:.3f} Н·м")

    if dynamics['constraints_ok']:
        print("Все ограничения соблюдены")
    else:
        print("Нарушены ограничения:")
        for msg in dynamics['messages']:
            print(f"   - {msg}")

    # ============================================================
    # 7. ВИЗУАЛИЗАЦИЯ
    # ============================================================
    print("\n--- ВИЗУАЛИЗАЦИЯ ---")

    # Устанавливаем целевое состояние для визуализации
    robot.set_state(z=target_z, psi=target_psi, xi=target_xi)

    # 7.1. 3D вид
    print("\n7.1. 3D вид с целевой точкой...")
    ax1 = draw_3d(robot, title="3D модель манипулятора (достигнута цель)")
    ax1.scatter(target_x, target_y, target_z,
                color='green', s=150, marker='*', zorder=10,
                label=f'Целевая точка ({target_x:.2f}, {target_y:.2f}, {target_z:.2f})')
    ax1.legend()
    plt.show()

    # 7.2. Вид сверху
    print("\n7.2. Вид сверху с целевой точкой...")
    ax2 = draw_top_view(robot, title="Вид сверху (проекция на Oxy)")
    ax2.scatter(target_x, target_y,
                color='green', s=150, marker='*', zorder=10,
                label=f'Целевая точка ({target_x:.2f}, {target_y:.2f})')
    ax2.legend()
    plt.show()

    # 7.3. Рабочая зона
    print("\n7.3. Рабочая зона с целевой точкой...")
    fig3, ax3 = plot_workspace(robot, resolution=50)
    ax3.scatter(target_x, target_y,
                color='green', s=200, marker='*', zorder=10,
                label=f'Целевая точка ({target_x:.2f}, {target_y:.2f})')
    ax3.legend()
    plt.show()

    # 7.4. Анимация движения к цели
    print("\n7.4. Анимация движения к целевой точке...")
    print(f"   Всего кадров: {len(trajectory)}")
    print("   Робот движется к цели, затем останавливается")
    animate_manipulator(robot, trajectory, interval=50)

    # ============================================================
    # 8. ИТОГИ
    # ============================================================
    print("\n" + "=" * 70)
    print("РЕЗУЛЬТАТЫ МОДЕЛИРОВАНИЯ")
    print("=" * 70)

    print("\n1. Кинематика:")
    print(f"   - Целевая точка: ({target_x:.4f}, {target_y:.4f}, {target_z:.2f}) м")
    print(f"   - Найденные углы: ψ={target_psi:.4f}, ξ={target_xi:.4f}")
    print(f"   - Ошибка позиционирования: {abs(target_x - x_actual):.2e} м")

    print("\n2. Динамика:")
    print(f"   - Требуемая сила F_z = {dynamics['F_z']:.3f} Н")
    print(f"   - Требуемый момент M_ψ = {dynamics['M_psi']:.3f} Н·м")
    print(f"   - Требуемый момент M_ξ = {dynamics['M_xi']:.3f} Н·м")

    print("\n3. Траектория:")
    print(f"   - Количество шагов: {len(trajectory)}")
    print(f"   - Начальное состояние: z=0.30, ψ=0.00, ξ=0.00")
    print(f"   - Конечное состояние: z={trajectory[-1][0]:.2f}, "
          f"ψ={trajectory[-1][1]:.4f}, ξ={trajectory[-1][2]:.4f}")

    print("\n4. Визуализация:")
    print("   3D модель с целевой точкой")
    print("   Вид сверху с целевой точкой")
    print("   Рабочая зона с целевой точкой")
    print("   Анимация движения к цели (100 шагов)")

    print("\n" + "=" * 70)
    print("МОДЕЛИРОВАНИЕ ЗАВЕРШЕНО")
    print("=" * 70)


if __name__ == "__main__":
    main()