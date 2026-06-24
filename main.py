"""
Главный файл. Демонстрирует полный цикл работы с роботом.
Пользователь задаёт целевую точку (x, y, z) и время движения.
"""

import numpy as np
import matplotlib.pyplot as plt
from robot import Robot
from visualization import draw_3d, draw_top_view, animate_manipulator, plot_workspace


def main():
    # ============================================================
    # 1. СОЗДАНИЕ РОБОТА <<<<<
    # ============================================================
    robot = Robot(
        L1=1.0, L2=0.8,
        m1=1.0, m2=0.5, m3=0.3,
        I_z2=0.1, I_z3=0.05,
        z_min=0.0, z_max=1.5,
        psi_min=-np.pi, psi_max=np.pi,
        g=9.81,
        max_velocity=2.0,
        max_acceleration=5.0,
        max_torque=10.0
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
    # 3. ПОЛЬЗОВАТЕЛЬ ЗАДАЁТ ТОЧКУ И ВРЕМЯ <<<<<
    # ============================================================
    print("\n" + "-" * 50)
    print("ПОЛЬЗОВАТЕЛЬ ЗАДАЁТ ЦЕЛЕВУЮ ТОЧКУ И ВРЕМЯ")
    print("-" * 50)

    target_x = 0.9142
    target_y = 1.4798
    target_z = 0.5
    T = 2.0

    print(f"Целевая точка: ({target_x:.4f}, {target_y:.4f}, {target_z:.2f}) м")
    print(f"Время движения: {T:.1f} сек")

    # ============================================================
    # 4. СОХРАНЯЕМ НАЧАЛЬНОЕ СОСТОЯНИЕ
    # ============================================================
    start_z = robot.z
    start_psi = robot.psi
    start_xi = robot.xi

    # ============================================================
    # 5. ОБРАТНАЯ КИНЕМАТИКА
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

    x_actual, y_actual, z_actual = robot.forward_kinematics()
    print(f"\nОшибка позиционирования:")
    print(f"  Δx = {abs(target_x - x_actual):.2e} м")
    print(f"  Δy = {abs(target_y - y_actual):.2e} м")
    print(f"  Δz = {abs(target_z - z_actual):.2e} м")

    # ============================================================
    # 6. АВТОМАТИЧЕСКИЙ РАСЧЁТ УСКОРЕНИЙ
    # ============================================================
    print("\nАВТОМАТИЧЕСКИЙ РАСЧЁТ УСКОРЕНИЙ")

    robot.set_state(start_z, start_psi, start_xi)
    ddz, ddpsi, ddxi = robot.compute_accelerations_from_time(
        target_z, target_psi, target_xi, T
    )

    print(f"Требуемые ускорения для движения за {T:.1f} сек:")
    print(f"  ddz  = {ddz:.3f} м/с²")
    print(f"  ddψ  = {ddpsi:.3f} рад/с²")
    print(f"  ddξ  = {ddxi:.3f} рад/с²")

    robot.set_state(target_z, target_psi, target_xi)

    # ============================================================
    # 7. ДИНАМИКА
    # ============================================================
    print("\n РАСЧЁТ ДИНАМИКИ ")

    robot.set_state(start_z, start_psi, start_xi)
    dynamics = robot.compute_dynamics(ddz, ddpsi, ddxi)
    robot.set_state(target_z, target_psi, target_xi)

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
    # 8. ДИНАМИКА ВДОЛЬ ТРАЕКТОРИИ
    # ============================================================
    print("\n ДИНАМИКА ВДОЛЬ ТРАЕКТОРИИ")

    freq = 50
    robot.set_state(z=0.3, psi=0.0, xi=0.0)
    trajectory_motion = robot.plan_trajectory_with_time(
        target_z=target_z,
        target_psi=target_psi,
        target_xi=target_xi,
        T=T,
        freq=freq
    )

    dynamics_result = robot.compute_dynamics_for_trajectory(trajectory_motion, freq=freq)

    print(f"Анализ траектории (без удержания):")
    print(f"  Шагов: {len(trajectory_motion)}")
    print(f"  Время: {T:.1f} сек")

    print("\nСостояния вдоль траектории (каждые 10 шагов):")
    for i in range(0, len(trajectory_motion), 10):
        z, psi, xi = trajectory_motion[i]

        # Проверяем, есть ли состояние с скоростями
        if i < len(dynamics_result['states']):
            _, _, _, dz, dpsi, dxi = dynamics_result['states'][i]
        else:
            dz, dpsi, dxi = 0.0, 0.0, 0.0

        tau = dynamics_result['tau'][i]
        print(f"  Шаг {i:3d}: z={z:.3f}, ψ={psi:.3f}, ξ={xi:.3f}, "
              f"dz={dz:.3f}, dψ={dpsi:.3f}, dξ={dxi:.3f}, "
              f"F_z={tau[0]:.3f}, M_ψ={tau[1]:.3f}, M_ξ={tau[2]:.3f}")

    # ============================================================
    # 9. ПЛАНИРОВАНИЕ ТРАЕКТОРИИ
    # ============================================================
    print("\n ПЛАНИРОВАНИЕ ТРАЕКТОРИИ ")

    robot.set_state(z=0.3, psi=0.0, xi=0.0)

    trajectory = robot.plan_trajectory_with_time(
        target_z=target_z,
        target_psi=target_psi,
        target_xi=target_xi,
        T=T,
        freq=freq
    )

    hold_frames = int(0.5 * freq)
    final_state = trajectory[-1]
    for _ in range(hold_frames):
        trajectory.append(final_state)

    print(f"Создана траектория из {len(trajectory)} шагов")
    print(f"  - Движение: {int(T * freq)} шагов ({T:.1f} сек)")
    print(f"  - Удержание: {hold_frames} шагов (0.5 сек)")

    # ============================================================
    # 10. ВИЗУАЛИЗАЦИЯ
    # ============================================================
    print("\n ВИЗУАЛИЗАЦИЯ")

    robot.set_state(z=target_z, psi=target_psi, xi=target_xi)

    print("\n10.1. 3D вид с целевой точкой...")
    ax1 = draw_3d(robot, title="3D модель манипулятора (достигнута цель)")
    ax1.scatter(target_x, target_y, target_z,
                color='green', s=150, marker='*', zorder=10,
                label=f'Целевая точка ({target_x:.2f}, {target_y:.2f}, {target_z:.2f})')
    ax1.legend()
    plt.show()

    print("\n10.2. Вид сверху с целевой точкой...")
    ax2 = draw_top_view(robot, title="Вид сверху (проекция на Oxy)")
    ax2.scatter(target_x, target_y,
                color='green', s=150, marker='*', zorder=10,
                label=f'Целевая точка ({target_x:.2f}, {target_y:.2f})')
    ax2.legend()
    plt.show()

    print("\n10.3. Рабочая зона с целевой точкой...")
    fig3, ax3 = plot_workspace(robot, resolution=50)
    ax3.scatter(target_x, target_y,
                color='green', s=200, marker='*', zorder=10,
                label=f'Целевая точка ({target_x:.2f}, {target_y:.2f})')
    ax3.legend()
    plt.show()

    print("\n10.4. Анимация движения к целевой точке...")
    print(f"   Всего кадров: {len(trajectory)}")
    print(f"   - Движение: {int(T * freq)} кадров ({T:.1f} сек)")
    print(f"   - Удержание: {hold_frames} кадров (0.5 сек)")
    animate_manipulator(robot, trajectory, interval=50)

    # ============================================================
    # 11. ИТОГИ
    # ============================================================
    print("\n" + "=" * 70)
    print("РЕЗУЛЬТАТЫ МОДЕЛИРОВАНИЯ")
    print("=" * 70)

    print("\n1. Кинематика:")
    print(f"   - Целевая точка: ({target_x:.4f}, {target_y:.4f}, {target_z:.2f}) м")
    print(f"   - Найденные углы: ψ={target_psi:.4f}, ξ={target_xi:.4f}")
    print(f"   - Ошибка позиционирования: {abs(target_x - x_actual):.2e} м")

    print("\n2. Динамика:")
    print(f"   - Время движения: {T:.1f} сек")
    print(f"   - Ускорения: ddz={ddz:.3f}, ddψ={ddpsi:.3f}, ddξ={ddxi:.3f}")
    print(f"   - Требуемая сила F_z = {dynamics['F_z']:.3f} Н")
    print(f"   - Требуемый момент M_ψ = {dynamics['M_psi']:.3f} Н·м")
    print(f"   - Требуемый момент M_ξ = {dynamics['M_xi']:.3f} Н·м")
    if dynamics['constraints_ok']:
        print("   - Все ограничения соблюдены")
    else:
        print("   - Обнаружены нарушения ограничений")

    print("\n3. Траектория:")
    print(f"   - Всего шагов: {len(trajectory)}")
    print(f"   - Движение: {int(T * freq)} шагов ({T:.1f} сек)")
    print(f"   - Удержание: {hold_frames} шагов (0.5 сек)")

    print("\n4. Визуализация:")
    print("   3D модель с целевой точкой")
    print("   Вид сверху с целевой точкой")
    print("   Рабочая зона с целевой точкой")
    print(f"   Анимация движения к цели ")

    print("\n" + "=" * 70)
    print("МОДЕЛИРОВАНИЕ ЗАВЕРШЕНО")
    print("=" * 70)


if __name__ == "__main__":
    main()