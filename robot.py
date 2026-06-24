"""
Основной класс робота-манипулятора.
Объединяет кинематику, динамику и визуализацию.
"""

import numpy as np
from kinematics import (
    forward_kinematics,
    inverse_kinematics_2d,
    inverse_kinematics_alternative,
    get_joint_positions,
    check_constraints
)
from dynamics import (
    compute_torques,
    check_velocity_constraints,
    check_acceleration_constraints,
    check_torque_constraints
)


class Robot:
    """
    Класс трёхзвенного робота-манипулятора.

    Пользователь задаёт параметры робота, затем может:
        1. Задать целевую точку и получить углы (обратная кинематика)
        2. Получить усилия для движения (динамика)
        3. Визуализировать текущее состояние
    """

    def __init__(self,
                 L1=1.0, L2=0.8,  # длины звеньев
                 m1=1.0, m2=0.5, m3=0.3,  # массы звеньев
                 I_z2=0.1, I_z3=0.05,  # моменты инерции
                 z_min=0.0, z_max=1.5,  # ограничения по высоте
                 psi_min=-np.pi, psi_max=np.pi,  # ограничения по углам
                 g=9.81,  # ускорение свободного падения
                 max_velocity=2.0,  # максимальная скорость (рад/с, м/с)
                 max_acceleration=5.0,  # максимальное ускорение (рад/с², м/с²)
                 max_torque=10.0):  # максимальный момент/сила
        """
        Создание робота с заданными параметрами.

        Параметры:
            L1, L2 (float): длины звеньев (м)
            m1, m2, m3 (float): массы звеньев (кг)
            I_z2, I_z3 (float): моменты инерции (кг·м²)
            z_min, z_max (float): диапазон высоты (м)
            psi_min, psi_max (float): диапазон угла psi (рад)
            g (float): ускорение свободного падения (м/с²)
            max_velocity (float): максимальная скорость (рад/с, м/с)
            max_acceleration (float): максимальное ускорение (рад/с², м/с²)
            max_torque (float): максимальный момент/сила
        """
        # === Параметры робота ===
        self.L1 = L1
        self.L2 = L2
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.I_z2 = I_z2
        self.I_z3 = I_z3
        self.g = g

        # === Ограничения ===
        self.z_min = z_min
        self.z_max = z_max
        self.psi_min = psi_min
        self.psi_max = psi_max
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.max_torque = max_torque

        # === Текущее состояние ===
        self.z = 0.0
        self.psi = 0.0
        self.xi = 0.0
        self.dz = 0.0
        self.dpsi = 0.0
        self.dxi = 0.0

        # === Информация о последнем решении ===
        self.last_solution = None

    # ============================================================
    # 1. УПРАВЛЕНИЕ СОСТОЯНИЕМ
    # ============================================================

    def set_state(self, z, psi, xi, dz=0.0, dpsi=0.0, dxi=0.0):
        """
        Установить состояние робота.

        Параметры:
            z (float): высота (м)
            psi (float): угол первого звена (рад)
            xi (float): угол второго звена (рад)
            dz, dpsi, dxi (float): скорости (опционально)
        """
        self.z = z
        self.psi = psi
        self.xi = xi
        self.dz = dz
        self.dpsi = dpsi
        self.dxi = dxi

    def get_state(self):
        """Получить текущее состояние."""
        return {
            'z': self.z,
            'psi': self.psi,
            'xi': self.xi,
            'dz': self.dz,
            'dpsi': self.dpsi,
            'dxi': self.dxi
        }

    def get_info(self):
        """Получить информацию о параметрах робота."""
        return {
            'L1': self.L1,
            'L2': self.L2,
            'm1': self.m1,
            'm2': self.m2,
            'm3': self.m3,
            'I_z2': self.I_z2,
            'I_z3': self.I_z3,
            'g': self.g,
            'z_range': (self.z_min, self.z_max),
            'psi_range': (self.psi_min, self.psi_max),
            'reachability_range': (abs(self.L1 - self.L2), self.L1 + self.L2),
            'max_velocity': self.max_velocity,
            'max_acceleration': self.max_acceleration,
            'max_torque': self.max_torque
        }

    # ============================================================
    # 2. КИНЕМАТИКА
    # ============================================================

    def forward_kinematics(self):
        """
        Прямая задача кинематики.

        Возвращает:
            tuple: (x, y, z) координаты схвата
        """
        x = self.L1 * np.cos(self.psi) + self.L2 * np.cos(self.psi + self.xi)
        y = self.L1 * np.sin(self.psi) + self.L2 * np.sin(self.psi + self.xi)
        return x, y, self.z

    def inverse_kinematics(self, x, y, z):
        """
        Обратная задача кинематики.

        По заданной целевой точке вычисляет углы.

        Параметры:
            x, y, z (float): целевые координаты

        Возвращает:
            dict: {
                'success': bool,
                'z': float, 'psi': float, 'xi': float,
                'message': str,
                'reachable': bool
            }
        """
        result = {
            'success': False,
            'z': z,
            'psi': 0.0,
            'xi': 0.0,
            'message': '',
            'reachable': False
        }

        # 1. Проверка высоты
        if z < self.z_min or z > self.z_max:
            result['message'] = f"Высота {z:.2f} вне диапазона [{self.z_min}, {self.z_max}]"
            self.last_solution = result
            return result

        # 2. Проверка достижимости в плоскости
        r = np.sqrt(x ** 2 + y ** 2)
        if r < abs(self.L1 - self.L2) or r > self.L1 + self.L2:
            result[
                'message'] = f"Расстояние {r:.2f} вне диапазона [{abs(self.L1 - self.L2):.2f}, {self.L1 + self.L2:.2f}]"
            self.last_solution = result
            return result

        # 3. Решаем обратную задачу
        psi, xi, reachable = inverse_kinematics_2d(self.L1, self.L2, x, y)

        if not reachable:
            result['message'] = "Не удалось найти решение"
            self.last_solution = result
            return result

        # 4. Проверка ограничений на углы
        if psi < self.psi_min or psi > self.psi_max:
            # Пробуем альтернативное решение
            psi_alt, xi_alt, reachable_alt = inverse_kinematics_alternative(
                self.L1, self.L2, x, y
            )
            if reachable_alt and self.psi_min <= psi_alt <= self.psi_max:
                psi, xi = psi_alt, xi_alt
                result['message'] = "Альтернативное решение (в пределах ограничений)"
            else:
                result['message'] = f"Угол psi={psi:.2f} вне диапазона [{self.psi_min:.2f}, {self.psi_max:.2f}]"
                self.last_solution = result
                return result

        # 5. Если цель достижима - ок
        result['success'] = True
        result['reachable'] = True
        result['z'] = z
        result['psi'] = psi
        result['xi'] = xi
        result['message'] = "OK"

        # Автоматически устанавливаем состояние
        self.set_state(z, psi, xi)
        self.last_solution = result

        return result

    def go_to_target(self, x, y, z):
        """
        Переместить робота в целевую точку.
        Это обёртка над inverse_kinematics.

        Возвращает:
            bool: True если успешно
        """
        result = self.inverse_kinematics(x, y, z)
        return result['success']

    # ============================================================
    # 3. ДИНАМИКА
    # ============================================================

    def compute_accelerations_from_time(self, target_z, target_psi, target_xi, T):
        """
        Вычисляет ускорения, необходимые для достижения цели за время T.

        Формула для равноускоренного движения с нулевой начальной скоростью:
            q_target = q_start + 0.5 * a * T^2
            a = 2 * (q_target - q_start) / T^2

        Параметры:
            target_z, target_psi, target_xi (float): целевые координаты
            T (float): время движения (сек)

        Возвращает:
            tuple: (ddz, ddpsi, ddxi)
        """
        dz = target_z - self.z
        dpsi = target_psi - self.psi
        dxi = target_xi - self.xi

        ddz = 2 * dz / (T ** 2)
        ddpsi = 2 * dpsi / (T ** 2)
        ddxi = 2 * dxi / (T ** 2)

        # Проверка на превышение максимального ускорения
        if abs(ddz) > self.max_acceleration:
            print(f"Предупреждение: ускорение ddz={ddz:.2f} превышает максимум {self.max_acceleration:.2f}")
        if abs(ddpsi) > self.max_acceleration:
            print(f"Предупреждение: ускорение ddpsi={ddpsi:.2f} превышает максимум {self.max_acceleration:.2f}")
        if abs(ddxi) > self.max_acceleration:
            print(f"Предупреждение: ускорение ddxi={ddxi:.2f} превышает максимум {self.max_acceleration:.2f}")

        return ddz, ddpsi, ddxi

    def plan_trajectory(self, target_z, target_psi, target_xi, n_steps=50):
        """
        Создаёт простую траекторию от текущего состояния к целевому.

        Параметры:
            target_z, target_psi, target_xi (float): целевые координаты
            n_steps (int): количество шагов

        Возвращает:
            list: траектория [(z, psi, xi), ...]
        """
        trajectory = []
        for i in range(n_steps + 1):
            t = i / n_steps
            # Линейная интерполяция
            z = self.z + t * (target_z - self.z)
            psi = self.psi + t * (target_psi - self.psi)
            xi = self.xi + t * (target_xi - self.xi)
            trajectory.append((z, psi, xi))
        return trajectory

    def plan_trajectory_with_time(self, target_z, target_psi, target_xi, T, freq=50):
        """
        Планирует траекторию на время T с частотой freq Гц.

        Параметры:
            target_z, target_psi, target_xi (float): целевые координаты
            T (float): время движения (сек)
            freq (int): частота обновления (Гц)

        Возвращает:
            list: траектория [(z, psi, xi), ...]
        """
        n_steps = int(T * freq)
        if n_steps < 10:
            n_steps = 10
            print(f"Предупреждение: слишком мало шагов ({n_steps}), установлено минимум 10")
        return self.plan_trajectory(target_z, target_psi, target_xi, n_steps)

    def compute_dynamics(self, ddz=0.0, ddpsi=0.0, ddxi=0.0):
        """
        Вычислить требуемые усилия для заданных ускорений.

        Параметры:
            ddz, ddpsi, ddxi (float): требуемые ускорения

        Возвращает:
            dict: {
                'F_z': float,
                'M_psi': float,
                'M_xi': float,
                'tau': np.ndarray,
                'constraints_ok': bool,
                'messages': list
            }
        """
        q = np.array([self.z, self.psi, self.xi])
        dq = np.array([self.dz, self.dpsi, self.dxi])
        ddq = np.array([ddz, ddpsi, ddxi])

        tau = compute_torques(
            self.m1, self.m2, self.m3,
            self.I_z2, self.I_z3,
            q, dq, ddq, self.g
        )

        # Проверка ограничений
        constraints_ok = True
        messages = []

        # Скорости (используем max_velocity)
        ok, msgs = check_velocity_constraints(
            self.dz, self.dpsi, self.dxi,
            dz_max=self.max_velocity,
            dpsi_max=self.max_velocity,
            dxi_max=self.max_velocity
        )
        if not ok:
            constraints_ok = False
            messages.extend(msgs)

        # Ускорения (используем max_acceleration)
        ok, msgs = check_acceleration_constraints(
            ddz, ddpsi, ddxi,
            ddz_max=self.max_acceleration,
            ddpsi_max=self.max_acceleration,
            ddxi_max=self.max_acceleration
        )
        if not ok:
            constraints_ok = False
            messages.extend(msgs)

        # Усилия (используем max_torque)
        ok, msgs = check_torque_constraints(
            tau[0], tau[1], tau[2],
            F_z_max=self.max_torque,
            M_psi_max=self.max_torque,
            M_xi_max=self.max_torque
        )
        if not ok:
            constraints_ok = False
            messages.extend(msgs)

        return {
            'F_z': tau[0],
            'M_psi': tau[1],
            'M_xi': tau[2],
            'tau': tau,
            'constraints_ok': constraints_ok,
            'messages': messages
        }

    def compute_dynamics_for_trajectory(self, trajectory, freq=50):
        """
        Вычисляет динамику для всей траектории.

        Параметры:
            trajectory (list): список состояний [(z, psi, xi), ...]
            freq (int): частота обновления (Гц)

        Возвращает:
            dict: {
                'times': list,
                'tau': list,
                'states': list,  # (z, psi, xi, dz, dpsi, dxi)
                'constraints_ok': bool,
                'messages': list
            }
        """
        times = []
        tau_list = []
        states_list = []
        all_messages = []
        constraints_ok = True

        dt = 1.0 / freq

        for i, state in enumerate(trajectory):
            z, psi, xi = state

            # Вычисляем скорости (конечные разности)
            if i == 0:
                dz = 0.0
                dpsi = 0.0
                dxi = 0.0
            else:
                prev_state = trajectory[i - 1]
                dz = (z - prev_state[0]) / dt
                dpsi = (psi - prev_state[1]) / dt
                dxi = (xi - prev_state[2]) / dt

            # Вычисляем ускорения (конечные разности)
            if i == 0 or i == 1:
                ddz = 0.0
                ddpsi = 0.0
                ddxi = 0.0
            else:
                prev_prev_state = trajectory[i - 2]
                ddz = (z - 2 * prev_state[0] + prev_prev_state[0]) / (dt ** 2)
                ddpsi = (psi - 2 * prev_state[1] + prev_prev_state[1]) / (dt ** 2)
                ddxi = (xi - 2 * prev_state[2] + prev_prev_state[2]) / (dt ** 2)

            # ОБНОВЛЯЕМ СОСТОЯНИЕ с вычисленными скоростями
            self.set_state(z, psi, xi, dz, dpsi, dxi)

            # Сохраняем состояние с скоростями
            states_list.append((z, psi, xi, dz, dpsi, dxi))

            # Вычисляем усилия
            result = self.compute_dynamics(ddz, ddpsi, ddxi)

            times.append(i * dt)
            tau_list.append(result['tau'])

            if not result['constraints_ok']:
                constraints_ok = False
                all_messages.extend(result['messages'])

        # Возвращаем начальное состояние (чтобы не испортить робота)
        if states_list:
            first_state = states_list[0]
            self.set_state(first_state[0], first_state[1], first_state[2], 0, 0, 0)

        return {
            'times': times,
            'tau': tau_list,
            'states': states_list,
            'constraints_ok': constraints_ok,
            'messages': list(set(all_messages))
        }

    def compute_static_torques(self):
        """
        Вычислить статические усилия (для удержания позиции).
        """
        from dynamics import gravity_vector
        G = gravity_vector(self.m1, self.m2, self.m3, self.g)
        return {
            'F_z': G[0],
            'M_psi': G[1],
            'M_xi': G[2]
        }

    # ============================================================
    # 4. ВИЗУАЛИЗАЦИЯ
    # ============================================================

    def get_joint_positions(self):
        """Получить координаты всех сочленений."""
        return get_joint_positions(self.L1, self.L2, self.z, self.psi, self.xi)

    def check_constraints(self):
        """Проверить все ограничения."""
        return check_constraints(
            self.L1, self.L2, self.z, self.psi, self.xi,
            self.z_min, self.z_max, self.psi_min, self.psi_max
        )

    # ============================================================
    # 5. ВЫВОД ИНФОРМАЦИИ
    # ============================================================

    def __str__(self):
        x, y, z = self.forward_kinematics()
        return (f"Робот-манипулятор\n"
                f"  Состояние: z={self.z:.3f} м, ψ={self.psi:.3f} рад, ξ={self.xi:.3f} рад\n"
                f"  Скорости:  dz={self.dz:.3f}, dψ={self.dpsi:.3f}, dξ={self.dxi:.3f}\n"
                f"  Схват:     ({x:.3f}, {y:.3f}, {z:.3f}) м")

    def print_info(self):
        """Вывести информацию о роботе."""
        info = self.get_info()
        print("=" * 50)
        print("ПАРАМЕТРЫ РОБОТА")
        print("=" * 50)
        print(f"  L1 = {info['L1']:.2f} м")
        print(f"  L2 = {info['L2']:.2f} м")
        print(f"  m1 = {info['m1']:.2f} кг")
        print(f"  m2 = {info['m2']:.2f} кг")
        print(f"  m3 = {info['m3']:.2f} кг")
        print(f"  I_z2 = {info['I_z2']:.3f} кг·м²")
        print(f"  I_z3 = {info['I_z3']:.3f} кг·м²")
        print(f"  g = {info['g']:.2f} м/с²")
        print(f"  Диапазон z: {info['z_range']}")
        print(f"  Диапазон ψ: {info['psi_range']}")
        print(f"  Достижимость: {info['reachability_range']}")
        print(f"  Макс. скорость: {info['max_velocity']:.1f}")
        print(f"  Макс. ускорение: {info['max_acceleration']:.1f}")
        print(f"  Макс. усилие: {info['max_torque']:.1f}")
        print("=" * 50)