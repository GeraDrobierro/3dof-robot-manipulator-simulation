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
    check_reachability,
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
                 g=9.81):  # ускорение свободного падения
        """
        Создание робота с заданными параметрами.

        Параметры:
            L1, L2 (float): длины звеньев (м)
            m1, m2, m3 (float): массы звеньев (кг)
            I_z2, I_z3 (float): моменты инерции (кг·м²)
            z_min, z_max (float): диапазон высоты (м)
            psi_min, psi_max (float): диапазон угла psi (рад)
            g (float): ускорение свободного падения (м/с²)
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
            'reachability_range': (abs(self.L1 - self.L2), self.L1 + self.L2)
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

        # Скорости
        ok, msgs = check_velocity_constraints(self.dz, self.dpsi, self.dxi)
        if not ok:
            constraints_ok = False
            messages.extend(msgs)

        # Ускорения
        ok, msgs = check_acceleration_constraints(ddz, ddpsi, ddxi)
        if not ok:
            constraints_ok = False
            messages.extend(msgs)

        # Усилия
        ok, msgs = check_torque_constraints(tau[0], tau[1], tau[2])
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
    # 5. ПЛАНИРОВАНИЕ ТРАЕКТОРИИ
    # ============================================================

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

    # ============================================================
    # 6. ВЫВОД ИНФОРМАЦИИ
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
        print("=" * 50)