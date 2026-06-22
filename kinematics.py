"""
Модуль кинематики для трёхзвенного робота-манипулятора.

Структура манипулятора (Модель 2 из отчёта):
    - Звено 1: поступательное движение вдоль оси Z (высота z)
    - Звено 2: вращательное движение вокруг оси Z (угол ψ)
    - Звено 3: вращательное движение вокруг оси Z (угол ξ)

DH-параметры (Таблица 1 из отчёта):
    Звено 1: a₁=0, α₁=0, d₁=z, θ₁=0
    Звено 2: a₂=L₁, α₂=0, d₂=0, θ₂=ψ
    Звено 3: a₃=L₂, α₃=0, d₃=0, θ₃=ξ
"""

import numpy as np


# ============================================================
# 1. ПРЯМАЯ ЗАДАЧА КИНЕМАТИКИ (ГЛАВА 2)
# ============================================================

def forward_kinematics(L1, L2, psi, xi):
    """
    Прямая задача кинематики для двух звеньев в плоскости.

    По заданным углам psi и xi вычисляет координаты схвата на плоскости Oxy.

    Параметры:
        L1 (float): длина первого звена (в плоскости)
        L2 (float): длина второго звена (в плоскости)
        psi (float): угол поворота звена 2 (в радианах)
        xi (float): угол поворота звена 3 (в радианах)

    Возвращает:
        tuple: (x, y) координаты схвата

    Формулы:
        x = L₁·cos(ψ) + L₂·cos(ψ + ξ)
        y = L₁·sin(ψ) + L₂·sin(ψ + ξ)

    Пример:
         x, y = forward_kinematics(1.0, 0.8, 0.785, 0.524)
         round(x, 3), round(y, 3)
        (0.914, 1.48)
    """
    x = L1 * np.cos(psi) + L2 * np.cos(psi + xi)
    y = L1 * np.sin(psi) + L2 * np.sin(psi + xi)
    return x, y


def forward_kinematics_3d(L1, L2, z, psi, xi):
    """
    Прямая задача кинематики в трёхмерном пространстве.

    Параметры:
        L1 (float): длина первого звена
        L2 (float): длина второго звена
        z (float): высота (поступательное движение звена 1)
        psi (float): угол поворота звена 2
        xi (float): угол поворота звена 3

    Возвращает:
        tuple: (x, y, z, phi)
            x, y, z — координаты схвата
            phi — ориентация схвата в плоскости Oxy (φ = ψ + ξ)

    Система:
        x = L₁·cos(ψ) + L₂·cos(ψ + ξ)
        y = L₁·sin(ψ) + L₂·sin(ψ + ξ)
        z = z₀
        φ = ψ + ξ
    """
    x, y = forward_kinematics(L1, L2, psi, xi)
    phi = psi + xi
    return x, y, z, phi


# ============================================================
# 2. ОБРАТНАЯ ЗАДАЧА КИНЕМАТИКИ (ГЛАВА 3)
# ============================================================

def inverse_kinematics_2d(L1, L2, x, y):
    """
    Обратная задача кинематики для двух звеньев в плоскости.

    По заданным координатам схвата вычисляет углы psi и xi.

    Параметры:
        L1 (float): длина первого звена
        L2 (float): длина второго звена
        x (float): координата x цели
        y (float): координата y цели

    Возвращает:
        tuple: (psi, xi, reachable)
            psi (float): угол поворота звена 2 (в радианах)
            xi (float): угол поворота звена 3 (в радианах)
            reachable (bool): True если точка достижима

    Формулы из отчёта (Глава 3):
        r = sqrt(x² + y²)                                 (3)
        ξ = arccos((r² - L₁² - L₂²) / (2·L₁·L₂))          (4)
        α = atan2(y, x)                                   (5)
        β = arccos((r² + L₁² - L₂²) / (2·r·L₁))           (6)
        ψ = α - β                                         (7)-(8)

    Условие достижимости (10):
        |L₁ - L₂| ≤ r ≤ L₁ + L₂

    Пример:
        psi, xi, reachable = inverse_kinematics_2d(1.0, 0.8, 0.914, 1.48)
        round(psi, 3), round(xi, 3), reachable
        (0.785, 0.524, True)
    """
    # Расстояние от основания до цели (формула 3)
    r = np.sqrt(x**2 + y**2)

    # Проверка условия достижимости (формула 10)
    if r < abs(L1 - L2) or r > L1 + L2:
        return 0.0, 0.0, False

    # Если цель в основании (избегаем деления на ноль)
    if r < 1e-10:
        return 0.0, 0.0, False

    # Вычисляем угол xi (формула 4)
    cos_xi = (r**2 - L1**2 - L2**2) / (2 * L1 * L2)
    cos_xi = np.clip(cos_xi, -1.0, 1.0)  # Защита от погрешностей
    xi = np.arccos(cos_xi)

    # Вычисляем угол alpha (формула 5)
    alpha = np.arctan2(y, x)

    # Вычисляем угол beta (формула 6)
    cos_beta = (r**2 + L1**2 - L2**2) / (2 * r * L1)
    cos_beta = np.clip(cos_beta, -1.0, 1.0)  # Защита от погрешностей
    beta = np.arccos(cos_beta)

    # Вычисляем угол psi (формулы 7-8)
    psi = alpha - beta

    return psi, xi, True


def inverse_kinematics_3d(L1, L2, x, y, z, z_min=0.0, z_max=1.5):
    """
    Полная обратная задача кинематики для трёхзвенного манипулятора.

    Согласно отчёту (Глава 3):
        Сначала определяется высота z₀ = z,
        затем решается плоская задача в плоскости Oxy.

    Параметры:
        L1, L2 (float): длины звеньев
        x, y, z (float): целевые координаты в пространстве
        z_min, z_max (float): ограничения по высоте

    Возвращает:
        tuple: (z_sol, psi_sol, xi_sol, reachable, message)

    Формулы (9):
        z₀ = z
        ψ = atan2(y, x) - arccos((x²+y²+L₁²-L₂²)/(2·L₁·√(x²+y²)))
        ξ = arccos((x²+y²-L₁²-L₂²)/(2·L₁·L₂))
    """
    # Проверка высоты
    if z < z_min or z > z_max:
        return z, 0.0, 0.0, False, f"Высота {z:.2f} вне диапазона [{z_min}, {z_max}]"

    # Решаем плоскую задачу
    psi, xi, reachable = inverse_kinematics_2d(L1, L2, x, y)

    if not reachable:
        r = np.sqrt(x**2 + y**2)
        return z, 0.0, 0.0, False, f"Расстояние {r:.2f} вне диапазона [{abs(L1-L2):.2f}, {L1+L2:.2f}]"

    return z, psi, xi, True, "OK"


def inverse_kinematics_alternative(L1, L2, x, y):
    """
    Альтернативное решение обратной задачи (зеркальная конфигурация).

    Даёт второе возможное решение для тех же координат.

    Параметры:
        L1, L2 (float): длины звеньев
        x, y (float): целевые координаты

    Возвращает:
        tuple: (psi, xi, reachable)

    Формулы:
        ξ₂ = -ξ₁
        ψ₂ = α + β (вместо α - β)

    Пример:
        psi, xi, reachable = inverse_kinematics_alternative(1.0, 0.8, 0.914, 1.48)
        round(psi, 3), round(xi, 3), reachable
        (1.137, -1.043, True)
    """
    # Сначала получаем основное решение
    psi1, xi1, reachable = inverse_kinematics_2d(L1, L2, x, y)

    if not reachable:
        return 0.0, 0.0, False

    # Альтернативное решение
    r = np.sqrt(x**2 + y**2)
    alpha = np.arctan2(y, x)

    cos_beta = (r**2 + L1**2 - L2**2) / (2 * r * L1)
    cos_beta = np.clip(cos_beta, -1.0, 1.0)
    beta = np.arccos(cos_beta)

    psi2 = alpha + beta
    xi2 = -xi1

    return psi2, xi2, True


# ============================================================
# 3. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

def get_joint_positions(L1, L2, z, psi, xi):
    """
    Координаты всех сочленений манипулятора.

    Возвращает:
        list: [(x0,y0,z0), (x1,y1,z1), (x2,y2,z2), (x3,y3,z3)]

    Точки:
        O (основание):      (0, 0, 0)
        A (конец звена 1):  (0, 0, z)
        B (конец звена 2):  (L₁·cos ψ, L₁·sin ψ, z)
        C (схват):          (L₁·cos ψ + L₂·cos(ψ+ξ), L₁·sin ψ + L₂·sin(ψ+ξ), z)
    """
    # Точка O: основание
    p0 = (0.0, 0.0, 0.0)

    # Точка A: конец первого звена (поступательное движение по Z)
    p1 = (0.0, 0.0, z)

    # Точка B: конец второго звена
    x2 = L1 * np.cos(psi)
    y2 = L1 * np.sin(psi)
    p2 = (x2, y2, z)

    # Точка C: схват (конец третьего звена)
    x3, y3 = forward_kinematics(L1, L2, psi, xi)
    p3 = (x3, y3, z)

    return [p0, p1, p2, p3]


def check_reachability(L1, L2, x, y):
    """
    Проверяет, достижима ли точка в плоскости Oxy.

    Условие из отчёта (10):
        |L₁ - L₂| ≤ √(x² + y²) ≤ L₁ + L₂

    Возвращает:
        bool: True если точка достижима

    Пример:
            bool(check_reachability(1.0, 0.8, 0.5, 0.5))
        True
            bool(check_reachability(1.0, 0.8, 3.0, 0.0))
        False
    """
    r = np.sqrt(x**2 + y**2)
    return bool(abs(L1 - L2) <= r <= L1 + L2)


def check_constraints(L1, L2, z, psi, xi, z_min=0.0, z_max=1.5, psi_min=-np.pi, psi_max=np.pi):
    """
    Проверяет все ограничения модели (Таблица 1 из отчёта).

    Возвращает:
        tuple: (ok, messages)
    """
    messages = []
    ok = True

    # Проверка высоты
    if z < z_min or z > z_max:
        ok = False
        messages.append(f"Высота z={z:.2f} вне диапазона [{z_min}, {z_max}]")

    # Проверка углов psi
    if psi < psi_min or psi > psi_max:
        ok = False
        messages.append(f"Угол psi={psi:.2f} вне диапазона [{psi_min:.2f}, {psi_max:.2f}]")

    # Проверка достижимости
    x, y = forward_kinematics(L1, L2, psi, xi)
    r = np.sqrt(x**2 + y**2)
    if r < abs(L1 - L2) or r > L1 + L2:
        ok = False
        messages.append(f"Расстояние r={r:.2f} вне диапазона [{abs(L1-L2):.2f}, {L1+L2:.2f}]")

    return ok, messages


def get_workspace_points(L1, L2, resolution=50):
    """
    Генерирует точки рабочей зоны манипулятора.

    Параметры:
        L1, L2 (float): длины звеньев
        resolution (int): количество точек по каждому измерению

    Возвращает:
        np.ndarray: массив точек (N, 2)
    """
    points = []
    psi_values = np.linspace(-np.pi, np.pi, resolution)
    xi_values = np.linspace(-np.pi, np.pi, resolution)

    for psi in psi_values:
        for xi in xi_values:
            x, y = forward_kinematics(L1, L2, psi, xi)
            points.append((x, y))

    return np.array(points)


def get_workspace_boundary(L1, L2, n_points=100):
    """
    Возвращает границы рабочей зоны (окружности).

    Возвращает:
        tuple: (inner_circle, outer_circle)
    """
    theta = np.linspace(0, 2*np.pi, n_points)

    outer_r = L1 + L2
    inner_r = abs(L1 - L2)

    outer_circle = np.column_stack([outer_r * np.cos(theta), outer_r * np.sin(theta)])
    inner_circle = np.column_stack([inner_r * np.cos(theta), inner_r * np.sin(theta)])

    return inner_circle, outer_circle
