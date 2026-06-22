"""
Модуль динамики для трёхзвенного робота-манипулятора.


Уравнения динамики (26):
    M(q)·d(dq) + C(q, dq)·dq + G(q) = tau

где:
    q = [z, ψ, ξ]^T — обобщённые координаты
    dq = [dz, dψ, dξ]^T — обобщённые скорости
    ddq = [ddz, ddψ, ddξ]^T — обобщённые ускорения
    tau = [F_z, M_ψ, M_ξ]^T — обобщённые силы
"""
# ============================================================
# 0. ДИНАМИКА (ГЛАВА 4)
# ============================================================

import numpy as np


# ============================================================
# 1. МАТРИЦА ИНЕРЦИИ M(q) — ФОРМУЛА (27)
# ============================================================

def inertia_matrix(m1, m2, m3, I_z2, I_z3):
    """
    Матрица инерции M(q).

    Параметры:
        m1, m2, m3 (float): массы звеньев
        I_z2, I_z3 (float): моменты инерции звеньев 2 и 3 относительно оси z

    Возвращает:
        np.ndarray: матрица 3x3

    Формула (27):
        M(q) = [[m1,      0,          0    ],
                [0,   I_z2 + I_z3,   I_z3 ],
                [0,       I_z3,      I_z3 ]]


    Пример:
         M = inertia_matrix(1.0, 0.5, 0.3, 0.1, 0.05)
         M.shape
        (3, 3)
    """
    M = np.array([
        [m1, 0, 0],
        [0, I_z2 + I_z3, I_z3],
        [0, I_z3, I_z3]
    ], dtype=float)
    return M


# ============================================================
# 2. МАТРИЦА КОРИОЛИСОВЫХ СИЛ C(q, dq) — ФОРМУЛА (28)
# ============================================================

def coriolis_matrix(I_z3, dpsi, dxi):
    """
    Матрица кориолисовых и центробежных сил C(q, dq).

    Параметры:
        I_z3 (float): момент инерции звена 3 относительно оси z
        dpsi (float): угловая скорость звена 2 (dot_psi)
        dxi (float): угловая скорость звена 3 (dot_xi)

    Возвращает:
        np.ndarray: матрица 3x3

    Формула (28):
        C(q, dq) = [[0,      0,           0     ],
                    [0,      0,     -I_z3·dot_xi],
                    [0,  I_z3·dot_psi,     0     ]]

    Пример:
        C = coriolis_matrix(0.05, 0.5, 0.3)
        C[1, 2]
        -0.015
        C[2, 1]
        0.025
    """
    C = np.array([
        [0, 0, 0],
        [0, 0, -I_z3 * dxi],
        [0, I_z3 * dpsi, 0]
    ], dtype=float)
    return C


# ============================================================
# 3. ВЕКТОР ГРАВИТАЦИОННЫХ СИЛ G(q) — ФОРМУЛА (29)
# ============================================================

def gravity_vector(m1, m2, m3, g=9.81):
    """
    Вектор гравитационных сил G(q).

    Параметры:
        m1, m2, m3 (float): массы звеньев
        g (float): ускорение свободного падения (по умолчанию 9.81)

    Возвращает:
        np.ndarray: вектор 3x1

    Формула из отчёта (29):
        G(q) = [m1·g/2 + (m2 + m3)·g, 0, 0]^T

    Примечание:
        Вектор не зависит от q, так как все звенья находятся
        на фиксированной высоте z = z0.

    Пример:
        G = gravity_vector(1.0, 0.5, 0.3)
        G[0]
        12.757999999999999  # (1.0*9.81/2 + (0.5+0.3)*9.81)
    """
    G = np.array([
        m1 * g / 2 + (m2 + m3) * g,
        0.0,
        0.0
    ], dtype=float)
    return G


# ============================================================
# 4. ПОЛНОЕ УРАВНЕНИЕ ДИНАМИКИ — ФОРМУЛА (26)
# ============================================================

def compute_torques(m1, m2, m3, I_z2, I_z3, q, dq, ddq, g=9.81):
    """
    Вычисляет обобщённые силы tau по уравнению динамики (26).

    Параметры:
        m1, m2, m3 (float): массы звеньев
        I_z2, I_z3 (float): моменты инерции
        q (array): обобщённые координаты [z, psi, xi]
        dq (array): обобщённые скорости [dz, dpsi, dxi]
        ddq (array): обобщённые ускорения [ddz, ddpsi, ddxi]
        g (float): ускорение свободного падения

    Возвращает:
        np.ndarray: вектор обобщённых сил tau = [F_z, M_psi, M_xi]

    Формула (26):
        tau = M(q)·ddq + C(q, dq)·dq + G(q)

    Пример:
        q = np.array([0.5, 0.785, 0.524])
        dq = np.array([0.0, 0.0, 0.0])
        ddq = np.array([0.0, 0.5, 0.3])
        tau = compute_torques(1.0, 0.5, 0.3, 0.1, 0.05, q, dq, ddq)
        tau.shape
        (3,)
    """
    # Вычисляем матрицы
    M = inertia_matrix(m1, m2, m3, I_z2, I_z3)
    C = coriolis_matrix(I_z3, dq[1], dq[2])
    G = gravity_vector(m1, m2, m3, g)

    # Уравнение динамики (26)
    tau = M @ ddq + C @ dq + G

    return tau


# ============================================================
# 5. УПРОЩЁННЫЕ ФУНКЦИИ
# ============================================================

def compute_torques_with_names(m1, m2, m3, I_z2, I_z3,
                               z, psi, xi,
                               dz, dpsi, dxi,
                               ddz, ddpsi, ddxi,
                               g=9.81):
    """
    Упрощённая версия compute_torques с именованными параметрами.

    Параметры:
        m1, m2, m3 (float): массы звеньев
        I_z2, I_z3 (float): моменты инерции
        z, psi, xi (float): обобщённые координаты
        dz, dpsi, dxi (float): обобщённые скорости
        ddz, ddpsi, ddxi (float): обобщённые ускорения
        g (float): ускорение свободного падения

    Возвращает:
        dict: {'F_z': float, 'M_psi': float, 'M_xi': float}

    Пример:
         tau = compute_torques_with_names(1.0, 0.5, 0.3, 0.1, 0.05,
                                           0.5, 0.785, 0.524,
                                           0.0, 0.0, 0.0,
                                           0.0, 0.5, 0.3)
         tau['M_psi']
        0.075
    """
    q = np.array([z, psi, xi])
    dq = np.array([dz, dpsi, dxi])
    ddq = np.array([ddz, ddpsi, ddxi])

    tau = compute_torques(m1, m2, m3, I_z2, I_z3, q, dq, ddq, g)

    return {
        'F_z': tau[0],
        'M_psi': tau[1],
        'M_xi': tau[2]
    }


def compute_torques_static(m1, m2, m3, g=9.81):
    """
    Вычисляет статические усилия (при q=0, dq=0, ddq=0).

    В статике уравнение (26) сводится к:
        tau = G(q)

    Возвращает:
        dict: {'F_z': float, 'M_psi': float, 'M_xi': float}

    Пример:
         tau = compute_torques_static(1.0, 0.5, 0.3)
         tau['F_z']
        12.758
         tau['M_psi']
        0.0
         tau['M_xi']
        0.0
    """
    G = gravity_vector(m1, m2, m3, g)
    return {
        'F_z': G[0],
        'M_psi': G[1],
        'M_xi': G[2]
    }


def compute_torques_dynamic(m1, m2, m3, I_z2, I_z3,
                            z, psi, xi,
                            dz, dpsi, dxi,
                            ddz, ddpsi, ddxi,
                            g=9.81):
    """
    Аналог compute_torques_with_names, но возвращает numpy array.

    Используется для численного интегрирования.

    Возвращает:
        np.ndarray: [F_z, M_psi, M_xi]
    """
    q = np.array([z, psi, xi])
    dq = np.array([dz, dpsi, dxi])
    ddq = np.array([ddz, ddpsi, ddxi])

    return compute_torques(m1, m2, m3, I_z2, I_z3, q, dq, ddq, g)


# ============================================================
# 6. ПРОВЕРКА ОГРАНИЧЕНИЙ ДЛЯ ДИНАМИКИ
# ============================================================

def check_velocity_constraints(dz, dpsi, dxi,
                               dz_max=1.0, dpsi_max=2.0, dxi_max=2.0):
    """
    Проверка ограничений на скорости (Таблица 1 из отчёта).

    Параметры:
        dz, dpsi, dxi (float): текущие скорости
        dz_max, dpsi_max, dxi_max (float): максимальные скорости

    Возвращает:
        tuple: (ok, messages)
    """
    messages = []
    ok = True

    if abs(dz) > dz_max:
        ok = False
        messages.append(f"Скорость dz={dz:.3f} превышает максимум {dz_max:.3f}")

    if abs(dpsi) > dpsi_max:
        ok = False
        messages.append(f"Скорость dpsi={dpsi:.3f} превышает максимум {dpsi_max:.3f}")

    if abs(dxi) > dxi_max:
        ok = False
        messages.append(f"Скорость dxi={dxi:.3f} превышает максимум {dxi_max:.3f}")

    return ok, messages


def check_acceleration_constraints(ddz, ddpsi, ddxi,
                                   ddz_max=2.0, ddpsi_max=5.0, ddxi_max=5.0):
    """
    Проверка ограничений на ускорения (Таблица 1 из отчёта).

    Параметры:
        ddz, ddpsi, ddxi (float): текущие ускорения
        ddz_max, ddpsi_max, ddxi_max (float): максимальные ускорения

    Возвращает:
        tuple: (ok, messages)
    """
    messages = []
    ok = True

    if abs(ddz) > ddz_max:
        ok = False
        messages.append(f"Ускорение ddz={ddz:.3f} превышает максимум {ddz_max:.3f}")

    if abs(ddpsi) > ddpsi_max:
        ok = False
        messages.append(f"Ускорение ddpsi={ddpsi:.3f} превышает максимум {ddpsi_max:.3f}")

    if abs(ddxi) > ddxi_max:
        ok = False
        messages.append(f"Ускорение ddxi={ddxi:.3f} превышает максимум {ddxi_max:.3f}")

    return ok, messages


def check_torque_constraints(F_z, M_psi, M_xi,
                             F_z_max=10.0, M_psi_max=5.0, M_xi_max=5.0):
    """
    Проверка ограничений на усилия (из конструктивных ограничений).

    Параметры:
        F_z, M_psi, M_xi (float): текущие усилия
        F_z_max, M_psi_max, M_xi_max (float): максимальные усилия

    Возвращает:
        tuple: (ok, messages)
    """
    messages = []
    ok = True

    if abs(F_z) > F_z_max:
        ok = False
        messages.append(f"Сила F_z={F_z:.3f} превышает максимум {F_z_max:.3f}")

    if abs(M_psi) > M_psi_max:
        ok = False
        messages.append(f"Момент M_psi={M_psi:.3f} превышает максимум {M_psi_max:.3f}")

    if abs(M_xi) > M_xi_max:
        ok = False
        messages.append(f"Момент M_xi={M_xi:.3f} превышает максимум {M_xi_max:.3f}")

    return ok, messages