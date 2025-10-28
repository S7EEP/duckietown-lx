from typing import Tuple

import numpy as np


def get_motor_left_matrix(shape: Tuple[int, int]) -> np.ndarray:
    # TODO: write your function instead of this one
    res = np.zeros(shape=shape, dtype="float32")
    # Generar valores de 0 a 1 para la parte izquierda
    left_gradient = np.linspace(0, 1, shape[1] // 2)
    res[:, : shape[1] // 2] = left_gradient
    
    # Generar valores de 0 a -1 para la parte derecha
    right_gradient = np.linspace(0, -1, shape[1] // 2)
    res[:shape[0] // 2, shape[1] // 2:] = right_gradient
    # ---
    res[shape[0] // 2:, :shape[1] // 2] = np.linspace(1, 0.5, shape[1] // 2)
    res[shape[0] // 2:, shape[1] // 2:] = np.linspace(-1, -0.5, shape[1] // 2)
    #res[200:400, 200:400] = 0
    res/np.sum(res)
    return res


def get_motor_right_matrix(shape: Tuple[int, int]) -> np.ndarray:
    # TODO: write your function instead of this one
    res = np.zeros(shape=shape, dtype="float32")
    # Generar valores de -1 a 0 para la parte izquierda
    left_gradient = np.linspace(-1, 0, shape[1] // 2)
    res[:shape[0] // 2, :shape[1] // 2] = left_gradient
    
    # Generar valores de 1 a 0 para la parte derecha
    right_gradient = np.linspace(1, 0, shape[1] // 2)
    res[:, shape[1] // 2:] = right_gradient
    # ---
    res[shape[0] // 2:, shape[1] // 2:] = np.linspace(0.5, 1, shape[1] // 2)
    res[shape[0] // 2:, :shape[1] // 2] = np.linspace(-0.5, -1, shape[1] // 2)
    #res[200:400, 200:400] = 0
    res/np.sum(res)
    return res