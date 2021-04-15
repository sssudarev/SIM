from random import *
from pprint import pprint
import numpy as np
from math import sqrt
import sys

from scipy.stats import f
from scipy.stats import t as t_check

stepM = 3  # шаг, на який буде збільшуватись М, якщо дисперсія неоднорідна

M, n, D = 3, 4, 4


def lab3(m , N, d):
    x = [[-30, 0], [-15, 35], [-30, -25]]
    normalizedX = [[1, -1, -1, -1], [1, -1, 1, 1], [1, 1, -1, 1], [1, 1, 1, -1]]

    tran1 = [list(i) for i in zip(*normalizedX)]

    x_min_max = [round(sum(x[i][k] for i in range(3)) / 3, 3) for k in range(2)]

    y_min_max = [int(200 + x_min_max[i]) for i in range(2)]

    print('\nЗадана матриця Х:\n', x, '\nНормована матриця Х:')
    pprint(normalizedX, width=17)

    mat_Y = [[randint(y_min_max[0], y_min_max[1]) for i in range(3)] for k in range(4)]
    print('\nXср min and max:\n', x_min_max, '\nY min and max:\n', y_min_max, '\nМатриця Y:')
    pprint(mat_Y, width=17)

    mat_serY = [round(sum(mat_Y[k1])/3, 3) for k1 in range(4)]
    print('\nСередні значення Y:\n', mat_serY, '\nМатриця Х:')

    mat_X = [[-25, -30, -5], [-25, 45, 5], [-5, -30, 5], [-5, 45, -5]]
    pprint(mat_X, width=17)

    mx = [round(sum(mat_X[i][k] for i in range(4))/4, 3) for k in range(3)]
    my = round(sum(mat_serY)/4, 3)
    print('\nЗначення mxi:\n', mx, '\nЗначення my:\n', my)

    tran = [list(i) for i in zip(*mat_X)]
    ai = [round(sum(tran[k][i] * mat_serY[i] for i in range(4)) / 4, 3) for k in range(3)]
    aii = [round(sum(tran[k][i]**2 for i in range(4)) / 4, 3) for k in range(3)]
    print('\nЗначення ai:\n', ai, '\nЗначення aii:\n', aii)

    a12, a21, a13, a31, a23, a32 = 0, 0, 0, 0, 0, 0

    for i in range(len(tran[0])):
        a12 += tran[0][i] * tran[1][i] / 4

    for i in range(len(tran[0])):
        a13 += tran[0][i] * tran[2][i] / 4

    for i in range(len(tran[1])):
        a23 += tran[1][i] * tran[2][i] / 4

    a21 = a12
    a31 = a13
    a32 = a23

    denominator = np.linalg.det(np.array([
        [1, mx[0], mx[1], mx[2] ],
        [mx[0], aii[0], a12, a13],
        [mx[1], a12, aii[1], a32],
        [mx[2], a13, a23, aii[2]]]))

    b = []
    b.append(np.linalg.det(np.array([[my, mx[0], mx[1], mx[2]], [ai[0], aii[0], a12, a13], [ai[1], a12, aii[1], a32], [ai[2], a13, a23, aii[2]]])) / denominator)
    b.append(np.linalg.det(np.array([[1, my, mx[1], mx[2]], [mx[0], ai[0], a12, a13], [mx[1], ai[1], aii[1], a32], [mx[2], ai[2], a23, aii[2]]])) / denominator)
    b.append(np.linalg.det(np.array([[1, mx[0], my, mx[2]], [mx[0], aii[0], ai[0], a13], [mx[1], a12, ai[1], a32], [mx[2], a13, ai[2], aii[2]]])) / denominator)
    b.append(np.linalg.det(np.array([[1, mx[0], mx[1], my], [mx[0], aii[0], a12, ai[0]], [mx[1], a12, aii[1], ai[1]], [mx[2], a13, a23, ai[2]]])) / denominator)

    checking = [round(b[0] + b[1] * tran[0][i] + b[2] * tran[1][i] + b[3] * tran[2][i], 3) for i in range(4)]

    print("\nПеревірка порівнянням з середніми значеннями Y:\n", checking)
    mat_disY = [round(sum([((k1 - mat_serY[j]) ** 2) for k1 in mat_Y[j]]) / m, 3) for j in range(4)]
    print("\nДисперсії в рядках:\n", mat_disY)
    print('\nПеревірка однорідності дисперсії за критерієм Кохрена:\n')

    if max(mat_disY)/sum(mat_disY) < 0.7679:
        print('\nДисперсія однорідна')
    else:
        print('\nДисперсія неоднорідна, розпочинаємо спочатку!')
        lab3(M + stepM, n, D)


    print('\nПеревірка значущості коефіцієнтів за критерієм Стьюдента:')

    S2b = sum(mat_disY) / N
    S2bs = S2b / (m * N)
    Sbs = sqrt(S2bs)

    print('\nSbs:\n', round(Sbs, 3))

    bb = [round(sum(mat_serY[k] * tran1[i][k] for k in range(N))/N, 3) for i in range(N)]
    t = [round(abs(bb[i])/Sbs, 3) for i in range(N)]

    print('\nbi:\n', bb, '\nti:\n', t)

    f1, f2 = m - 1, N
    f3 = f1 * f2

    for i in range(N):
        if t[i] < t_check.ppf(q=0.975, df=f3):
            b[i] = 0
            d -= 1
            print('Виключаємо з рівняння коефіціент b', i)

    y_reg = [round(b[0] + b[1] * mat_X[i][0] + b[2] * mat_X[i][1] + b[3] * mat_X[i][2], 3) for i in range(4)]

    print('Значення рівнянь регресій:\n', y_reg)
    print('\nПеревірка адекватності за критерієм Фішера:')

    f4 = N - d
    Sad = (m / (N - d)) * int(sum(y_reg[i] - mat_serY[i] for i in range(N))**2)

    Fp = Sad / S2b

    print('\nКількість значимих коефіціентів:\n', d, '\nFp:\n', round(Fp, 3))

    if Fp > f.ppf(q=0.95, dfn=f4, dfd=f3):
        print('Рівняння регресії неадекватно оригіналу при рівні значимості 0.05')
    else:
        print('Рівняння регресії адекватно оригіналу при рівні значимості 0.05')


lab3(M, n, D)
