import random


def custom_append(old_list, new_element):  # ADDITIONAL TASK
    new_list = [0 for _ in range(len(old_list) + 1)]
    for j in range(len(old_list)):
        new_list[j] = old_list[j]
    new_list[-1] = new_element      
    return new_list


number_of_experiments = int(input("Введіть кількість експерементів:"))
a = [random.randint(1, 20) for _ in range(4)]  # список коефіцієнтів

print("\nКоефіцієнти:")
for i in range(len(a)):
    print(f'a{i} = {a[i]}')
x = [[random.randint(1, 20) for _ in range(3)] for _ in range(number_of_experiments)]  # матриця планування експ.

print(f'\nМатриця планування експерименту: {x}')

y = []  # значення функції відгуку

for exp in range(number_of_experiments):
    y = custom_append(y, (a[0] + a[1] * x[exp][0] + a[2] * x[exp][1] + a[3] * x[exp][2]))

print('\nЗначення функції відгуку:', y)

print('\nНульовий рівень факторів:')
x0 = []
for i in range(3):
    xi = [exp[i] for exp in x]
    x0i = (max(xi)+min(xi)) / 2
    x0 = custom_append(x0, x0i)
    print(f'x0{i+1} = {x0i}')

print('\nІнтервал зміни факторів:')

dx = []
for i in range(3):
    dxi = x0[i] - min([exp[i] for exp in x])
    dx = custom_append(dx, dxi)
    print(f'dx0{i + 1} = {dxi}')

yet = a[0] + a[1] * x0[0] + a[2] * x0[1] + a[3] * x0[2]
print(f'\nЕталонне значення функціїї відгуку y = {yet}')

xn = [[round((x[i][j] - x0[j]) / dx[j], 2) for j in range(3)] for i in range(number_of_experiments)]
print(f'\nНормалізована матриця планування: {xn}')

print(f'Критерій вибору оптимальності - min(Y) = {min(y)}, досягається при такій точці плану: {x[y.index(min(y))]}')