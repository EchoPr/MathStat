from itertools import combinations_with_replacement
from collections import Counter
import matplotlib.pyplot as plt
from math import floor, log
import numpy as np
import scipy

# Генерируем выборку
n = 25
data = np.random.exponential(size=n)

print(f"Выборка при {n = }:\n{data = }\n")

# Вариационный ряд
data.sort()

print(f"Вариацонный ряд:\n{data = }\n")

# Размах выборки
l = max(data) - min(data)
print(f"Размах выборки: {l = }")

# Медиана
print(f"Медиана: {(data[n // 2] + data[n // 2 - 1]) / 2 if n % 2 == 0 else data[n // 2]}")

# Мода
# print(Counter(data))
print(f"Мода: 26")

# Квартили
q1 = (52 + 56) / 2
q2 = (86 + 87) / 2
print(f"Квартили: {q1} и {q2}")

eps = q2 - q1
print(f"epsilon: {eps}")

print(f"min_boxplot {q1 - 1.5*eps}; max_boxplot: {q2 + 1.5*eps}\n")

# Коэф. асимметрии
gamma = (sum((data - data.mean()) ** 3) / n) / np.var(data) ** (3 / 2)
print(f"Коэфф ассиметрри {gamma = }\n")

# Эмпирическая функция распределения
_, axs = plt.subplots(1, 1, figsize=(10, 6))

axs.set_title("Эмпирическая функция распределения")
axs.set_xlabel("x")
axs.set_ylabel("F(x)")

axs.step(data,
         np.arange(1, n + 1) / n,
         label="F(x)")

plt.grid(which='minor', linestyle='--', linewidth=0.2)
plt.grid(which='major', linewidth=0.4)

plt.legend()
plt.show()

# Гистограмма
k = 1 + floor(log(n, 2))
delta = l / k

print(f"Данные гистограммы:\n{k = }\n{delta = }")

_, axs = plt.subplots(1, 1, figsize=(10, 6))

axs.set_title("Гистограмма выборки")
axs.set_xlabel("x")
axs.set_ylabel("[frequency]")

axs.hist(data,
         bins=k,
         label="F(x)",
         density=True)

plt.grid(which='minor', linestyle='--', linewidth=0.2)
plt.grid(which='major', linewidth=0.4)

plt.legend()
plt.show()

# Boxplot
_, axs = plt.subplots(1, 1, figsize=(10, 1))

axs.set_title("Boxplot")

axs.boxplot(data)

plt.grid(which='minor', linestyle='--', linewidth=0.2)
plt.grid(which='major', linewidth=0.4)

plt.show()

# Bootstrap
bootstrap_data = data[np.random.choice(n, size=(1000, n))]

_, axs = plt.subplots(1, 1, figsize=(10, 7))

axs.set_title("ЦПТ vs Bootstrap")
axs.set_xlabel("x")
axs.set_ylabel("Плотность вероятности")

axs.hist(np.mean(bootstrap_data, axis=1),
          bins=1+int(np.log2(1000)),
          label="Bootstrap",
          density=True)

x_data = np.linspace(1 - 3 * 1/5, 1 + 3 * 1/5, 1000)
axs.plot(x_data,
         scipy.stats.norm.pdf(x_data, 1, 1/5),
         label="Нормальное распределение",
         color='red')

plt.grid(which='minor', linestyle='--', linewidth=0.2)
plt.grid(which='major', linewidth=0.4)

plt.legend()
plt.show()

#(sum((data - data.mean()) ** 3) / n) / np.var(data) ** (3 / 2)
assim = []
for i in range(1000):
    dat = np.random.choice(data, 25)
    assim.append((sum((dat - dat.mean()) ** 3) / n) / np.var(dat) ** (3 / 2))

_, axs = plt.subplots(1, 1, figsize=(10, 7))

axs.set_title("bootstrap: Коэффициент асимметрии")
axs.set_xlabel("ɣ")
axs.set_ylabel("Плотность вероятности")

axs.hist(assim,
          bins=1 + int(np.log2(1000)),
          label="Плотность вероятности",
          density=True)

plt.grid(which='minor', linestyle='--', linewidth=0.2)
plt.grid(which='major', linewidth=0.4)

plt.legend()
plt.show()

gamma_less_one = np.sum([1 if assim[i] < 1 else 0 for i in range(1000)])/1000

print(f"Оценка вероятности gamma < 1\n{gamma_less_one = }")
