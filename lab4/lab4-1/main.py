import math


def exact_solution(x):
    return math.cos(math.sin(x)) + math.sin(math.cos(x))


def f(x, y, z):
    h = 1e-5
    return (exact_solution(x + h) - 2 * exact_solution(x) + exact_solution(x - h)) / h ** 2


def euler_method(y0, z0, interval, h):
    l, r = interval
    x = [l + i * h for i in range(int(round((r - l) / h)) + 1)]
    y = [y0]
    z = z0
    for i in range(len(x) - 1):
        z += h * f(x[i], y[i], z)
        y.append(y[i] + h * z)
    return x, y


def runge_kutta_method(y0, z0, interval, h):
    l, r = interval
    x = [l + i * h for i in range(int(round((r - l) / h)) + 1)]
    y = [y0]
    z = [z0]
    for i in range(len(x) - 1):
        k1 = h * z[i]
        l1 = h * f(x[i], y[i], z[i])
        k2 = h * (z[i] + l1 / 2)
        l2 = h * f(x[i] + h / 2, y[i] + k1 / 2, z[i] + l1 / 2)
        k3 = h * (z[i] + l2 / 2)
        l3 = h * f(x[i] + h / 2, y[i] + k2 / 2, z[i] + l2 / 2)
        k4 = h * (z[i] + l3)
        l4 = h * f(x[i] + h, y[i] + k3, z[i] + l3)
        y.append(y[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6)
        z.append(z[i] + (l1 + 2 * l2 + 2 * l3 + l4) / 6)
    return x, y, z


def adams_method(y0, z0, interval, h):
    x, y, z = runge_kutta_method(y0, z0, interval, h)
    y, z = y[:4], z[:4]
    for i in range(3, len(x) - 1):
        z.append(z[i] + h * (55 * f(x[i], y[i], z[i]) - 59 * f(x[i - 1], y[i - 1], z[i - 1]) + 37 * f(x[i - 2], y[i - 2], z[i - 2]) - 9 * f(x[i - 3], y[i - 3], z[i - 3])) / 24)
        y.append(y[i] + h * (55 * z[i] - 59 * z[i - 1] + 37 * z[i - 2] - 9 * z[i - 3]) / 24)
    return x, y


def main():
    with open('input.txt', 'r') as file:
        data = [list(map(float, line.split())) for line in file.readlines()]
    y0, z0 = data[0][0], data[1][0]
    interval = data[2]
    h = data[3][0]

    x_e, y_e = euler_method(y0, z0, interval, h)
    x_r, y_r, _ = runge_kutta_method(y0, z0, interval, h)
    x_a, y_a = adams_method(y0, z0, interval, h)

    with open('output.txt', 'w') as file:
        file.write("Cauchy problem, variant 6\n")
        file.write("Exact solution: y = cos(sin(x)) + sin(cos(x))\n\n")
        file.write("x Euler Runge-Kutta Adams Exact\n")
        for i in range(len(x_r)):
            file.write(f"{x_r[i]:.2f} {y_e[i]:.6f} {y_r[i]:.6f} {y_a[i]:.6f} {exact_solution(x_r[i]):.6f}\n")


if __name__ == "__main__":
    main()
