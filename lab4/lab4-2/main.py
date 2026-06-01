import math


def f(x, y, z):
    return 2 * (1 + math.tan(x) ** 2) * y


def exact_solution(x):
    return math.tan(x)


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


def shooting_method(y0, y1, interval, h, eps=1e-6):
    n0, n1 = 0.0, 2.0
    for it in range(100):
        _, y_prev, _ = runge_kutta_method(y0, n0, interval, h)
        x, y_cur, z_cur = runge_kutta_method(y0, n1, interval, h)
        if abs(y_cur[-1] - y1) < eps:
            return x, y_cur, z_cur, n1, it
        n0, n1 = n1, n1 - (y_cur[-1] - y1) * (n1 - n0) / (y_cur[-1] - y_prev[-1])
    return x, y_cur, z_cur, n1, 100


def finite_difference_method(y0, y1, interval, h):
    l, r = interval
    n = int(round((r - l) / h))
    x = [l + i * h for i in range(n + 1)]
    a = [0.0] * (n + 1)
    b = [0.0] * (n + 1)
    c = [0.0] * (n + 1)
    d = [0.0] * (n + 1)
    b[0], d[0] = 1, y0
    b[n], d[n] = 1, y1
    for i in range(1, n):
        a[i] = 1 / h ** 2
        b[i] = -2 / h ** 2 - 2 * (1 + math.tan(x[i]) ** 2)
        c[i] = 1 / h ** 2

    p = [0.0] * (n + 1)
    q = [0.0] * (n + 1)
    p[0] = -c[0] / b[0]
    q[0] = d[0] / b[0]
    for i in range(1, n + 1):
        den = b[i] + a[i] * p[i - 1]
        p[i] = -c[i] / den if i < n else 0
        q[i] = (d[i] - a[i] * q[i - 1]) / den
    y = [0.0] * (n + 1)
    y[n] = q[n]
    for i in range(n - 1, -1, -1):
        y[i] = p[i] * y[i + 1] + q[i]
    return x, y


def main():
    with open('input.txt', 'r') as file:
        data = [list(map(float, line.split())) for line in file.readlines()]
    y0, y1 = data[0][0], data[1][0]
    interval = data[2]
    h = data[3][0]

    x_s, y_s, _, n, it = shooting_method(y0, y1, interval, h)
    x_f, y_f = finite_difference_method(y0, y1, interval, h)

    with open('output.txt', 'w') as file:
        file.write("Boundary problem, variant 6\n")
        file.write("Equation: y'' - 2*(1 + tg(x)^2)*y = 0\n")
        file.write("Exact solution: y = tg(x)\n\n")
        file.write(f"Shooting parameter = {n}, iterations = {it}\n\n")
        file.write("x Shooting Finite-difference Exact\n")
        for i in range(len(x_f)):
            file.write(f"{x_f[i]:.2f} {y_s[i]:.6f} {y_f[i]:.6f} {exact_solution(x_f[i]):.6f}\n")


if __name__ == "__main__":
    main()
