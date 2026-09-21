import math


def exact_solution(x):
    return (1 + x) * math.exp(-x * x)


def f(x, y, z):
    return -4 * x * z - (4 * x * x + 2) * y


def euler_method(y0, z0, interval, h):
    x = make_grid(interval, h)
    y = [y0]
    z = z0
    for i in range(len(x) - 1):
        y.append(y[i] + h * z)
        z += h * f(x[i], y[i], z)
    return x, y


def make_grid(interval, h):
    l, r = interval
    if not all(math.isfinite(v) for v in (l, r, h)) or h <= 0 or r <= l:
        raise ValueError("Expected a finite interval l < r and h > 0")
    n = round((r - l) / h)
    if n < 1 or not math.isclose(n * h, r - l, rel_tol=1e-12, abs_tol=1e-14):
        raise ValueError("The step must divide the interval into an integer number of parts")
    return [l + i * h for i in range(n)] + [r]


def runge_romberg(coarse, fine, order):
    """Оценка погрешности на мелкой сетке в общих узлах."""
    if len(fine) != 2 * len(coarse) - 1:
        raise ValueError("Expected nested grids with step ratio 2")
    return max(abs(a - fine[2*i]) for i, a in enumerate(coarse)) / (2**order - 1)


def runge_kutta_method(y0, z0, interval, h):
    x = make_grid(interval, h)
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
    with open('input.txt', encoding='utf-8') as file:
        data = [list(map(float, line.split())) for line in file if line.strip()]
    y0, z0, interval, h = data[0][0], data[1][0], data[2], data[3][0]
    methods = [('Euler', euler_method, 1), ('Runge-Kutta', runge_kutta_method, 4),
               ('Adams', adams_method, 4)]
    results = []
    for name, method, order in methods:
        x, y, *_ = method(y0, z0, interval, h)
        xf, yf, *_ = method(y0, z0, interval, h / 2)
        results.append((name, order, y, yf, runge_romberg(y, yf, order)))

    with open('output.txt', 'w', encoding='utf-8') as file:
        file.write("Cauchy problem, variant 6\n")
        file.write("Equation: y'' + 4*x*y' + (4*x^2 + 2)*y = 0\n")
        file.write("Exact solution: y = (1 + x)*exp(-x^2)\n")
        file.write(f"y({interval[0]:g}) = {y0:g}, y'({interval[0]:g}) = {z0:g}\n")
        file.write(f"Interval = {interval}, h = {h:.12g}, h/2 = {h/2:.12g}\n\n")
        file.write("x        Euler          Runge-Kutta    Adams          Exact\n")
        for i, t in enumerate(x):
            values = [result[2][i] for result in results] + [exact_solution(t)]
            file.write(f"{t:.6f} " + ' '.join(f'{v: .10f}' for v in values) + '\n')
        file.write('\nMaximum absolute errors over each full grid:\n')
        for name, order, y, yf, rr in results:
            coarse_error = max(abs(v - exact_solution(t)) for t, v in zip(x, y))
            fine_error = max(abs(v - exact_solution(t)) for t, v in zip(xf, yf))
            file.write(f'{name}, p={order}:\n')
            file.write(f'  max error (h)   = {coarse_error:.10e}\n')
            file.write(f'  max error (h/2) = {fine_error:.10e}\n')
            file.write(f'  Runge-Romberg (h/2, common nodes) = {rr:.10e}\n')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dense = [interval[0] + (interval[1] - interval[0])*i/500 for i in range(501)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(dense, [exact_solution(t) for t in dense], 'k-', label='Exact')
    for name, _, y, _, _ in results:
        axes[0].plot(x, y, 'o--', markersize=3, label=name)
        axes[1].semilogy(x[1:], [max(abs(v-exact_solution(t)), 1e-16) for t,v in zip(x[1:],y[1:])], 'o-', markersize=3, label=name)
    axes[0].set(xlabel='x', ylabel='y(x)', title=f'Cauchy problem: variant 6, h={h:g}')
    axes[1].set(xlabel='x', ylabel='Absolute error', title='Error on the coarse grid')
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig('solution.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
