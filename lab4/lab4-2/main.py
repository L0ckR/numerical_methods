import math


def f(x, y, z):
    return 2 * (1 + math.tan(x) ** 2) * y


def exact_solution(x):
    return -math.tan(x)


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


def shooting_method(y0, y1, interval, h, eps=1e-12, max_iter=100):
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    n0, n1 = 0.0, -2.0
    _, previous, _ = runge_kutta_method(y0, n0, interval, h)
    residual0 = previous[-1] - y1
    for it in range(max_iter + 1):
        x, y, z = runge_kutta_method(y0, n1, interval, h)
        residual1 = y[-1] - y1
        if abs(residual1) <= eps:
            return x, y, z, n1, it
        denominator = residual1 - residual0
        if denominator == 0:
            raise ValueError("Shooting secant has zero denominator")
        n0, n1 = n1, n1 - residual1 * (n1 - n0) / denominator
        residual0 = residual1
    raise RuntimeError("Shooting method did not converge")


def finite_difference_method(y0, y1, interval, h):
    x = make_grid(interval, h)
    n = len(x) - 1
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
    with open('input.txt', encoding='utf-8') as file:
        data = [list(map(float, line.split())) for line in file if line.strip()]
    y0, y1, interval, h = data[0][0], data[1][0], data[2], data[3][0]
    x, ys, zs, slope, iterations = shooting_method(y0, y1, interval, h)
    xf, ysf, *_ = shooting_method(y0, y1, interval, h / 2)
    _, yd = finite_difference_method(y0, y1, interval, h)
    _, ydf = finite_difference_method(y0, y1, interval, h / 2)
    results = [('Shooting', 4, ys, ysf), ('Finite difference', 2, yd, ydf)]
    with open('output.txt', 'w', encoding='utf-8') as file:
        file.write("Boundary problem, variant 6\n")
        file.write("Equation: y'' - 2*(1 + tan(x)^2)*y = 0\n")
        file.write("Exact solution: y = -tan(x)\n")
        file.write("Conditions: y(0)=0, y(pi/6)=-sqrt(3)/3\n")
        file.write(f'Interval = {interval}\nh = {h:.12g}, h/2 = {h/2:.12g}\n')
        file.write(f"Shooting slope = {slope:.12g}, secant updates = {iterations}\n")
        file.write(f"Right boundary residual = {abs(ys[-1]-y1):.6e}\n\n")
        file.write("x        Shooting       Finite-difference Exact\n")
        for t, a, b in zip(x, ys, yd):
            file.write(f'{t:.6f} {a: .10f} {b: .10f} {exact_solution(t): .10f}\n')
        file.write('\nMaximum absolute errors over each full grid:\n')
        for name, order, y, yf in results:
            coarse_error = max(abs(v - exact_solution(t)) for t, v in zip(x, y))
            fine_error = max(abs(v - exact_solution(t)) for t, v in zip(xf, yf))
            file.write(f'{name}, p={order}:\n')
            file.write(f'  max error (h)   = {coarse_error:.10e}\n')
            file.write(f'  max error (h/2) = {fine_error:.10e}\n')
            file.write(f'  Runge-Romberg (h/2, common nodes) = {runge_romberg(y,yf,order):.10e}\n')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dense = [interval[0] + (interval[1] - interval[0])*i/500 for i in range(501)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(dense, [exact_solution(t) for t in dense], 'k-', label='Exact')
    for name, _, y, _ in results:
        axes[0].plot(x, y, 'o--', markersize=3, label=name)
        axes[1].semilogy(x[1:-1], [max(abs(v-exact_solution(t)), 1e-16) for t,v in zip(x[1:-1],y[1:-1])], 'o-', markersize=3, label=name)
    axes[0].set(xlabel='x', ylabel='y(x)', title='Boundary problem: variant 6')
    axes[1].set(xlabel='x', ylabel='Absolute error', title='Error at interior nodes')
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig('solution.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
