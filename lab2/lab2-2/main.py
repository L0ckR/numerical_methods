"""Variant 6: x1 - cos(x2) = 1; x2 - log10(x1 + 1) = 3."""

import math
from pathlib import Path

a = 3.0


def f1(x1, x2):
    return x1 - math.cos(x2) - 1


def f2(x1, x2):
    return x2 - math.log10(x1 + 1) - a


def jacobian(x1, x2):
    return [[1.0, math.sin(x2)], [-1 / ((x1 + 1) * math.log(10)), 1.0]]


def phi1(x1, x2):
    return 1 + math.cos(x2)


def phi2(x1, x2):
    return a + math.log10(x1 + 1)


def residual_norm(x):
    return max(abs(f1(*x)), abs(f2(*x)))


def trig_range(function, left, right, critical_offset):
    """Exact endpoint/critical-point extrema, without sampling the interval."""
    points = [left, right]
    first = math.ceil((left-critical_offset)/math.pi)
    last = math.floor((right-critical_offset)/math.pi)
    points.extend(critical_offset + k*math.pi for k in range(first, last+1))
    values = [function(x) for x in points]
    return min(values), max(values)


def convergence_data(intervals, eps, max_iter):
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    if not isinstance(max_iter, int) or max_iter <= 0:
        raise ValueError("max_iter must be a positive integer")
    if len(intervals) != 2 or any(len(pair) != 2 for pair in intervals):
        raise ValueError("Two coordinate intervals are required")
    if any(not all(math.isfinite(x) for x in pair) or not 0 <= pair[0] < pair[1]
           for pair in intervals):
        raise ValueError("Intervals must have finite nonnegative increasing endpoints")
    (a1, b1), (a2, b2) = intervals
    sin_min, sin_max = trig_range(math.sin, a2, b2, math.pi / 2)
    cos_min, cos_max = trig_range(math.cos, a2, b2, 0)
    q = max(abs(sin_min), abs(sin_max), 1/((a1+1)*math.log(10)))
    if q >= 1:
        raise ValueError("The fixed-point map must satisfy q < 1 on the whole rectangle")
    if not (a1 <= 1+cos_min <= 1+cos_max <= b1 and
            a2 <= phi2(a1, a2) <= phi2(b1, b2) <= b2):
        raise ValueError("The fixed-point map must map the rectangle into itself")
    return q


def simple_iteration_method(intervals, eps, max_iter=1000, history=None):
    q = convergence_data(intervals, eps, max_iter)
    x1, x2 = [sum(pair)/2 for pair in intervals]
    for iters in range(1, max_iter + 1):
        x_next = [phi1(x1, x2), phi2(x1, x2)]
        step = max(abs(x_next[0]-x1), abs(x_next[1]-x2))
        error_bound = q / (1-q) * step
        residual = residual_norm(x_next)
        if history is not None:
            history.append(dict(iteration=iters, x=x_next, step=step,
                                error_bound=error_bound, residual=residual))
        if error_bound <= eps and residual <= eps:
            return x_next, iters
        x1, x2 = x_next
    raise RuntimeError("Simple iteration did not converge within max_iter")


def newton_method(intervals, eps, max_iter=1000, history=None):
    q = convergence_data(intervals, eps, max_iter)
    x1, x2 = [sum(pair)/2 for pair in intervals]
    for iters in range(1, max_iter + 1):
        j = jacobian(x1, x2)
        det = j[0][0]*j[1][1] - j[0][1]*j[1][0]
        if abs(det) < 1e-14:
            raise ArithmeticError("Newton Jacobian is singular")
        dx1 = (-f1(x1, x2)*j[1][1] + j[0][1]*f2(x1, x2))/det
        dx2 = (-j[0][0]*f2(x1, x2) + f1(x1, x2)*j[1][0])/det
        x_next = [x1+dx1, x2+dx2]
        if any(not math.isfinite(x) or not left <= x <= right
               for x, (left, right) in zip(x_next, intervals)):
            raise ArithmeticError("Newton iterate left the validated rectangle")
        residual = residual_norm(x_next)
        # F(x)=x-phi(x), hence ||x-x*|| <= ||F(x)||/(1-q) on the rectangle.
        error_bound = residual / (1-q)
        if history is not None:
            history.append(dict(iteration=iters, x=x_next, step=max(abs(dx1), abs(dx2)),
                                error_bound=error_bound, residual=residual))
        if error_bound <= eps and residual <= eps:
            return x_next, iters
        x1, x2 = x_next
    raise RuntimeError("Newton method did not converge within max_iter")


def convergence_plot(arguments, reference, folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    fig, ax = plt.subplots(figsize=(7.5, 4.1), layout="constrained")
    for method, title, color in ((simple_iteration_method, "Простая итерация", "#2166ac"),
                                 (newton_method, "Ньютон", "#b2182b")):
        history = []
        method(*arguments, history=history)
        actual = [max(float(np.max(np.abs(np.asarray(h["x"])-reference))), 1e-16) for h in history]
        bounds = [max(h["error_bound"], 1e-16) for h in history]
        steps = np.arange(1, len(history)+1)
        ax.semilogy(steps, actual, "o-", color=color, label=title + ": ошибка")
        ax.semilogy(steps, bounds, ".--", color=color, alpha=.7, label=title + ": оценка")
    ax.axhline(arguments[-1], color="#555555", linestyle=":", label="Заданная точность")
    ax.set(xlabel="Номер итерации k", ylabel="Абсолютная ошибка / её верхняя оценка",
           title="Зависимость ошибки от числа итераций")
    ax.legend(fontsize=8)
    fig.savefig(folder/"convergence.png", dpi=180)
    plt.close(fig)


def draw_plots():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Rectangle
    from scipy.optimize import root
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25})
    folder = Path(__file__).resolve().parent
    values = [list(map(float, row.split())) for row in (folder/"input.txt").read_text().splitlines() if row.strip()]
    intervals, eps = values[:2], values[2][0]
    initial = [sum(pair)/2 for pair in intervals]
    reference = root(lambda x: [x[0]-np.cos(x[1])-1, x[1]-np.log10(x[0]+1)-3],
                     initial, tol=1e-12)
    if not reference.success:
        raise RuntimeError(reference.message)
    fig, ax = plt.subplots(figsize=(7.5, 4.1), layout="constrained")
    x1 = np.linspace(-.005, .06, 400)
    x2 = np.linspace(2.995, 3.055, 400)
    ax.plot(1+np.cos(x2), x2, label=r"$x_1=1+\cos x_2$", color="#2166ac")
    ax.plot(x1, 3+np.log10(x1+1), label=r"$x_2=3+\lg(x_1+1)$", color="#b2182b")
    (a1, b1), (a2, b2) = intervals
    ax.add_patch(Rectangle((a1, a2), b1-a1, b2-a2, color="#2166ac", alpha=.09,
                           label="Область D"))
    ax.scatter(*initial, color="#666666", marker="x", label="Начальное приближение", zorder=5)
    ax.scatter(*reference.x, color="black", s=28, label="Решение", zorder=6)
    ax.set(xlabel=r"$x_1$", ylabel=r"$x_2$", xlim=(-.005, .06), ylim=(2.995, 3.055),
           title="2.2. Пересечение кривых и начальная область")
    ax.legend(fontsize=8, loc="upper right")
    fig.savefig(folder/"roots.png", dpi=180)
    plt.close(fig)
    convergence_plot((intervals, eps), reference.x, folder)


def main():
    folder = Path(__file__).resolve().parent
    data = [list(map(float, line.split()))
            for line in (folder / "input.txt").read_text().splitlines() if line.strip()]
    if len(data) != 3 or any(len(pair) != 2 for pair in data[:2]) or len(data[2]) != 1:
        raise ValueError("Expected two coordinate intervals followed by eps")
    intervals, eps = data[:2], data[2][0]
    q = convergence_data(intervals, eps, 1000)
    lines = ["Variant: 6; a = 3", "System:",
             "x1 - cos(x2) = 1", "x2 - log10(x1 + 1) = 3",
             f"Intervals: {intervals}", f"Epsilon: {eps}", f"q = {q}", ""]
    for name, method in (("Simple iteration method", simple_iteration_method),
                         ("Newton method", newton_method)):
        history = []
        result, iters = method(intervals, eps, history=history)
        lines.extend([name + ":", f"Initial approximation: {[sum(pair)/2 for pair in intervals]}",
                      f"x = [{result[0]:.15g}, {result[1]:.15g}]", f"iterations = {iters}",
                      f"f1(x) = {f1(*result):.10e}", f"f2(x) = {f2(*result):.10e}",
                      f"Residual norm = {residual_norm(result):.10e}",
                      f"Error bound = {history[-1]['error_bound']:.10e}",
                      "k    x1_k             x2_k             error_bound    ||F||_inf"])
        for row in history:
            lines.append(f"{row['iteration']:2d}   {row['x'][0]:.11f}    {row['x'][1]:.11f}    "
                         f"{row['error_bound']:.5e}    {row['residual']:.5e}")
        lines.append("")
    (folder / "output.txt").write_text("\n".join(lines), encoding="utf-8")
    draw_plots()


if __name__ == "__main__":
    main()
