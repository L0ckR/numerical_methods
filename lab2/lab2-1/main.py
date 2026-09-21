"""Variant 6: the positive root of exp(x) - 2*x - 2 = 0."""

import math
from pathlib import Path


def f(x):
    return math.exp(x) - 2 * x - 2


def df(x):
    return math.exp(x) - 2


def ddf(x):
    return math.exp(x)


def phi(x):
    return math.log(2 * x + 2)


def dphi(x):
    return 1 / (x + 1)


def convergence_data(a, b, eps, max_iter):
    """Validate an increasing positive-root interval and a contraction on it."""
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    if not isinstance(max_iter, int) or max_iter <= 0:
        raise ValueError("max_iter must be a positive integer")
    if not all(math.isfinite(x) for x in (a, b)) or not 0 < a < b:
        raise ValueError("The interval must satisfy 0 < a < b")
    if f(a) > 0 or f(b) < 0 or df(a) <= 0:
        raise ValueError("The interval must bracket the positive root with f' > 0")
    q = dphi(a)  # phi' decreases throughout a positive interval.
    if q >= 1 or not (a <= phi(a) <= phi(b) <= b):
        raise ValueError("phi must map the interval into itself with q < 1")
    return q, df(a)


def simple_iteration_method(a, b, eps, max_iter=1000, history=None):
    q, _ = convergence_data(a, b, eps, max_iter)
    x = (a + b) / 2
    for iters in range(1, max_iter + 1):
        x_next = phi(x)
        step = abs(x_next - x)
        error_bound = q / (1 - q) * step
        residual = abs(f(x_next))
        if history is not None:
            history.append(dict(iteration=iters, x=x_next, step=step,
                                error_bound=error_bound, residual=residual))
        if error_bound <= eps and residual <= eps:
            return x_next, iters
        x = x_next
    raise RuntimeError("Simple iteration did not converge within max_iter")


def newton_method(a, b, eps, max_iter=1000, history=None):
    _, m1 = convergence_data(a, b, eps, max_iter)
    # f'' > 0; f(b) >= 0 implies monotone convergence from the right.
    x = b
    for iters in range(1, max_iter + 1):
        derivative = df(x)
        if derivative == 0:
            raise ArithmeticError("Newton derivative is zero")
        x_next = x - f(x) / derivative
        if not math.isfinite(x_next) or not a <= x_next <= b:
            raise ArithmeticError("Newton iterate left the validated interval")
        residual = abs(f(x_next))
        # Mean value theorem: |x - x*| <= |f(x)| / min |f'|.
        error_bound = residual / m1
        if history is not None:
            history.append(dict(iteration=iters, x=x_next, step=abs(x_next-x),
                                error_bound=error_bound, residual=residual))
        if error_bound <= eps and residual <= eps:
            return x_next, iters
        x = x_next
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
    from scipy.optimize import brentq
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25})
    folder = Path(__file__).resolve().parent
    values = [list(map(float, row.split())) for row in (folder/"input.txt").read_text().splitlines() if row.strip()]
    (a, b), (eps,) = values
    reference = brentq(lambda x: np.exp(x)-2*x-2, a, b, xtol=1e-14)
    fig, ax = plt.subplots(figsize=(7.5, 4.1), layout="constrained")
    x = np.linspace(0, b+.15, 400)
    ax.plot(x, np.exp(x), label=r"$y=e^x$", color="#2166ac")
    ax.plot(x, 2*x+2, label=r"$y=2x+2$", color="#b2182b")
    ax.axvspan(a, b, alpha=.10, color="#2166ac", label="Начальный интервал [1; 2]")
    ax.scatter([reference], [np.exp(reference)], color="black", s=28, zorder=5)
    ax.annotate(f"x* ≈ {reference:.6f}", (reference, np.exp(reference)), xytext=(1.03, 6.7),
                arrowprops={"arrowstyle": "->", "color": "#555555"})
    ax.set(xlabel="x", ylabel="y", title="2.1. Графическое отделение положительного корня")
    ax.legend(fontsize=9)
    fig.savefig(folder/"roots.png", dpi=180)
    plt.close(fig)
    convergence_plot((a, b, eps), reference, folder)


def main():
    folder = Path(__file__).resolve().parent
    data = [list(map(float, line.split()))
            for line in (folder / "input.txt").read_text().splitlines() if line.strip()]
    if len(data) != 2 or len(data[0]) != 2 or len(data[1]) != 1:
        raise ValueError("Expected interval endpoints followed by eps")
    a, b = data[0]
    eps = data[1][0]
    q, m1 = convergence_data(a, b, eps, 1000)
    lines = ["Variant: 6", "Equation: exp(x) - 2*x - 2 = 0",
             f"Interval: [{a}, {b}]", f"Epsilon: {eps}", f"q = {q}",
             f"min |f'(x)| = {m1}", ""]
    for name, method, initial in (("Simple iteration method", simple_iteration_method, (a+b)/2),
                                  ("Newton method", newton_method, b)):
        history = []
        result, iters = method(a, b, eps, history=history)
        lines.extend([name + ":", f"Initial approximation: {initial}",
                      f"x = {result:.15g}", f"iterations = {iters}",
                      f"f(x) = {f(result):.10e}",
                      f"Error bound = {history[-1]['error_bound']:.10e}",
                      "k    x_k                 error_bound     |f(x_k)|"])
        for row in history:
            lines.append(f"{row['iteration']:2d}   {row['x']:.14f}   "
                         f"{row['error_bound']:.6e}    {row['residual']:.6e}")
        lines.append("")
    (folder / "output.txt").write_text("\n".join(lines), encoding="utf-8")
    draw_plots()


if __name__ == "__main__":
    main()
