import math


def f(x):
    return math.log(x + 1) - 2 * x ** 2 + 1


def df(x):
    return 1 / (x + 1) - 4 * x


def phi(x):
    return math.sqrt((math.log(x + 1) + 1) / 2)


def simple_iteration_method(a, b, eps):
    x = (a + b) / 2
    iters = 0
    while True:
        x_next = phi(x)
        iters += 1
        if abs(x_next - x) < eps:
            return x_next, iters
        x = x_next


def newton_method(a, b, eps):
    x = (a + b) / 2
    iters = 0
    while True:
        x_next = x - f(x) / df(x)
        iters += 1
        if abs(x_next - x) < eps:
            return x_next, iters
        x = x_next


def main():
    with open('input.txt', 'r') as file:
        data = [list(map(float, line.split())) for line in file.readlines()]
    a, b = data[0]
    eps = data[1][0]

    si_root, si_iters = simple_iteration_method(a, b, eps)
    n_root, n_iters = newton_method(a, b, eps)

    with open('output.txt', 'w') as file:
        file.write("Equation: ln(x + 1) - 2*x^2 + 1 = 0\n\n")
        file.write(f"Simple iteration method:\nx = {si_root}\niterations = {si_iters}\nf(x) = {f(si_root)}\n\n")
        file.write(f"Newton method:\nx = {n_root}\niterations = {n_iters}\nf(x) = {f(n_root)}")


if __name__ == "__main__":
    main()
