import math

a = 2.0


def f1(x1, x2):
    return a * x1 - math.cos(x2)


def f2(x1, x2):
    return a * x2 - math.exp(x1)


def jacobian(x1, x2):
    return [[a, math.sin(x2)], [-math.exp(x1), a]]


def phi1(x1, x2):
    return math.cos(x2) / a


def phi2(x1, x2):
    return math.exp(x1) / a


def simple_iteration_method(intervals, eps):
    x1 = sum(intervals[0]) / 2
    x2 = sum(intervals[1]) / 2
    iters = 0
    while True:
        x1_next, x2_next = phi1(x1, x2), phi2(x1, x2)
        iters += 1
        if max(abs(x1_next - x1), abs(x2_next - x2)) < eps:
            return [x1_next, x2_next], iters
        x1, x2 = x1_next, x2_next


def newton_method(intervals, eps):
    x1 = sum(intervals[0]) / 2
    x2 = sum(intervals[1]) / 2
    iters = 0
    while True:
        j = jacobian(x1, x2)
        det = j[0][0] * j[1][1] - j[0][1] * j[1][0]
        dx1 = (-f1(x1, x2) * j[1][1] + j[0][1] * f2(x1, x2)) / det
        dx2 = (-j[0][0] * f2(x1, x2) + f1(x1, x2) * j[1][0]) / det
        x1_next, x2_next = x1 + dx1, x2 + dx2
        iters += 1
        if max(abs(x1_next - x1), abs(x2_next - x2)) < eps:
            return [x1_next, x2_next], iters
        x1, x2 = x1_next, x2_next


def main():
    with open('input.txt', 'r') as file:
        data = [list(map(float, line.split())) for line in file.readlines()]
    intervals = data[:-1]
    eps = data[-1][0]

    si_x, si_iters = simple_iteration_method(intervals, eps)
    n_x, n_iters = newton_method(intervals, eps)

    with open('output.txt', 'w') as file:
        file.write("System:\n2*x1 - cos(x2) = 0\n2*x2 - exp(x1) = 0\n\n")
        file.write(f"Simple iteration method:\nx = {si_x}\niterations = {si_iters}\n\n")
        file.write(f"Newton method:\nx = {n_x}\niterations = {n_iters}")


if __name__ == "__main__":
    main()
