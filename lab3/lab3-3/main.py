def LU_decompose(A):
    n = len(A)
    # lower
    L = [[0 for _ in range(n)] for _ in range(n)]
    # upper
    U = [row[:] for row in A]

    for k in range(1, n):
        for i in range(k - 1, n):
            for j in range(i, n):
                L[j][i] = U[j][i] / U[i][i]

        for i in range(k, n):
            for j in range(k - 1, n):
                U[i][j] = U[i][j] - L[i][k - 1] * U[k - 1][j]

    return L, U

def solve_system(L, U, b):
    # L * y = b
    n = len(L)
    y = [0 for _ in range(n)]
    for i in range(n):
        s = 0
        for j in range(i):
            s += L[i][j] * y[j]
        y[i] = (b[i] - s) / L[i][i]

    # U * x = y
    x = [0 for _ in range(n)]
    for i in range(n - 1, -1, -1):
        s = 0
        for j in range(n - 1, i - 1, -1):
            s += U[i][j] * x[j]
        x[i] = (y[i] - s) / U[i][i]
    return x

def least_squares(x, y, n):
    assert len(x) == len(y)
    A = []
    b = []
    for k in range(n + 1):
        A.append([sum(map(lambda x: x ** (i + k), x)) for i in range(n + 1)])
        b.append(sum(map(lambda x: x[0] * x[1] ** k, zip(y, x))))
    L, U = LU_decompose(A)
    return solve_system(L, U, b)

def P(coefs, x):
    return sum([c * x**i for i, c in enumerate(coefs)])

def sum_squared_errors(x, y, ls_coefs):
    y_ls = [P(ls_coefs, x_i) for x_i in x]
    return sum((y_i - y_ls_i)**2 for y_i, y_ls_i in zip(y, y_ls))


def draw_plot(x, y, ls1, ls2):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = [min(x) + (max(x) - min(x)) * i / 400 for i in range(401)]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(x, y, "o--", color="black", linewidth=1, label="Табличная функция")
    ax.plot(grid, [P(ls1, point) for point in grid], label="МНК: степень 1")
    ax.plot(grid, [P(ls2, point) for point in grid], label="МНК: степень 2")
    ax.set(xlabel="x", ylabel="y", title="Метод наименьших квадратов — вариант 6")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig("least_squares.png", dpi=180)
    plt.close(fig)

def main():
    with open('input.txt', 'r') as file:
        data = [list(map(float, line.split())) for line in file.readlines()]
    x = data[0]
    y = data[1]

    ls1 = least_squares(x, y, 1)
    ls2 = least_squares(x, y, 2)

    with open('output.txt', 'w') as file:
        file.write("Least squares method, degree = 1:\n")
        file.write(f"P(x) = {ls1[0]} + {ls1[1]} * x\n")
        file.write(f"Sum of squared errors = {sum_squared_errors(x, y, ls1)}\n\n")
        file.write("Least squares method, degree = 2:\n")
        file.write(f"P(x) = {ls2[0]} + {ls2[1]} * x + {ls2[2]} * x^2\n")
        file.write(f"Sum of squared errors = {sum_squared_errors(x, y, ls2)}")

    draw_plot(x, y, ls1, ls2)

if __name__ == "__main__":
    main()
