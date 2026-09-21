import math


def infinity_norm(x):
    return max(abs(value) for value in x)


def iteration_coefficients(A, b, eps, max_iter):
    if eps <= 0 or max_iter < 1:
        raise ValueError("eps and max_iter must be positive")
    n = len(A)
    if any(A[i][i] == 0 for i in range(n)):
        raise ValueError("A zero diagonal entry prevents fixed-point iteration")
    alpha = [[-A[i][j] / A[i][i] if i != j else 0.0 for j in range(n)] for i in range(n)]
    beta = [b[i] / A[i][i] for i in range(n)]
    return alpha, beta


def converged(A, b, old, new, eps, q, numerator):
    delta = infinity_norm([a - b for a, b in zip(new, old)])
    if not all(math.isfinite(value) for value in new):
        raise RuntimeError("The iteration diverged to non-finite values")
    if q < 1:
        # A posteriori bound in the same vector/matrix infinity norm.
        return numerator / (1 - q) * delta <= eps
    # Without a contraction bound, require both a small step and residual.
    residual = infinity_norm([value - rhs for value, rhs in zip(matrix_vector_mult(A, new), b)])
    return delta <= eps and residual <= eps


def simple_iteration_method(A, b, eps, max_iter):
    alpha, beta = iteration_coefficients(A, b, eps, max_iter)
    q = max(sum(abs(value) for value in row) for row in alpha)
    x = beta[:]
    for iterations in range(1, max_iter + 1):
        new = [sum(alpha[i][j] * x[j] for j in range(len(A))) + beta[i] for i in range(len(A))]
        if converged(A, b, x, new, eps, q, q):
            return new, iterations
        x = new
    raise RuntimeError(f"Simple iteration did not converge in {max_iter} iterations")


def seidel_method(A, b, eps, max_iter):
    alpha, beta = iteration_coefficients(A, b, eps, max_iter)
    n = len(A)
    q = max(sum(abs(value) for value in row) for row in alpha)
    c_norm = max(sum(abs(alpha[i][j]) for j in range(i + 1, n)) for i in range(n))
    x = beta[:]
    for iterations in range(1, max_iter + 1):
        new = x[:]
        for i in range(n):
            # Updated components are used immediately, without a matrix inverse.
            new[i] = beta[i] + sum(alpha[i][j] * new[j] for j in range(i)) + sum(alpha[i][j] * x[j] for j in range(i + 1, n))
        if converged(A, b, x, new, eps, q, c_norm):
            return new, iterations
        x = new
    raise RuntimeError(f"Seidel iteration did not converge in {max_iter} iterations")


def matrix_vector_mult(A, x):
    n = len(A)
    m = len(x)
    return [sum(A[i][k] * x[k] for k in range(m)) for i in range(n)]

def matrix_mult(A, B):
    n = len(A)
    return [[sum(A[i][k] * B[k][j] for k in range(n)) for j in range(n)] for i in range(n)]

def format_matrix(matrix):
    return '\n'.join(' '.join(f"{0.00 if abs(elem) < 1e-10 else elem:12.8f}" for elem in row) for row in matrix)

def main():
    with open('input.txt', 'r') as f:
        data = [list(map(float, line.split())) for line in f.readlines()]

    A = data[:-2]
    b = data[-2]
    eps = data[-1][0]

    simple_iter_x, simple_iter_i = simple_iteration_method(A, b, eps, 100)

    seidel_x, seidel_i = seidel_method(A, b, eps, 100)

    with open('output.txt', 'w') as f:
        f.write(f"Matrix A:\n{format_matrix(A)}\n\n")
        f.write(f"Requested accuracy: {eps:.1e}\n\n")
        q = max(sum(abs(A[i][j] / A[i][i]) for j in range(len(A)) if j != i) for i in range(len(A)))
        f.write(f"Contraction norm ||alpha||_inf: {q:.9f}\n\n")
        f.write(f"Vector b:\n{' '.join(f'{elem:12.8f}' for elem in b)}\n\n")

        f.write(f"Simple iterations method:\n\nSolution x:\n{' '.join(f'{elem:12.8f}' for elem in simple_iter_x)}\n\n")
        f.write(f"Number of iterations: {simple_iter_i}\n\n")
        f.write(f"Check A * x = b:\n{' '.join(f'{elem:12.8f}' for elem in (matrix_vector_mult(A, simple_iter_x)))}\n\n")

        f.write(f"Seidel method:\n\nSolution x:\n{' '.join(f'{elem:12.8f}' for elem in seidel_x)}\n\n")
        f.write(f"Number of iterations: {seidel_i}\n\n")
        f.write(f"Check A * x = b:\n{' '.join(f'{elem:12.8f}' for elem in (matrix_vector_mult(A, seidel_x)))}\n")

        for name, x in (("Simple iterations", simple_iter_x), ("Seidel", seidel_x)):
            residual = infinity_norm([value - rhs for value, rhs in zip(matrix_vector_mult(A, x), b)])
            f.write(f"{name} residual ||A*x-b||_inf: {residual:.3e}\n")

if __name__ == "__main__":
    main()
