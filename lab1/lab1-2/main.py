def read_tridiagonal_matrix(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    matrix = []
    for line in lines[:-1]:
        row = list(map(float, line.split()))
        matrix.append(row)

    b = list(map(float, lines[-1].split()))

    n = len(matrix)
    A = [[0.0] * n for _ in range(n)]

    for i in range(n):
        if n == 1:
            A[0][0] = matrix[0][0]
        elif i == 0:
            A[i][i] = matrix[i][0]  # Главная диагональ
            A[i][i + 1] = matrix[i][1]  # Верхняя диагональ
        elif i == n - 1:
            A[i][i - 1] = matrix[i][0]  # Нижняя диагональ
            A[i][i] = matrix[i][1]  # Главная диагональ
        else:
            A[i][i - 1] = matrix[i][0]  # Нижняя диагональ
            A[i][i] = matrix[i][1]  # Главная диагональ
            A[i][i + 1] = matrix[i][2]  # Верхняя диагональ

    return A, b

def tridiagonal_matrix_algorithm(A, d):
    n = len(d)
    p = [0.0] * n
    q = [0.0] * n
    # Прямой ход: x_i = p_i*x_(i+1) + q_i.
    for i in range(n):
        a = A[i][i - 1] if i > 0 else 0.0
        c = A[i][i + 1] if i < n - 1 else 0.0
        denominator = A[i][i] + (a * p[i - 1] if i else 0.0)
        if denominator == 0:
            raise ValueError("Thomas algorithm encountered a zero denominator")
        p[i] = -c / denominator
        q[i] = (d[i] - (a * q[i - 1] if i else 0.0)) / denominator
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = q[i] + (p[i] * x[i + 1] if i < n - 1 else 0.0)
    return x

def matrix_vector_mult(A, x):
    n = len(A)
    m = len(x)
    return [sum(A[i][k] * x[k] for k in range(m)) for i in range(n)]

def format_matrix(matrix):
    return '\n'.join(' '.join(f"{0.00 if abs(elem) < 1e-10 else elem:12.8f}" for elem in row) for row in matrix)

def main():
    A, b = read_tridiagonal_matrix('input.txt')

    x = tridiagonal_matrix_algorithm(A, b)

    with open('output.txt', 'w') as f:
        f.write(f"Matrix A:\n{format_matrix(A)}\n\n")
        f.write(f"Vector b:\n{' '.join(f'{elem:12.8f}' for elem in b)}\n\n")
        f.write(f"Solution x:\n{' '.join(f'{elem:12.8f}' for elem in x)}\n\n")
        f.write(f"Check A * x = b:\n{' '.join(f'{elem:12.8f}' for elem in (matrix_vector_mult(A, x)))}\n")
        residual = max(abs(value - expected) for value, expected in zip(matrix_vector_mult(A, x), b))
        f.write(f"Residual ||A*x-b||_inf: {residual:.3e}\n")


if __name__ == "__main__":
    main()
