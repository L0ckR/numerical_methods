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
        if i == 0:
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
    a_diag = [A[i][i - 1] if i > 0 else 0 for i in range(n)] 
    b_diag = [A[i][i] for i in range(n)]
    c_diag = [A[i][i + 1] if i < n - 1 else 0 for i in range(n)]
    

    p = [0 for _ in range(n)]
    q = [0 for _ in range(n)]

    # Задаем начальные значения прогоночных коэффициентов
    p[0] = A[0][1] / -A[0][0]
    q[0] = d[0] / A[0][0]

    # Прямой ход
    for i in range(1, n - 1):
        p[i] = -c_diag[i] / (a_diag[i] * p[i - 1] + b_diag[i])
        q[i] = (d[i] - a_diag[i] * q[i - 1]) / (a_diag[i] * p[i-1] + b_diag[i])
    
    q[n - 1] = (d[n - 1] - a_diag[n - 1] * q[n - 2]) / (b_diag[n - 1] + a_diag[n - 1] * p[n - 2])

    x = [0 for _ in range(n)]
    x[n - 1] = q[n - 1]

    # Обратный ход
    for i in range(n - 1, 0, -1):
        x[i - 1] = p[i - 1] * x[i] + q[i - 1]
    return x

def matrix_vector_mult(A, x):
    n = len(A)
    m = len(x)
    return [sum(A[i][k] * x[k] for k in range(m)) for i in range(n)]

def format_matrix(matrix):
    return '\n'.join(' '.join(f"{0.00 if abs(elem) < 1e-10 else elem:6.2f}" for elem in row) for row in matrix)

def main():
    A, b = read_tridiagonal_matrix('input.txt')

    x = tridiagonal_matrix_algorithm(A, b)

    with open('output.txt', 'w') as f:
        f.write(f"Matrix A:\n{format_matrix(A)}\n\n")
        f.write(f"Vector b:\n{' '.join(f'{elem:6.2f}' for elem in b)}\n\n")
        f.write(f"Solution x:\n{' '.join(f'{elem:6.2f}' for elem in x)}\n\n")
        f.write(f"Check A * x = b:\n{' '.join(f'{elem:6.2f}' for elem in (matrix_vector_mult(A, x)))}\n")


if __name__ == "__main__":
    main()