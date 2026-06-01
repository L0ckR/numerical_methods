import math

def l2_norm(x):
    n = len(x)
    l2_norm = 0
    for i in range(n):
        l2_norm += x[i] * x[i]
    return math.sqrt(l2_norm)

# Метод простых итераций
def simple_iteration_method(A, b, eps, max_iter):
    n = len(A)
    alpha = [[-A[i][j] / A[i][i] if i != j else 0 for j in range(n)] for i in range(n)]
    beta = [b[i] / A[i][i] for i in range(n)]
    
    # Начальное приближение
    x_new = [beta[i] for i in range(n)]
    iterations = 0

    eps_k = 0
    alpha_norm = l2_norm([alpha[i][j] for i in range(n) for j in range(n)]) 

    while iterations == 0 or eps_k > eps:
        x = x_new[:]
        x_new = [sum(alpha[i][j] * x[j] for j in range(n)) + beta[i] for i in range(n)]
        iterations += 1
        if (alpha_norm >= 1):
            eps_k = l2_norm([x_new[i] - x[i] for i in range(n)])
        else: 
            eps_k = alpha_norm / (1 - alpha_norm) * l2_norm([x_new[i] - x[i] for i in range(n)])
        if (iterations >= max_iter):
            return x_new, iterations

    return x_new, iterations


# Метод Зейделя (используя подсчет обратной матрицы с помощью LU разложения)
def permutation_matrix(A):
    n = len(A)
    P = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    for i in range(n):
        max_row = max(range(i, n), key=lambda k: abs(A[k][i]))
        if max_row != i:
            P[i], P[max_row] = P[max_row], P[i]
    return P
                       

def LU_decompose(PA):
    n = len(PA)
    L = [[0 for _ in range(n)] for _ in range(n)]
    U = [row[:] for row in PA]

    for i in range(n):
        L[i][i] = 1
        for j in range(i + 1, n):
            if U[i][i] != 0:
                L[j][i] = U[j][i] / U[i][i]
                for k in range(i, n):
                    U[j][k] -= L[j][i] * U[i][k]

    return L, U

def solve(L, U, b):
    n = len(L)

    y = [0 for _ in range(n)]
    for i in range(n):
        y[i] = b[i] - sum(L[i][j] * y[j] for j in range(i)) / L[i][i]

    x = [0 for _ in range(n)]
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(U[i][j] * x[j] for j in range(i + 1, n))) / U[i][i]
    return x

def transpose(A):
    m = len(A)
    n = len(A[0])
    A_T = [[A[j][i] for j in range(n)] for i in range(m)]
    return A_T

def inverse_matrix(A):
    n = len(A)
    E = [[1 if (i == j) else 0 for j in range(n)] for i in range(n)]
    
    P = permutation_matrix(A)
    PA = matrix_mult(P, A)
    L, U = LU_decompose(PA)
    
    A_inv = []
    for i in range(n):
        Pb = [sum(P[j][k] * E[k][i] for k in range(n)) for j in range(n)]
        row_inv = solve(L, U, Pb)
        A_inv.append(row_inv)
    return transpose(A_inv)

def seidel_method(A, b, eps, max_iter):
    n = len(A)
    alpha = [[-A[i][j] / A[i][i] if i != j else 0 for j in range(n)] for i in range(n)]
    beta = [b[i] / A[i][i] for i in range(n)]
    
    # Разделяем матрицу alpha на нижнюю треугольную (B) и оставшуюся часть (C)
    B = [[alpha[i][j] if j < i else 0 for j in range(n)] for i in range(n)]
    C = [[alpha[i][j] if j >= i else 0 for j in range(n)] for i in range(n)]
    
    # Инвертируем (E - B)
    E_minus_B = [[1 if i == j else -B[i][j] for j in range(n)] for i in range(n)]
    inv_E_minus_B = inverse_matrix(E_minus_B)
    
    # Вычисляем tmp1 и tmp2
    tmp1 = matrix_mult(inv_E_minus_B, C)
    tmp2 = matrix_vector_mult(inv_E_minus_B, beta)
    
    # Начальное приближение
    x_new = [tmp2[i] for i in range(n)]
    iterations = 0    

    eps_k = 0
    alpha_norm = l2_norm([alpha[i][j] for i in range(n) for j in range(n)])

    while iterations == 0 or eps_k > eps:
        x = x_new[:]
        for i in range(n):
            x_new = [sum(tmp1[i][j] * x[j] for j in range(n)) + tmp2[i] for i in range(n)]
        iterations += 1
        if (alpha_norm >= 1):
            eps_k = l2_norm([x_new[i] - x[i] for i in range(n)])
        else: 
            eps_k = l2_norm([C[i][j] for i in range(n) for j in range(n)]) / (1 - alpha_norm) * l2_norm([x_new[i] - x[i] for i in range(n)])
        if (iterations >= max_iter):
            return x_new, iterations
        
    return x_new, iterations


def invert_matrix(matrix):
    # Прямой и обратный ход метода Гаусса
    size = len(matrix)
    identity = [[1 if i == j else 0 for j in range(size)] for i in range(size)]
    for i in range(size):
        factor = matrix[i][i]
        for j in range(size):
            matrix[i][j] /= factor
            identity[i][j] /= factor
        for k in range(size):
            if k != i:
                factor = matrix[k][i]
                for j in range(size):
                    matrix[k][j] -= factor * matrix[i][j]
                    identity[k][j] -= factor * identity[i][j]
    return identity

def matrix_vector_mult(A, x):
    n = len(A)
    m = len(x)
    return [sum(A[i][k] * x[k] for k in range(m)) for i in range(n)]

def matrix_mult(A, B):
    n = len(A)
    return [[sum(A[i][k] * B[k][j] for k in range(n)) for j in range(n)] for i in range(n)]

def format_matrix(matrix):
    return '\n'.join(' '.join(f"{0.00 if abs(elem) < 1e-10 else elem:6.2f}" for elem in row) for row in matrix)

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
        f.write(f"Vector b:\n{' '.join(f'{elem:6.2f}' for elem in b)}\n\n")

        f.write(f"Simple iterations method:\n\nSolution x:\n{' '.join(f'{elem:6.2f}' for elem in simple_iter_x)}\n\n")
        #f.write(f"Simple iterations method:\n\nSolution x:\n{' '.join(f'{elem}' for elem in simple_iter_x)}\n\n")
        f.write(f"Number of iterations: {simple_iter_i}\n\n")
        f.write(f"Check A * x = b:\n{' '.join(f'{elem:6.2f}' for elem in (matrix_vector_mult(A, simple_iter_x)))}\n\n")

        f.write(f"Seidel method:\n\nSolution x:\n{' '.join(f'{elem:6.2f}' for elem in seidel_x)}\n\n")
        #f.write(f"Seidel method:\n\nSolution x:\n{' '.join(f'{elem}' for elem in seidel_x)}\n\n")
        f.write(f"Number of iterations: {seidel_i}\n\n")
        f.write(f"Check A * x = b:\n{' '.join(f'{elem:6.2f}' for elem in (matrix_vector_mult(A, seidel_x)))}\n")


if __name__ == "__main__":
    main()
