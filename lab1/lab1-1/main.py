def permutation_matrix(A):
    n = len(A)
    P = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    
    iter_count = 0
    # Находим максимальный элемент
    # Меняем первую строку со строкой с макс эл-том    
    for i in range(n):
        max_row = max(range(i, n), key=lambda k: abs(A[k][i]))
        if max_row != i:
            P[i], P[max_row] = P[max_row], P[i]
            iter_count += 1
    return P, iter_count
                       

def LU_decompose(PA):
    n = len(PA)
    # lower
    L = [[0 for _ in range(n)] for _ in range(n)]
    # upper
    U = [row[:] for row in PA]

    for i in range(n):
        # Заполняем матрицы L и U
        L[i][i] = 1
        for j in range(i + 1, n):
            if U[i][i] != 0:
                L[j][i] = U[j][i] / U[i][i]
                for k in range(i, n):
                    U[j][k] -= L[j][i] * U[i][k]

    return L, U

def solve(L, U, b):
    n = len(L)
    # L * y = b
    y = [0 for _ in range(n)]
    for i in range(n):
        y[i] = b[i] - sum(L[i][j] * y[j] for j in range(i)) / L[i][i]

    # U * x = y
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
    
    P, _ = permutation_matrix(A)
    PA = matrix_mult(P, A)
    L, U = LU_decompose(PA)
    
    A_inv = []
    for i in range(n):
        Pb = [sum(P[j][k] * E[k][i] for k in range(n)) for j in range(n)]
        row_inv = solve(L, U, Pb)
        A_inv.append(row_inv)
    return transpose(A_inv)

def determinant(L, U):
    n = len(U)
    det = 1
    for i in range(n):
        det *= U[i][i]
    return det

def matrix_mult(A, B):
    n = len(A)
    return [[sum(A[i][k] * B[k][j] for k in range(n)) for j in range(n)] for i in range(n)]

def matrix_vector_mult(A, x):
    n = len(A)
    m = len(x)
    return [sum(A[i][k] * x[k] for k in range(m)) for i in range(n)]

def format_matrix(matrix):
    return '\n'.join(' '.join(f"{0.00 if abs(elem) < 1e-10 else elem:6.2f}" for elem in row) for row in matrix)

def main():
    with open('input.txt', 'r') as f:
        data = [list(map(float, line.split())) for line in f.readlines()]
    
    A = data[:-1]
    b = data[-1]
    
    P, iter_count = permutation_matrix(A)
    PA = matrix_mult(P, A)
    L, U = LU_decompose(PA)

    det_A1 = determinant(L, U) * (-1) ** iter_count
    det_A2 = determinant(L, U)

    A_inv = inverse_matrix(A)

    b_T = [sum(P[i][j] * b[j] for j in range(len(b))) for i in range(len(b))]
    x = solve(L, U, b_T)
    check = matrix_mult(A, A_inv)
    

    with open('output.txt', 'w') as f:
        f.write(f"Matrix A:\n{format_matrix(A)}\n\n")
        f.write(f"Vector b:\n{' '.join(f'{elem:6.2f}' for elem in b)}\n\n")
        f.write(f"Matrix U:\n{format_matrix(U)}\n\n")
        f.write(f"Matrix L:\n{format_matrix(L)}\n\n")
        f.write(f"Solution x:\n{' '.join(f'{elem:6.2f}' for elem in x)}\n\n")
        f.write(f"Determinant A: {det_A1:6.2f}\n")
        f.write(f"Determinant PA: {det_A2:6.2f}\n\n")
        f.write(f"Inverse matrix A^(-1):\n{format_matrix(A_inv)}\n\n")
        f.write(f"Check A * x = b:\n{' '.join(f'{elem:6.2f}' for elem in (matrix_vector_mult(A, x)))}\n\n")
        f.write(f"Check A * A^(-1) = E:\n{format_matrix(check)}\n")

if __name__ == "__main__":
    main()
