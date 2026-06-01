import math
# import numpy as np

def find_max_upper_element(A):
    n = len(A)
    l, m = 0, 1
    max_elem = abs(A[0][1])
    for i in range(n):
        for j in range(i + 1, n):
            if abs(A[i][j]) > max_elem:
                max_elem = abs(A[i][j])
                l = i
                m = j
    return l, m

def matrix_norm(A):
    n = len(A)
    norm = 0
    for i in range(n):
        for j in range(i + 1, n):
            norm += A[i][j] * A[i][j]
    return math.sqrt(norm)

def matrix_mult(A, B):
    n = len(A)
    return [[sum(A[i][k] * B[k][j] for k in range(n)) for j in range(n)] for i in range(n)]


def transpose(A):
    m = len(A)
    n = len(A[0])
    A_T = [[A[j][i] for j in range(n)] for i in range(m)]
    return A_T


def rotation_method(A, eps, max_iter):
    n = len(A)
    A_i = [row[:] for row in A]
    eigen_vectors = [[1 if i == j else 0 for j in range(n)] for i in range(n)] # создаем единичную матрицу
    iters = 0

    while matrix_norm(A_i) > eps:
        l, m = find_max_upper_element(A_i)
        if A_i[l][l] - A_i[m][m] == 0:
            phi = math.pi / 4
        else:
            phi = 0.5 * math.atan(2 * A_i[l][m] / (A_i[l][l] - A_i[m][m]))

        # Матрица вращения
        U = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
        U[l][l] = math.cos(phi)
        U[l][m] = -math.sin(phi)
        U[m][l] = math.sin(phi)
        U[m][m] = math.cos(phi)

        U_T = transpose(U)
        A_i = matrix_mult(matrix_mult(U_T, A_i), U)
        eigen_vectors = matrix_mult(eigen_vectors, U) # СВ - столбцы
        eigen_values = [A_i[i][i] for i in range(n)] # СЗ - диагональные элементы
        iters += 1
        if (iters >= max_iter):
            return eigen_values, eigen_vectors, iters, A_i

    return eigen_values, eigen_vectors, iters, A_i


def format_matrix(matrix):
    return '\n'.join(' '.join(f"{0.00 if abs(elem) < 1e-10 else elem:6.2f}" for elem in row) for row in matrix)

def format_eigen_vectors(eigen_vectors):
    formatted_vectors = []
    for i, row in enumerate(eigen_vectors, start=1):
        formatted_row = ' '.join(f"{elem:6.2f}" for elem in row)
        formatted_vectors.append(f"eigen vector num {i}: {formatted_row}")

    return '\n'.join(formatted_vectors)

def main():
    with open('input.txt', 'r') as f:
        data = [list(map(float, line.split())) for line in f.readlines()]
    
    A = data[:-1]
    eps = data[-1][0]

    eigen_values, eigen_vectors, iters, A_i = rotation_method(A, eps, 100)
    eigen_vectors = transpose(eigen_vectors) # в столбцах наши СВ => транспонируем, чтобы теперь СВ были в строках

    # проверка через numpy
    # eigenvalues, eigenvectors = np.linalg.eig(A)
    # eigenvectors = transpose(eigen_vectors)

    with open('output.txt', 'w') as f:
        f.write(f"Matrix A:\n{format_matrix(A)}\n\n")
        f.write(f"Eigen values:\n{' '.join(f'{elem:6.2f}' for elem in eigen_values)}\n\n")
        f.write(f"Eigen vectors:\n{format_eigen_vectors(eigen_vectors)}\n\n")
        f.write(f"Number of iterations: {iters}\n\n")
        f.write(f"Matrix A result::\n{format_matrix(A_i)}\n\n")
        # f.write(f"Eigen values:\n{' '.join(f'{elem:6.2f}' for elem in eigenvalues)}\n\n")
        # f.write(f"Eigen vectors:\n{format_eigen_vectors(eigenvectors)}\n\n")

if __name__ == "__main__":
    main()