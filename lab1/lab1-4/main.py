import math

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


def rotation_method(A, eps, max_iter, history=None):
    n = len(A)
    if eps <= 0 or max_iter < 1:
        raise ValueError("eps and max_iter must be positive")
    if any(A[i][j] != A[j][i] for i in range(n) for j in range(n)):
        raise ValueError("The rotation method requires a symmetric matrix")
    A_i = [row[:] for row in A]
    eigen_vectors = [[1 if i == j else 0 for j in range(n)] for i in range(n)] # создаем единичную матрицу
    iters = 0
    eigen_values = [A_i[i][i] for i in range(n)]
    if history is not None:
        history.append((iters, matrix_norm(A_i)))

    while matrix_norm(A_i) > eps:
        if iters >= max_iter:
            raise RuntimeError(f"Rotation method did not converge in {max_iter} iterations")
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
        if history is not None:
            history.append((iters, matrix_norm(A_i)))

    return eigen_values, eigen_vectors, iters, A_i


def format_matrix(matrix):
    return '\n'.join(' '.join(f"{0.00 if abs(elem) < 1e-10 else elem:12.8f}" for elem in row) for row in matrix)

def format_eigen_vectors(eigen_vectors):
    formatted_vectors = []
    for i, row in enumerate(eigen_vectors, start=1):
        formatted_row = ' '.join(f"{elem:12.8f}" for elem in row)
        formatted_vectors.append(f"eigen vector num {i}: {formatted_row}")

    return '\n'.join(formatted_vectors)

def main():
    with open('input.txt', 'r') as f:
        data = [list(map(float, line.split())) for line in f.readlines()]

    A = data[:-1]
    eps = data[-1][0]

    history = []
    eigen_values, eigen_vectors, iters, A_i = rotation_method(A, eps, 100, history)
    eigen_vectors = transpose(eigen_vectors) # в столбцах наши СВ => транспонируем, чтобы теперь СВ были в строках

    with open('output.txt', 'w') as f:
        f.write(f"Matrix A:\n{format_matrix(A)}\n\n")
        f.write(f"Eigen values:\n{' '.join(f'{elem:12.8f}' for elem in eigen_values)}\n\n")
        f.write(f"Eigen vectors:\n{format_eigen_vectors(eigen_vectors)}\n\n")
        f.write(f"Number of iterations: {iters}\n\n")
        f.write(f"Matrix A result:\n{format_matrix(A_i)}\n\n")

        f.write(f"Requested accuracy: {eps:.1e}\n")
        f.write("Iteration / upper off-diagonal norm:\n")
        for iteration, error in history:
            f.write(f"{iteration:3d} {error:.9e}\n")
        for i, vector in enumerate(eigen_vectors):
            residual = math.sqrt(sum((sum(A[j][k] * vector[k] for k in range(len(A))) - eigen_values[i] * vector[j]) ** 2 for j in range(len(A))))
            f.write(f"Eigenpair {i + 1} residual ||A*v-lambda*v||_2: {residual:.3e}\n")
        orthogonality = max(abs(sum(u[k] * v[k] for k in range(len(A))) - (i == j)) for i, u in enumerate(eigen_vectors) for j, v in enumerate(eigen_vectors))
        f.write(f"Orthogonality max |V^T*V-I|: {orthogonality:.3e}\n")

if __name__ == "__main__":
    main()
