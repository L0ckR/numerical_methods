import math

def sign(x):
    return -1 if x < 0 else (1 if x > 0 else 0)

def l2_norm(x):
    n = len(x)
    l2_norm = 0
    for i in range(n):
        l2_norm += x[i] * x[i]
    return math.sqrt(l2_norm)

def matrix_mult(A, B):
    n = len(A)
    return [[sum(A[i][k] * B[k][j] for k in range(n)) for j in range(n)] for i in range(n)]

def transpose(A):
    m = len(A)
    n = len(A[0])
    A_T = [[A[j][i] for j in range(n)] for i in range(m)]
    return A_T

def matrix_subtract(A, B):
    n = len(A)
    return [[A[i][j] - B[i][j] for j in range(len(A[0]))] for i in range(n)]

def householder(A, col):
    n = len(A)
    a = [A[i][col] for i in range(n)]
    v = [0.0] * n
    
    v[col] = a[col] + sign(a[col]) * l2_norm(a[col:])
    
    for i in range(col + 1, n):
        v[i] = a[i]
    
    E = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    scalar = 2.0 / sum(v[i] * v[i] for i in range(n))
    H = matrix_subtract(E, [[scalar * v[i] * v[j] for j in range(n)] for i in range(n)])
    
    return H

def get_QR(A):
    n = len(A)
    Q = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    R = [row.copy() for row in A]

    for i in range(n-1):
        H = householder(R, i)
        Q = matrix_mult(Q, H)
        R = matrix_mult(H, R)
    
    return Q, R

def solve_quad(a, b, c):
    discr = b * b - 4 * a * c
    
    if discr < 0:
        real = -b / (2 * a)
        imag = math.sqrt(-discr)/(2 * a)
        return [complex(real, imag), complex(real, -imag)]
        # return [(real + imag * 1j), (real - imag * 1j)]
    else:
        root1 = (-b + math.sqrt(discr)) / (2 * a)
        root2 = (-b - math.sqrt(discr)) / (2 * a)
        return [root1, root2]

def get_roots(A, i):
    n = len(A)
    a11 = A[i][i]
    a12 = A[i][i + 1] if i + 1 < n else 0.0
    a21 = A[i + 1][i] if i + 1 < n else 0.0
    a22 = A[i + 1][i + 1] if i + 1 < n else 0.0
    
    return solve_quad(1.0, -a11 - a22, a11 * a22 - a12 * a21)

def subdiag_norm(A, i):
    return math.sqrt(sum(A[row][i] ** 2 for row in range(i + 1, len(A))))

def eigenvals_QR(A, eps, max_iter):
    n = len(A)
    A_k = [row.copy() for row in A]
    eigen = []
    i = 0
    iterations = 0
    
    while i < n:
        while True:
            iterations += 1
            if iterations > max_iter:
                return eigen, iterations, A_k

            Q, R = get_QR(A_k)
            A_k = matrix_mult(R, Q)
            
            if subdiag_norm(A_k, i) <= eps:
                eigen.append(A_k[i][i]) # вещественные
                i += 1
                break
            elif i + 1 < n and subdiag_norm(A_k, i + 1) <= eps:
                roots = get_roots(A_k, i) # комплексные
                eigen.extend(roots)
                i += 2
                break
    
    return eigen, iterations, A_k

def format_matrix(matrix):
    return '\n'.join(' '.join(f"{0.00 if abs(elem) < 1e-10 else elem:6.2f}" for elem in row) for row in matrix)

def main():
    with open('input.txt', 'r') as f:
        data = [list(map(float, line.split())) for line in f.readlines()]
    
    A = data[:-1]
    eps = data[-1][0]

    eigenvalues, iters, A_k = eigenvals_QR(A, eps, 100)

    with open('output.txt', 'w') as f:
        f.write(f"Matrix A:\n{format_matrix(A)}\n\n")
        f.write(f"Eigen values:\n{' '.join(f'{elem:6.2f}' for elem in eigenvalues)}\n\n")
        f.write(f"Matrix A result:\n{format_matrix(A_k)}\n\n")
        f.write(f"Number of iterations: {iters}\n\n")

if __name__ == "__main__":
    main()