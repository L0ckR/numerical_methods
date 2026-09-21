import math

def f(x):
    return math.exp(x)

def prod(points, x, i):
    res = 1
    for j in range(len(points)):
        if j != i:
            res *= (x - points[j])
    return res

def prod_to_print(points, i):
    prod = ""
    for j in range(len(points)):
        if i != j:
            prod += f"(x - {points[j]})"
    return prod

def Lagrange_interpolation(points, x):
    res = 0
    res_str = "L(x) = "
    for i in range(len(points)):
        f_prod = f(points[i]) / prod(points, points[i], i)
        res += f_prod * prod(points, x, i)

        sign = " + " if f_prod > 0 else ""
        res_str += f"{sign} {f_prod:.12g}*" + prod_to_print(points, i)

    return res, res_str

def Newton_interpolation(points, x):
    y = [f(p) for p in points]

    coefs = [y[i] for i in range(len(points))]
    for j in range(1, len(points)):
        for i in range(len(points) - 1, j - 1, -1):
            coefs[i] = float(coefs[i] - coefs[i - 1]) / float(points[i] - points[i - j])

    
    res = coefs[0]
    res_str = f"P(x) = {coefs[0]:.12g}"
    current_terms = []
    
    for i in range(1, len(coefs)):
        current_terms.append(f"(x - {points[i-1]:.12g})")
        term_str = "*".join(current_terms)
        p = 1
        for j in range(i):
            p *= x - points[j]
        res += coefs[i] * p
        
        sign = " + " if coefs[i] >= 0 else " - "
        res_str += f"{sign}{abs(coefs[i]):.12g}*{term_str}"
    
    return res, res_str



def main():
    with open('input.txt', 'r') as file:
        data = [list(map(float, line.split())) for line in file.readlines()]
    points = data[:-1]
    x = data[-1][0]
    

    val_lagrange1, str_lagrange1  = Lagrange_interpolation(points[0], x)
    abs_err_lag1 = abs(f(x) - val_lagrange1)
    val_newton1, str_newton1 = Newton_interpolation(points[0], x)
    abs_err_new1 = abs(f(x) - val_newton1)
    

    val_lagrange2, str_lagrange2  = Lagrange_interpolation(points[1], x)
    abs_err_lag2 = abs(f(x) - val_lagrange2)
    val_newton2, str_newton2 = Newton_interpolation(points[1], x)
    abs_err_new2 = abs(f(x) - val_newton2)
    
    with open('output.txt', 'w') as file:
        file.write(f"Function: y = exp(x)\n")
        file.write(f"x* = {x}, y = {f(x)}\n\n\n")

        file.write(f"Points: {points[0]}\n\n")
        file.write("-Lagrange interpolation:\n")
        file.write(str_lagrange1)
        file.write(f"\nValue: {val_lagrange1}\n")
        file.write(f"Error: {abs_err_lag1}\n")
        file.write("-Newton interpolation:\n")
        file.write(str_newton1)
        file.write(f"\nValue: {val_newton1}\n")
        file.write(f"Error: {abs_err_new1}\n\n\n")

        file.write(f"Points: {points[1]}\n\n")
        file.write("-Lagrange interpolation:\n")
        file.write(str_lagrange2)
        file.write(f"\nValue: {val_lagrange2}\n")
        file.write(f"Error: {abs_err_lag2}\n")
        file.write("-Newton interpolation:\n")
        file.write(str_newton2)
        file.write(f"\nValue: {val_newton2}\n")
        file.write(f"Error: {abs_err_new2}\n")

if __name__ == "__main__":
    main()
