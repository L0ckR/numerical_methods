def f(x):
    return 1 / (3 * x**2 + 4 * x + 2)

def integrate_rectangle_method(f, l, r, h):
    result = 0
    cur_x = l
    while cur_x < r:
        result += h * f((cur_x + cur_x + h) * 0.5)
        cur_x += h
    return result

def integrate_trapeze_method(f, l, r, h):
    result = 0
    cur_x = l
    while cur_x < r:
        result += h * 0.5 * (f(cur_x + h) + f(cur_x))
        cur_x += h
    return result

def integrate_simpson_method(f, l, r, h):
    result = 0
    cur_x = l + h
    while cur_x < r:
        result += f(cur_x - h) + 4 * f(cur_x) + f(cur_x + h)
        cur_x += 2 * h
    return result * h / 3

def runge_rombert_method(h1, h2, integral1, integral2, p):
    return (integral1 - integral2) / ((h2 / h1)**p - 1), integral1 + (integral1 - integral2) / ((h2 / h1)**p - 1)

def main():
    with open('input.txt', 'r') as file:
        data = [list(map(float, line.split())) for line in file.readlines()]
    l, r = data[0][0], data[0][1]
    h1, h2 = data[1][0], data[1][1]

    int_rectangle_h1 = integrate_rectangle_method(f, l, r, h1)
    int_rectangle_h2 = integrate_rectangle_method(f, l, r, h2)
    rectangle_err, rectangle_rr = runge_rombert_method(h1, h2, int_rectangle_h1, int_rectangle_h2, 2)

    int_trapeze_h1 = integrate_trapeze_method(f, l, r, h1)
    int_trapeze_h2 = integrate_trapeze_method(f, l, r, h2)
    trapeze_err, trapeze_rr = runge_rombert_method(h1, h2, int_trapeze_h1, int_trapeze_h2, 2)

    int_simpson_h1 = integrate_simpson_method(f, l, r, h1)
    int_simpson_h2 = integrate_simpson_method(f, l, r, h2)
    simpson_err, simpson_rr = runge_rombert_method(h1, h2, int_simpson_h1, int_simpson_h2, 4)

    with open('output.txt', 'w') as file:
        file.write(f"Rectangle method:\n")
        file.write(f"Step = {h1}: integral = {int_rectangle_h1}\n")
        file.write(f"Step = {h2}: integral = {int_rectangle_h2}\n")
        file.write(f"Error rate: = {abs(rectangle_err)}\n")
        file.write(f"More accurate integral (runge_rombert): = {rectangle_rr}\n\n")

        file.write(f"Trapeze method:\n")
        file.write(f"Step = {h1}: integral = {int_trapeze_h1}\n")
        file.write(f"Step = {h2}: integral = {int_trapeze_h2}\n")
        file.write(f"Error rate: = {abs(trapeze_err)}\n")
        file.write(f"More accurate integral (runge_rombert): = {trapeze_rr}\n\n")

        file.write(f"Simpson method:\n")
        file.write(f"Step = {h1}: integral = {int_simpson_h1}\n")
        file.write(f"Step = {h2}: integral = {int_simpson_h2}\n")
        file.write(f"Error rate: = {abs(simpson_err)}\n")
        file.write(f"More accurate integral (runge_rombert): = {simpson_rr}")

if __name__ == '__main__':
    main()
