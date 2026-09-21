import math


def f(x):
    return x / ((2 * x + 7) * (3 * x + 4))


def _interval_count(l, r, h):
    if not all(math.isfinite(value) for value in (l, r, h)) or h <= 0 or r <= l:
        raise ValueError("Require finite l < r and positive h")
    count = round((r - l) / h)
    if count < 1 or not math.isclose(count * h, r - l, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("Step must divide the integration interval")
    return count

def integrate_rectangle_method(f, l, r, h):
    count = _interval_count(l, r, h)
    return h * sum(f(l + (i + 0.5) * h) for i in range(count))

def integrate_trapeze_method(f, l, r, h):
    count = _interval_count(l, r, h)
    return h * (0.5 * (f(l) + f(r)) + sum(f(l + i * h) for i in range(1, count)))

def integrate_simpson_method(f, l, r, h):
    count = _interval_count(l, r, h)
    if count % 2:
        raise ValueError("Simpson's rule requires an even number of intervals")
    weighted_sum = sum((4 if i % 2 else 2) * f(l + i * h) for i in range(1, count))
    return h / 3 * (f(l) + weighted_sum + f(r))

def runge_rombert_method(h1, h2, integral1, integral2, p):
    """Поправка к интегралу на мелкой сетке и уточнённое значение."""
    if h1 > h2:
        coarse_h, fine_h, coarse, fine = h1, h2, integral1, integral2
    else:
        coarse_h, fine_h, coarse, fine = h2, h1, integral2, integral1
    correction = (fine - coarse) / ((coarse_h / fine_h)**p - 1)
    return correction, fine + correction

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
        file.write("Function: y = x / ((2*x + 7)*(3*x + 4))\n")
        file.write(f"Interval: [{l}, {r}]\n")
        file.write(f"Runge-Romberg error estimates refer to step {min(h1, h2)}.\n\n")
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
