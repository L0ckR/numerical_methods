def df(x_test, x, y):
    for i in range(len(x) - 2):
        if x[i] <= x_test <= x[i + 1]:
            break
    a1 = (y[i + 1] - y[i]) / (x[i + 1] - x[i])
    a2 = ((y[i + 2] - y[i + 1]) / (x[i + 2] - x[i + 1]) - a1)
    a2 = a2 / (x[i + 2] - x[i]) * (2 * x_test - x[i] - x[i + 1])
    return a1 + a2


def d2f(x_test, x, y):
    for i in range(len(x) - 2):
        if x[i] <= x_test <= x[i + 1]:
            break
    num = (y[i + 2] - y[i + 1]) / (x[i + 2] - x[i + 1]) - (y[i + 1] - y[i]) / (x[i + 1] - x[i])
    return 2 * num / (x[i + 2] - x[i])


def main():
    with open('input.txt', 'r') as file:
        data = [list(map(float, line.split())) for line in file.readlines()]
    x = data[0]
    y = data[1]
    x_test = data[2][0]

    with open('output.txt', 'w') as file:
        file.write(f"First derivative:\ndf({x_test}) = {df(x_test, x, y)}\n\n")
        file.write(f"Second derivative:\nd2f({x_test}) = {d2f(x_test, x, y)}")


if __name__ == '__main__':
    main()
