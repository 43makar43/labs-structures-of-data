def extended_gcd(a, b):
    if b == 0:
        return a, 1, 0
    d, x, y = extended_gcd(b, a % b)
    return d, y, x - (a // b) * y

print("Введите два числа: ")
a, b = int(input()), int(input())
d, x, y = extended_gcd(a, b)


print(f"Ответ: {d, x, y}")
print("Выполнил студент группы ПОВв-з-24 Макаров Даниил Денисович")
