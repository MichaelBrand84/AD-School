from sympy  import symbols, diff

def f(x):
    v1 = x ** 2
    v2 = 1 / x
    y = v1 + v2
    return y

x = symbols('x')
print("f(x) =", f(x))

df = diff(f(x),x)
print("f'(x) =", df)

x0 = 2
print("f'(" + str(x0) + ") =", df.evalf(subs={x:2}))