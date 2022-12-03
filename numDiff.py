'''
def f(x):
    y = x ** 2
    return y

def fdot(f, x0, h):
    df = (f(x0 + h) - f(x0)) / h
    return df

x0 = 0.2
H = [10 ** -9, 10 ** -10, 10 ** -11, 10 ** -12]
for h in H:
    ydot = fdot(f, x0, h)
    print("h = " + str(h) + "\t=> f'(x0) = " + str(ydot))
'''

x = -2
y = 1 if x>0 else -1

print(y)