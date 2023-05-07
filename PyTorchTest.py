"""# Beispiel 1.2
import torch

x = torch.tensor(2., requires_grad=True)
y = torch.log(x**2 + 1) / torch.sqrt(x**2 + 1 + x)

y0 = y.item()
y.backward()
print(y0)
print(x.grad)
"""


"""# Beispiel 5.2
import torch

def f(x):
    v1 = x[0] * x[1]**2
    v2 = 2**x[1] / x[2]
    v3 = 2 / x[2]**2
    y = v1 + v2 - v3
    return y

x0 = [3., 2., -1.]
x0 = torch.tensor(x0, requires_grad=True)

y = f(x0)
y0 = y.item() # Funktionswert aus Tensor extrahieren
y.backward()  # AAD anwenden
print(y0)
print(x0.grad)
"""

