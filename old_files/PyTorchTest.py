"""# Beispiel 1.2
import torch

x0 = torch.tensor(2., requires_grad=True)
y = torch.log(x0**2 + 1) / torch.sqrt(x0**2 + 1 + x0)

y.backward() # Ableitungen mit AAD berechnen
dy = x0.grad # Ableitung von dy nach dx0

# Werte extrahieren
y0 = y.item()
dy0 = dy.item()

print(y0)
print(dy0)
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


"""# Beispiel 4.2
import torch
from torch.autograd.functional import jacobian

def f(x):
    y1 = x[0]*torch.sqrt(x[1]) + 3*x[1]
    y2 = torch.cos(x[0]) / x[1]
    y3 = torch.exp(x[0]**2 * x[1])
    return torch.stack([y1, y2, y3])

x0 = torch.tensor([2., 1.], requires_grad=True)

y0 = f(x0)
j = jacobian(f, x0)

print(y0)
print(j)
"""