import math
import random
import itertools
import torch
import pickle
import torchvision
import numpy as np

IMG_S = (1, 28, 28)

def mnist(n: int):
  train = torchvision.datasets.MNIST(root="/tmp", train=True, download=True)
  test = torchvision.datasets.MNIST(root="/tmp", train=False, download=True)
  which_train = train.targets <= n
  which_test = test.targets <= n
  m_train, m_test = torch.sum(which_train).item(), torch.sum(which_test).item()
  return train.data[which_train].float().reshape(m_train, *IMG_S)/255., train.targets[which_train], \
         test.data[which_test].float().reshape(m_test, *IMG_S)/255., test.targets[which_test]

def indices(X: torch.tensor, n: int) -> list[range]:
  return [range(i*(s := X.shape[0]//(n*n)), (i+1)*s) for i in range(n*n)]

def empty(e: torch.tensor = torch.zeros(*IMG_S)) -> torch.tensor:
  if not hasattr(empty, "__single"): empty.__single = e
  return empty.__single
if hasattr(empty, "__single"): delattr(empty, "__single")

def squarify(X: torch.tensor, n: int, rand_empty: int = 3, **kwargs) -> torch.tensor:
  E = np.random.choice(n*n, size=rand_empty, replace=False)
  return torch.cat(tuple(torch.cat(tuple(empty(**kwargs) if (u := i+n*j) in E else X[u] \
                                         for i in range(n)), dim=2)
                         for j in range(n)), dim=1)

def isls(X: list) -> bool:
  "Returns true if the given square is a latin square; false otherwise."
  n = len(X)
  m = int(math.sqrt(n))
  for i in range(m):
    cols, rows = 0, 0
    for j in range(m):
      u = 1 << X[i+m*j]
      v = 1 << X[i*m+j]
      if cols & u: return False
      if rows & v: return False
      cols |= u
      rows |= v
  return True

def squares(n: int) -> list[list]:
  class Pair:
    def __init__(self, a, b): self.a, self.b = a, b

  def perm_unique(elements):
    S = set(elements)
    U = [Pair(i, elements.count(i)) for i in S]
    u = len(elements)
    return perm_unique_helper(U, [0]*u, u-1)

  def perm_unique_helper(U, result_list, d):
    if d < 0:
      yield tuple(result_list)
    else:
      for i in U:
        if i.b > 0:
          result_list[d] = i.a
          i.b -= 1
          for g in perm_unique_helper(U, result_list, d-1):
            yield g
          i.b += 1

  L, S = [], [i+1 for _ in range(n) for i in range(n)]
  for p in perm_unique(S):
    if isls(p): L.append(p)
  return L

def mnist_squares(X: torch.tensor, Y: torch.tensor, L: list, m: int, **kwargs) -> torch.tensor:
  n = int(math.sqrt(len(L[0])))
  I_D = {i: torch.where(Y == i)[0] for i in range(n+1)}
  imgs, targets = [], []
  for i in range(m):
    l = L[i % len(L)]
    I = [I_D[d][random.randint(0, len(I_D[d])-1)].item() for d in l]
    digits = X[I]
    imgs.append(squarify(digits, n, e = X[I_D[0][0].item()], **kwargs))
    targets.append(l)
  return torch.cat(imgs).reshape(len(imgs), 1, IMG_S[1]*n, IMG_S[2]*n), targets

def save(f_train: str, f_test: str, n: int, m_train: int, m_test: int, **kwargs):
  X_R, Y_R, X_T, Y_T = mnist(n)
  L = squares(n)
  imgs_train, targets_train = mnist_squares(X_R, Y_R, L, m_train, **kwargs)
  imgs_test, targets_test = mnist_squares(X_T, Y_T, L, m_test, **kwargs)
  torch.save(imgs_train, f"{f_train}.pt")
  torch.save(imgs_test, f"{f_test}.pt")
  with open(f"{f_train}.pk", "wb") as f: pickle.dump(targets_train, f)
  with open(f"{f_test}.pk", "wb") as f: pickle.dump(targets_test, f)
  return imgs_train, targets_train, imgs_test, targets_test

np.random.seed(101)
