import itertools
import sys
import math
import random

import clingo
import pasp

def create_rules(n: int) -> str:
  P = ""
  for i in range(n):
    P += f"sleep{i} :- not work{i}, not insomnia{i}.\nwork{i} :- not sleep{i}.\n\n"
  return P

def create(n: int, pr: list = None, full: bool = False) -> str:
  P = create_rules(n)
  if pr is None:
    for i in range(n): P += f"insomnia{i}; "
    P += f"insomnia{n}."
  elif full:
    for i in range(len(pr)-1): P += f"{pr[i]}::insomnia{i}; "
    P += f"{pr[len(pr)-1]}::insomnia{len(pr)-1}."
  else:
    for i in range(n): P += f"{pr[i]}::insomnia{i}; "
    P += f"{round(1.0-sum(pr), ndigits = 10)}::insomnia{n}."
  return P

def create_pasp(n: int, pr: list, full: bool = False) -> str:
  P = create(n, pr = pr, full = full) + "\n\n"
  for x in itertools.product([False, True], repeat = 2*n):
    P += "#query("
    for i in range(0, len(x), 2):
      P += f"{'' if x[i] else 'not '}sleep{i//2}, {'' if x[i+1] else 'not '}work{i//2}" + \
        (", " if i < len(x)-2 else "")
    P += ").\n"
  return P

def sample(n: int, m: int, pr: list) -> list:
  """ Samples m "work" and "sleep" atoms from the n-insomnia program. """
  P = create_pasp(n, pr)
  R = [x[0] for x in pasp.exact(pasp.parse(P, from_str = True), psemantics = "maxent")]
  D = []
  for i, x in enumerate(itertools.product([False, True], repeat = 2*n)):
    k = round(m*R[i])
    D.extend([x]*k)
  return D

def val2text(x: tuple) -> str:
  X = ""
  for i in range(0, len(x), 2):
    j = i // 2
    s = f"sleep{j}" if x[i] else f"not sleep{j}"
    w = f"work{j}" if x[i+1] else f"not work{j}"
    X += f"{s}, {w}, "
  return X[:-2]

def count(D: list) -> dict:
  M = {}
  for x in D:
    k = tuple(x)
    if k not in M: M[k] = 1
    else: M[k] += 1
  return M

def val2dict(x: tuple) -> dict:
  X = {}
  for i in range(0, len(x), 2):
    j = i // 2
    X[f"sleep{j}"] = x[i]
    X[f"work{j}"] = x[i+1]
  return X

def val2obs(x: tuple) -> list:
  X = []
  for i in range(0, len(x), 2):
    j = i // 2
    if x[i]: X.append(f"sleep{j}")
    if x[i+1]: X.append(f"work{j}")
  return X

def undef_atom_ignore(x, y):
  if x == clingo.MessageCode.AtomUndefined: return
  print(y, file = sys.stderr)

def num_models(P: str, atom: str) -> int:
  C = clingo.Control(["0"], logger = undef_atom_ignore)
  C.add("base", [], P + f"{atom}.")
  C.ground([("base", [])])
  n = 0
  with C.solve(yield_ = True) as h:
    for m in h:
      n += 1
  return n

def prob(P: str, O: list, theta: float, theta_f: str, n: float) -> float:
  if type(O) != tuple and type(O) != list: O = [O]
  for o in O: P += f":- not {o}."
  C = clingo.Control(["0"], logger = undef_atom_ignore)
  C.add("base", [], P + f"{theta_f}.")
  C.ground([("base", [])])
  p = 0
  with C.solve(yield_ = True) as h:
    for m in h:
      p += theta
  return p/n

def prob_obs(P: str, O: list, theta: list, theta_f: list, N: list) -> float:
  if type(O) != tuple and type(O) != list: O = [O]
  for o in O: P += f":- not {o}."
  p_o = 0
  for i in range(len(theta_f)):
    C = clingo.Control(["0"], logger = undef_atom_ignore)
    C.add("base", [], P + f"{theta_f[i]}.")
    C.ground([("base", [])])
    p = 0
    with C.solve(yield_ = True) as h:
      for m in h:
        p += theta[i]
    p_o += p/N[i]
  return p_o

def ll(R: list):
  return sum(math.log(x[0]) for x in R if x[0] != 0)

def ll_from(n: int, pr: list, full: bool = False):
  return ll(pasp.exact(pasp.parse(create_pasp(n, pr = pr, full = full), from_str = True), psemantics = "maxent"))

def learn(D: list, n: int, theta: list = None, n_iters: int = 1, H: list = None) -> list:
  """
  Learn using the softmax derivation as an inference:

  P(θ = i) = (∑_O P(θ = i, O)/P(O))/|O|.
  """
  m = n+1
  if theta is None: theta = [1/m for _ in range(m)]
  atoms = [f"insomnia{i}" for i in range(m)]
  R = create_rules(m-1)
  N = [num_models(R, a) for a in atoms]
  W = [0 for _ in range(m)]
  C = count(D)
  k = len(D)
  if H is not None:
    H.append(W)
  for it in range(n_iters):
    for i in range(m):
      W[i] = 0
      for O in C:
        p = prob(R, O, theta[i], atoms[i], N[i])
        q = prob_obs(R, O, theta, atoms, N)
        t = p / q
        W[i] += t*C[O]
      W[i] /= k
    if H is not None:
      H.append(W)
    print(W)
    theta = W.copy()
  return theta

def softmax(W: list) -> list:
  E = [math.exp(w) for w in W]
  n = len(W)
  s = sum(E)
  for i in range(n):
    E[i] /= s
  return E


def learn_neurasp(D: list, n: int, theta: list = None, n_iters: int = 1, H: list = None, \
                  eta: float = 0.01) -> list:
  """
  Learn using NeurASP's derivation.
  """
  m = n+1
  if theta is None: theta = [1/m for _ in range(m)]
  atoms = [f"insomnia{i}" for i in range(m)]
  R = create_rules(m-1)
  N = [num_models(R, a) for a in atoms]
  C = count(D)
  ones = [1 for j in range(m-1)]
  W = theta.copy()

  for it in range(n_iters):
    for O in C:
      j = 1
      for i in range(m):
        p = prob(R, O, 1, atoms[i], N[i])
        q = prob_obs(R, O, ones, [atoms[j] for j in range(m) if j != i], \
                     [N[j] for j in range(m) if j != i])
        o = prob_obs(R, O, theta, atoms, N)
        W[i] += C[O]*eta*(p-q)/o
    for i in range(m): theta[i] = W[i]
    print(theta)
  return theta

def learn_lagrange(D: list, n: int, theta: list = None, n_iters: int = 1, H: list = None, \
                   eta: float = 0.1) -> list:
  m = n+1
  if theta is None: theta = [1/m for _ in range(m)]
  atoms = [f"insomnia{i}" for i in range(m)]
  R = create_rules(m-1)
  N = [num_models(R, a) for a in atoms]
  C = count(D)
  ones = [1 for j in range(m-1)]
  W = theta.copy()
  f = 1/m

  for it in range(n_iters):
    for O in C:
      j = 1
      for i in range(m):
        p = prob(R, O, 1, atoms[i], N[i])
        q = prob_obs(R, O, ones, [atoms[j] for j in range(m) if j != i], \
                     [N[j] for j in range(m) if j != i])
        o = prob_obs(R, O, theta, atoms, N)
        W[i] += C[O]*eta*((1-f)*p-f*q)/o
    for i in range(m): theta[i] = W[i]
    print(theta)
  return theta
