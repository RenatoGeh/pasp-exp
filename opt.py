import itertools
import sys
import math
import random

import numpy as np
import matplotlib.pyplot as plt
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

def create_pasp(n: int, pr: list, full: bool = False, marg: bool = False) -> str:
  P = create(n, pr = pr, full = full) + "\n\n"
  if marg:
    for i in range(n):
      P += f"#query(sleep{i}).\n"
  else:
    for x in itertools.product([False, True], repeat = 2*n):
      P += "#query("
      for i in range(0, len(x), 2):
        P += f"{'' if x[i] else 'not '}sleep{i//2}, {'' if x[i+1] else 'not '}work{i//2}" + \
          (", " if i < len(x)-2 else "")
      P += ").\n"
  return P

def sample(n: int, m: int, pr: list, marg: bool = False) -> list:
  """ Samples m "work" and "sleep" atoms from the n-insomnia program. """
  P = create_pasp(n, pr, marg = marg)
  R = [x[0] for x in pasp.exact(pasp.parse(P, from_str = True), psemantics = "maxent")]
  D = []
  if marg:
    for i, p in enumerate(R): D.extend([f"sleep{i}"]*round(m*p))
  else:
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

def val2obs(x: tuple, marg: bool = False) -> list:
  if marg: return [x]
  X = []
  for i in range(0, len(x), 2):
    j = i // 2
    if x[i]: X.append(f"sleep{j}")
    if x[i+1]: X.append(f"work{j}")
  return X

def dataset(n: int, p: list, marg: bool = False):
  return [val2obs(x, marg = marg) for x in sample(len(p), n, p, marg = marg)]

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

def ll_r(C: list, P: str, theta: list, theta_f: list, N: list):
  if min(theta) < 0 or max(theta) > 1: return -10
  return sum(C[O]*math.log(prob_obs(P, O, theta, theta_f, N)) for O in C)/sum(C[O] for O in C)

def ll(C: list, n: int, theta: list):
  P = create_rules(n)
  theta_f = [f"insomnia{i}" for i in range(n+1)]
  N = [num_models(P, t) for t in theta_f]
  return ll_r(C, P, theta, theta_f, N)

def learn(D: list, n: int, theta: list = None, n_iters: int = 1, H: list = None, \
          debug: bool = False, H_ll: list = None) -> list:
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
  if H is not None: H.append(theta.copy())
  if H_ll is not None: H_ll.append(ll_r(C, R, theta, atoms, N))
  for it in range(n_iters):
    for i in range(m):
      W[i] = 0
      for O in C:
        p = prob(R, O, theta[i], atoms[i], N[i])
        q = prob_obs(R, O, theta, atoms, N)
        t = p / q
        W[i] += t*C[O]
      W[i] /= k
    if H is not None: H.append(W.copy())
    if debug: print(W)
    for i in range(m): theta[i] = W[i]
    if H_ll is not None: H_ll.append(ll_r(C, R, theta, atoms, N))
  return theta

def softmax(W: list) -> list:
  E = [math.exp(w) for w in W]
  n = len(W)
  s = sum(E)
  for i in range(n):
    E[i] /= s
  return E


def learn_neurasp(D: list, n: int, theta: list = None, n_iters: int = 1, H: list = None, \
                  eta: float = 0.01, debug: bool = False,  H_ll: list = None) -> list:
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
  if H is not None: H.append(theta.copy())
  if H_ll is not None: H_ll.append(ll_r(C, R, theta, atoms, N))

  for it in range(n_iters):
    for O in C:
      j = 1
      for i in range(m):
        p = prob(R, O, 1, atoms[i], N[i])
        q = prob_obs(R, O, ones, [atoms[j] for j in range(m) if j != i], \
                     [N[j] for j in range(m) if j != i])
        o = prob_obs(R, O, theta, atoms, N)
        #d = min(1, max(-1, (p-q)/o))
        W[i] += C[O]*eta*((p-q)/o)
    for i in range(m): theta[i] = W[i]
    #theta = softmax(W)
    # s = 0.0
    # for i in range(m):
      # if W[i] > 0: s += W[i]
    # for i in range(m): theta[i] = W[i]/s if W[i] > 0 else 0
    # W = theta.copy()
    if H is not None: H.append(theta.copy())
    if H_ll is not None: H_ll.append(ll_r(C, R, theta, atoms, N))
    if debug: print(theta)
  return theta

def learn_lagrange(D: list, n: int, theta: list = None, n_iters: int = 1, H: list = None, \
                   eta: float = 0.1, debug: bool = False, H_ll: list = None) -> list:
  m = n+1
  if theta is None: theta = [1/m for _ in range(m)]
  atoms = [f"insomnia{i}" for i in range(m)]
  R = create_rules(m-1)
  N = [num_models(R, a) for a in atoms]
  C = count(D)
  ones = [1 for j in range(n)]
  W = theta.copy()
  f = 1/m
  if H is not None: H.append(theta.copy())
  if H_ll is not None: H_ll.append(ll_r(C, R, theta, atoms, N))

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
    if H is not None: H.append(theta.copy())
    if H_ll is not None: H_ll.append(ll_r(C, R, theta, atoms, N))
    if debug: print(theta)
  return theta

def history_2d(n: int, p: list, n_iters: int = 30, eta: float = 0.01, D: list = None, \
               theta_0: list = None):
  if D is None: D = dataset(n, p)
  H_inf, H_neurasp, H_lagrange = [], [], []
  H_ll_inf, H_ll_neurasp, H_ll_lagrange = [], [], []
  learn(D, 1, n_iters = n_iters, H = H_inf, H_ll = H_ll_inf, theta = theta_0.copy())
  learn_neurasp(D, 1, n_iters = n_iters, H = H_neurasp, H_ll = H_ll_neurasp, eta = eta, \
                theta = theta_0.copy())
  learn_lagrange(D, 1, n_iters = n_iters, H = H_lagrange, H_ll = H_ll_lagrange, eta = eta, \
                 theta = theta_0.copy())
  f = lambda h, i: [e for e in map(lambda t: t[i], h)]
  X = [f(H_inf, 0), f(H_neurasp, 0), f(H_lagrange, 0)]
  Y = [f(H_inf, 1), f(H_neurasp, 1), f(H_lagrange, 1)]
  return X, Y, [H_ll_inf, H_ll_neurasp, H_ll_lagrange]

def history_3d(n: int, p: list, n_iters: int = 30, eta: float = 0.01, D: list = None, \
               theta_0: list = None):
  if D is None: D = dataset(n, p)
  H_inf, H_neurasp, H_lagrange = [], [], []
  H_ll_inf, H_ll_neurasp, H_ll_lagrange = [], [], []
  learn(D, 2, n_iters = n_iters, H = H_inf, H_ll = H_ll_inf, theta = theta_0.copy())
  learn_neurasp(D, 2, n_iters = n_iters, H = H_neurasp, H_ll = H_ll_neurasp, eta = eta, \
                theta = theta_0.copy())
  learn_lagrange(D, 2, n_iters = n_iters, H = H_lagrange, H_ll = H_ll_lagrange, eta = eta, \
                 theta = theta_0.copy())
  f = lambda h, i: [e for e in map(lambda t: t[i], h)]
  X = [f(H_inf, 0), f(H_neurasp, 0), f(H_lagrange, 0)]
  Y = [f(H_inf, 1), f(H_neurasp, 1), f(H_lagrange, 1)]
  Z = [f(H_inf, 2), f(H_neurasp, 2), f(H_lagrange, 2)]
  return X, Y, Z, [H_ll_inf, H_ll_neurasp, H_ll_lagrange]

def feasible_set_support(n: int, k: int):
  A = np.diag(np.ones(k)) if n == 1 else np.flip(np.tri(k, k).T, axis = 1)
  A[A == 0] = np.nan
  return A

def gradient(n: int, C: list, P: str, theta: list, theta_f: list, N: list, alg: str, \
             eta: float = 0.01):
  m = n + 1
  nabla = [0 for _ in range(m)]
  ones = [1 for _ in range(n)]
  f = 1/m
  k = sum(C[O] for O in C)
  is_fixed_point = (alg != "neurasp") and (alg != "lagrange")
  if is_fixed_point:
    for i in range(m):
      for O in C:
        p = prob(P, O, theta[i], theta_f[i], N[i])
        q = prob_obs(P, O, theta, theta_f, N)
        nabla[i] = C[O]*(p / q)
      nabla[i] /= k
  else:
    for O in C:
      for i in range(m):
        p = prob(P, O, 1, theta_f[i], N[i])
        q = prob_obs(P, O, ones, [theta_f[j] for j in range(m) if j != i], \
                     [N[j] for j in range(m) if j != i])
        o = prob_obs(P, O, theta, theta_f, N)
        if alg == "neurasp": nabla[i] += eta*C[O]*((p-q)/o)
        else: nabla[i] += eta*C[O]*(((1-f)*p-f*q)/o)
  return nabla

def plot_ll_surf(n: int, D: list, k: int = 100, n_iters: int = 30, eta: float = 0.001):
  P = create_rules(n)
  theta_f = [f"insomnia{i}" for i in range(n+1)]
  N = [num_models(P, t) for t in theta_f]
  C = count(D)
  theta = [0 for _ in range(n+1)]

  T = ["blue", "red", "green"]
  L = ["Fixed-point", "NeurASP", "Lagrange"]
  W = [l.lower() for l in L]

  fig, ax = plt.subplots(subplot_kw = {"projection" : "3d"})

  LL = feasible_set_support(n, k)
  X, Y = np.meshgrid(np.linspace(0.0, 1.0, k), np.linspace(0.0, 1.0, k))
  print("Computing LL and gradient map...")
  for i, j in np.argwhere(~np.isnan(LL)):
    theta[0], theta[1] = X[i,j], Y[i,j]
    theta[2] = 1-(theta[0]+theta[1])
    if theta[2] == 0: LL[i,j] = np.nan
    else:
      LL[i,j] = ll_r(C, P, theta, theta_f, N)
      if (i % 5 == 0) and (j % 5 == 0):
        for l in range(len(W)):
          nabla = gradient(n, C, P, theta, theta_f, N, W[l], eta = eta)
          norm = math.sqrt(nabla[0]**2 + nabla[1]**2 + nabla[2]**2)
          if norm > 0.5: continue
          ax.quiver(theta[0], theta[1], LL[i,j], nabla[0], nabla[1], nabla[2], color = T[l], \
                    length = norm*0.75 if l == 1 else norm*2)

  ax.set_xlabel("ℙ(insomnia0)")
  ax.set_ylabel("ℙ(insomnia1)")
  ax.set_zlabel("Log-likelihood")

  ax.set_xlim(0, 1)
  ax.set_ylim(0, 1)
  ax.set_zlim(-1.75, -1.3)
  print("Plotting LL surface...")
  surf = ax.plot_surface(X, Y, LL, color = "gray", alpha = 0.3)
  ax.scatter(0.3, 0.2, ll_r(C, P, [0.3, 0.2, 0.5], theta_f, N), color = "#000000", label = "Target")

  print("Retrieving learning history...")
  theta_X, theta_Y, theta_Z, H = history_3d(n, None, n_iters = n_iters, eta = eta, D = D, \
                                            theta_0 = [0.1, 0.8, 0.1])
  print("Plotting learning history...")
  for i in range(len(theta_X)):
    ax.plot3D(theta_X[i], theta_Y[i], H[i], T[i], label = L[i])

  ax.legend()
  plt.show()

  return LL, theta_X, theta_Y, theta_Z, H

def plot_2d(n: int, p: list, n_iters: int = 30, eta: float = 0.01, theta_0: list = None):
  ax = plt.figure().add_subplot(projection = "3d")
  ax.set_xlabel("# iterations")
  ax.set_ylabel("ℙ(insomnia0)")
  ax.set_zlabel("ℙ(insomnia1)")
  ax.set_ylim(0, 1)
  ax.set_zlim(0, 1)

  X, Y, LL = history_2d(n, p, n_iters = n_iters, eta = eta, theta_0 = theta_0)
  k = len(X)
  I = [[x for x in range(n_iters+1)] for _ in range(k)]
  T = ["blue", "red", "green"]
  L = ["Inference", "NeurASP", "Lagrange"]

  for i in range(k):
    ax.plot3D(I[i], X[i], Y[i], T[i], label = L[i])
  ax.plot3D(I[i], np.array([p[0] for _ in range(n_iters+1)]), \
            np.array([1-p[0] for _ in range(n_iters+1)]), "gray", alpha = 0.7, label = "gray")

  ax.legend()

  ax = plt.figure().add_subplot()
  ax.set_xlabel("# iterations")
  ax.set_ylabel("LL(θ)")

  for i in range(len(LL)):
    ax.plot(LL[i], T[i], label = L[i])

  ax.legend()
  plt.show()
  return X, Y

def plot_3d(n: int, p: list, n_iters: int = 30, eta: float = 0.01, theta_0: list = None):
  ax = plt.figure().add_subplot(projection = "3d")
  ax.set_xlabel("# iterations")
  ax.set_ylabel("ℙ(insomnia0)")
  ax.set_zlabel("ℙ(insomnia1)")
  ax.set_ylim(0, 1)
  ax.set_zlim(0, 1)

  X, Y, Z, LL = history_3d(n, p, n_iters = n_iters, eta = eta, theta_0 = theta_0)
  k = len(X)
  I = [[x for x in range(n_iters+1)] for _ in range(k)]
  T = ["blue", "red", "green"]
  L = ["Inference", "NeurASP", "Lagrange"]

  for i in range(k):
    ax.plot3D(I[i], X[i], Y[i], T[i], label = L[i])
  ax.plot3D(I[i], np.array([p[0] for _ in range(n_iters+1)]), \
            np.array([p[1] for _ in range(n_iters+1)]), "gray", alpha = 0.7, label = "Target")

  ax.legend()

  ax = plt.figure().add_subplot()
  ax.set_xlabel("# iterations")
  ax.set_ylabel("LL(θ)")
  ax.set_ylim(-1.35, -1.3)

  for i in range(len(LL)):
    ax.plot(LL[i], T[i], label = L[i])

  ax.legend()
  plt.show()
  return X, Y, Z
