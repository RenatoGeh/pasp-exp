from problog.logic import Term
from problog.program import PrologString
from problog.learning import lfi
import pasp
from random import random, randint
import timeit
import time
import numpy as np
import pandas as pd

def randomDigit():
    digit = randint(0, 10)
    if digit == 0:
        return [1, 1, 1, 0, 1, 1, 1], 0
    if digit == 1:
        return [0, 0, 1, 0, 0, 1, 0], 1
    if digit == 2:
        return [1, 0, 1, 1, 1, 0, 1], 2
    if digit == 3:
        return [1, 0, 1, 1, 0, 1, 1], 3
    if digit == 4:
        return [0, 1, 1, 1, 0, 1, 0], 4
    if digit == 5:
        return [1, 1, 0, 1, 0, 1, 1], 5
    if digit == 6:
        return [1, 1, 0, 1, 1, 1, 1], 6
    if digit == 7:
        return [1, 0, 1, 0, 0, 1, 0], 7
    if digit == 8:
        return [1, 1, 1, 1, 1, 1, 1], 8
    else:
        return [1, 1, 1, 1, 0, 1, 0], 9
    
def noise(digits, perturbation=0.05):
    return [d if random() > perturbation else (d + 1) % 2 for d in digits]

N_SAMPLES = 1_000
N_ITERS = 1
samples_pasp = []
samples_pl = []

for _ in range(N_SAMPLES):
    real, digit = randomDigit()
    digit_pasp = [1 if d == digit else 0 for d in range(10)]
    observed = noise(real)
    samples_pasp.append(real + observed + digit_pasp)
    
    real_problog = [(Term(f'real({i})'), True) if d else (Term(f'real({i})'), False) for i, d in enumerate(real)]
    observed_problog = [(Term(f'observed({i})'), True) if d else (Term(f'observed({i})'), False) for i, d in enumerate(observed)]
    digit_problog = [(Term(f'digit({d})'), True) if d == digit else (Term(f'digit({d})'), False) for d in range(10)]
    samples_pl.append(real_problog + observed_problog + digit_problog)

P = pasp.parse('led_logic.lp')
A = [f"real({r})" for r in range(7)] + [f"observed({o})" for o in range(7)] + [f"digit({d})" for d in range(10)]
start_time = timeit.default_timer()
pasp.learn(P, samples_pasp, A, niters = N_ITERS)
print(f'PASP time = {timeit.default_timer() - start_time}')
print(f'PASP Program = {P}')

df = pd.DataFrame(data=samples_pasp, columns=A)
df.to_csv("led.csv", index=False)

model = """
real(0). 
real(1). 
real(2). 
real(3). 
real(4). 
real(5). 
real(6).

t(_)::observed(0). 
t(_)::observed(1). 
t(_)::observed(2). 
t(_)::observed(3). 
t(_)::observed(4). 
t(_)::observed(5). 
t(_)::observed(6).

t(_):: observed(0) :- real(0).
t(_):: observed(1) :- real(1).
t(_):: observed(2) :- real(2).
t(_):: observed(3) :- real(3).
t(_):: observed(4) :- real(4).
t(_):: observed(5) :- real(5).
t(_):: observed(6) :- real(6).
t(_):: observed(7) :- real(7).

digit(0) :- real(0), real(1), real(2), real(4), real(5), real(6).
digit(1) :- real(2), real(5).
digit(2) :- real(0), real(2), real(3), real(4), real(6).
digit(3) :- real(0), real(2), real(3), real(5), real(6).
digit(4) :- real(1), real(2), real(3), real(4).
digit(5) :- real(0), real(1), real(3), real(5), real(6).
digit(6) :- real(0), real(1), real(3), real(4), real(5), real(6).
digit(7) :- real(0), real(2), real(5).
digit(8) :- real(0), real(1), real(2), real(3), real(4), real(5), real(6).
digit(9) :- real(0), real(1), real(2), real(3), real(5), real(6).
"""

prolog_model = PrologString(model)
start_time = timeit.default_timer()
score, weights, atoms, iteration, lfi_problem = lfi.run_lfi(prolog_model, samples_pl)
print(f'Problog time = {timeit.default_timer() - start_time}')