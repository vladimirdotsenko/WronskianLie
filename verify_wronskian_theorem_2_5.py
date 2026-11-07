#!/usr/bin/env python3
"""Exact, reproducible verification of the computational claims in Theorem 2.5.

Requires SymPy (e.g. `python3 -m pip install sympy`).
Run: `python3 verify_wronskian_theorem_2_5.py`

The variables x(t), y(t) are generic differential indeterminates, and
    a o b = a * d(b)/dt,   [a,b] = a o b - b o a.
All calculations use exact integer polynomial arithmetic, not numerical tests.
The mathematical argument that v1,...,v4 span the relevant homogeneous
component of the Lie ideal is in the paper, not established by this script.
"""

import sympy as sp


t = sp.Symbol("t")
x = sp.Function("x")(t)
y = sp.Function("y")(t)


def novikov(a, b):
    return a * sp.diff(b, t)


def bracket(a, b):
    return novikov(a, b) - novikov(b, a)


def verify(condition, description):
    if not condition:
        raise AssertionError(description)
    print("PASS:", description)


# The elements r, s, w from Theorem 2.5.
xy = bracket(x, y)
r = bracket(xy, bracket(xy, y))
s = bracket(bracket(xy, bracket(xy, x)), x)
w = bracket(xy, bracket(bracket(xy, x), bracket(xy, y)))

# Equation (1) of the paper: w belongs to the Novikov ideal (r,s).
rhs = (
    novikov(novikov(bracket(r, y), x), x)
    + bracket(novikov(x, r), novikov(x, y))
    + bracket(y, novikov(s, y))
    - 2 * novikov(novikov(y, s), y)
)
verify(sp.expand(w - rhs) == 0, "equation (1): w equals its Novikov-ideal expression")

# The four possible Lie-ideal generators in bidegree (4,4).
v1 = bracket(bracket(bracket(r, y), x), x)
v2 = bracket(bracket(bracket(r, x), y), x)
v3 = bracket(bracket(bracket(r, x), x), y)
v4 = bracket(bracket(s, y), y)

# The differential monomials chosen in the paper.
M1 = x**4 * y * sp.diff(y, t)**2 * sp.diff(y, t, 5)
M2 = x**4 * sp.diff(y, t)**3 * sp.diff(y, t, 4)
M3 = x**2 * sp.diff(x, t, 2) * sp.diff(x, t, 3) * y**2 * sp.diff(y, t)**2
monomials = (M1, M2, M3)


def coefficient_row(polynomial):
    expanded = sp.expand(polynomial)
    return tuple(expanded.coeff(m) for m in monomials)


names = ("v1", "v2", "v3", "v4", "w")
polynomials = (v1, v2, v3, v4, w)
rows = [coefficient_row(p) for p in polynomials]
expected_rows = [
    (4, 2, 6),
    (3, 0, 10),
    (2, -2, 14),
    (-1, 1, -7),
    (0, 1, -1),
]
verify(rows == expected_rows, "every coefficient in the paper's 5-by-3 table")

print("\n           M1   M2   M3")
for name, row in zip(names, rows):
    print(f"{name:>3s}:      {str(row[0]):>3s}  {str(row[1]):>3s}  {str(row[2]):>3s}")

# Membership would imply A*(alpha1,...,alpha4)^T = b.
# The M3 coefficient in w is -1, NOT +1 (a typo in the Mathematica notebook).
A = sp.Matrix([[rows[j][i] for j in range(4)] for i in range(3)])
b = sp.Matrix(rows[4])
verify(A.rank() == 2, "rank of the coefficient matrix is 2")
verify(A.row_join(b).rank() == 3, "the augmented matrix has rank 3: no solution")

# A short exact certificate: -10*M1 + 11*M2 + 3*M3 vanishes
# on v1,v2,v3,v4 but evaluates to 8 on w.
certificate = sp.Matrix([[-10, 11, 3]])
verify(certificate * A == sp.zeros(1, 4), "the certificate annihilates all four Lie-ideal candidates")
verify((certificate * b)[0] == 8, "the certificate evaluates to 8 on w")
print("\nConclusion: w is not in the span of v1,v2,v3,v4 (over characteristic zero).")
print("The free-Lie-ideal spanning statement used to finish the proof is in Theorem 2.5.")
