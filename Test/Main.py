
import numpy as np 
import sympy as sp 
import matplotlib.pyplot as plt 
from functions import *

global t, x, y, z; t, x, y, z, c = sp.symbols("t x y z c")

global u1; u1 = sp.Function("u_1")(t, x, y, z)
global u2; u2 = sp.Function("u_2")(t, x, y, z)
global w1; w1 = sp.Function("w_1")(t, x, y, z)
global w2; w2 = sp.Function("w_2")(t, x, y, z)


global I2; I2 = sp.Matrix([[1, 0], [0, 1]])
global sx; sx = sp.Matrix([[0, 1], [1, 0]])
global sy; sy = sp.Matrix([[0, -sp.I], [sp.I, 0]])
global sz; sz = sp.Matrix([[1, 0], [0, -1]])

global u1x, u1y, u1z; u1x, u1y, u1z = sp.symbols("u_1x, u_1y, u_1z")
global u2x, u2y, u2z; u2x, u2y, u2z = sp.symbols("u_2x, u_2y, u_2z")
global w1x, w1y, w1z; w1x, w1y, w1z = sp.symbols("w_1x, w_1y, w_1z")
global w2x, w2y, w2z; w2x, w2y, w2z = sp.symbols("w_2x, w_2y, w_2z")
global u1t, u2t, w1t, w2t; u1t, u2t, w1t, w2t = sp.symbols("u_1t, u_2t, w_1t, w_2t")

global w_eq; w_eq = sp.Matrix([u1 + sp.I*w1 , u2 + sp.I*w2 ])

#these are the ξ^i basically
global τ; τ = sp.Function("τ")(t, x, y, z, u1, u2, w1, w2)
global ξ; ξ = sp.Function("ξ")(t, x, y, z, u1, u2, w1, w2)
global γ; γ = sp.Function("γ")(t, x, y, z, u1, u2, w1, w2)
global ζ; ζ = sp.Function("ζ")(t, x, y, z, u1, u2, w1, w2)

#these are the u^a basically
global φ; φ = sp.Function("φ")(t, x, y, z, u1, u2, w1, w2)
global χ; χ = sp.Function("χ")(t, x, y, z, u1, u2, w1, w2)
global ψ; ψ = sp.Function("ψ")(t, x, y, z, u1, u2, w1, w2)
global ω; ω = sp.Function("ω")(t, x, y, z, u1, u2, w1, w2)


lvar = sp.Matrix([t, x, y, z])
lwlf = sp.Matrix([u1, u2, w1, w2])
lfn1 = sp.Matrix([τ, ξ, γ, ζ])
lfn2 = sp.Matrix([φ, χ, ψ, ω])
#In our case it correlated with lvar, because we only have derivatives of the first order
ders = sp.Matrix([t, x, y, z])


sym1 = symbols_1()
sym2 = symbols_2()


sp.pprint(sp.Matrix([[u1 + sp.I*w1], [u2 + sp.I*w2] ]))


weyl = subs_it(weyl_eq())
weyl0 = subs_it(weyl[0])
weyl1 = subs_it(weyl[1])

# --------------------EQUATION 1--------------------

eq1 = sp.simplify(weyl0).subs({sp.I:0})
eq1


pr1a = pr1(φ, u1, t)
pr1b = pr1(φ, u1, z)
pr1c = pr1(χ, u2, x)
pr1d = pr1(ω, w2, y)


expr1 = sp.simplify(subs_it(pr1a + pr1b + pr1c + pr1d) )
expr1 = sp.expand(expr1)
expr1 = sp.expand(subs_parts(expr1))
expr1 = sp.simplify(zero_coef(expr1))

# --------------------EQUATION 2--------------------

eq2 = sp.simplify(-1*sp.I*weyl0).subs({sp.I:0})
eq2


pr2a = pr1(χ, u2, y) #u2y
pr2b = pr1(ψ, w1, t) #w1t
pr2c = pr1(ψ, w1, z) #w1z
pr2d = pr1(ω, w2, x) #w2z


expr2 = sp.simplify(subs_it(-pr2a + pr2b + pr2c + pr2d) )
expr2 = sp.expand(expr2)
expr2 = sp.expand(subs_parts(expr2))
expr2 = sp.simplify(zero_coef(expr2))

# --------------------EQUATION 3--------------------

eq3 = sp.simplify(weyl1).subs({sp.I:0})
eq3


pr3a = pr1(φ, u1, x) #u1x
pr3b = pr1(χ, u2, z) #u2z
pr3c = pr1(ψ, w1, y) #w1y
pr3d = pr1(χ, u2, t) #u2t


expr3 = sp.simplify(subs_it(pr3a - pr3b - pr3c + pr3d) )
expr3 = sp.expand(expr3)
expr3 = sp.expand(subs_parts(expr3))
expr3 = sp.simplify(zero_coef(expr3))

# --------------------EQUATION 4--------------------

eq4 = sp.simplify(-1*sp.I*weyl1).subs({sp.I:0})
eq4


pr4a = pr1(φ, u1, y) #u1y
pr4b = pr1(ψ, w1, x) #w1x
pr4c = pr1(ω, w2, t) #w2t
pr4d = pr1(ω, w2, z) #w2z


expr4 = sp.simplify(subs_it(pr4a + pr4b + pr4c - pr4d) )
expr4 = sp.expand(expr4)
expr4 = sp.expand(subs_parts(expr4))
expr4 = sp.simplify(zero_coef(expr4))

expressions = [expr1, expr2, expr3, expr4]
num = 1
for exp in expressions:
    for k in range(0, 4):
        for l in range(1, 4):
            expr_zer = exp.copy()
            for i in range(0, 4):
                for j in range(1, 4):
                    if sym1[i, j] != sym1[k, l]:
                        expr_zer = expr_zer.subs( sym1[i, j], 0 )
            print("Expression No", num)
            print("Coefficient ", sym1[k, l] )
            sp.pprint(expr_zer)
    print("---------------------------------------------------------")
    num = num + 1