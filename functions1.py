import numpy as np 
import sympy as sp 
import matplotlib.pyplot as plt 
from IPython.display import display
from sympy.physics.mechanics import msubs


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
global u1t, u2t, w1t, w2t; u1t, u2t, w1t, w2t = sp.symbols("u1t, u2t, w1t, w2t")

global w_eq; w_eq = sp.Matrix([u1 + sp.I*w1 , u2 + sp.I*w2 ])

global τ; τ = sp.Function("τ")(t, x, y, z, u1, u2, w1, w2)
global ξ; ξ = sp.Function("ξ")(t, x, y, z, u1, u2, w1, w2)
global γ; γ = sp.Function("γ")(t, x, y, z, u1, u2, w1, w2)
global ζ; ζ = sp.Function("ζ")(t, x, y, z, u1, u2, w1, w2)

global φ; φ = sp.Function("φ")(t, x, y, z, u1, u2, w1, w2)
global χ; χ = sp.Function("χ")(t, x, y, z, u1, u2, w1, w2)
global ψ; ψ = sp.Function("ψ")(t, x, y, z, u1, u2, w1, w2)
global ω; ω = sp.Function("ω")(t, x, y, z, u1, u2, w1, w2)

global lvar; lvar = [t, x, y, z]
global lwlf; lwlf = [u1, u2, w1, w2]
global lfn1; lfn1 = [τ, ξ, γ, ζ]
global lfn2; lfn2 = [φ, χ, ψ, ω]

def weyl_eq():
    psi = w_eq
    p1 = I2*psi.diff(t)
    p2 = sx*psi.diff(x)
    p3 = sy*psi.diff(y)
    p4 = sz*psi.diff(z)
    return p1+p2+p3+p4

#This creates a 4x4 array with the main weyl function components and their derivatives
def symbols_1():

    s_lwlf_d = [
        sp.symbols(f"{symbol}_{index}{component}")
        for symbol in ("u", "w")
        for index in range(1, 3)
        for component in ("t", "x", "y", "z")
        ]
    s_lwlf_d = np.reshape(s_lwlf_d, (4,4))
    return s_lwlf_d

#This creates a 8x8 array with the prolongation functions and their derivatives 
def symbols_2():
    s_lfnN_d = [
        sp.symbols(f"{symbol}_{index}")
        for symbol in ("τ", "ξ", "γ", "ζ", "φ", "χ", "ψ", "ω")
        for index in ("t", "x", "y", "z", "u1", "u2", "w1", "w2" )
        ]
    s_lfnN_d = np.reshape(s_lfnN_d, (8,8))
    return s_lfnN_d

#This is the main substitution helper function
#It simplifies most of the expressions from fractal derivatives into indexed variables
def subs_it(func):
    
    #load the indexed symbols
    sym1 = symbols_1()
    sym2 = symbols_2()

    #replace the derivatives
    k1 = 0
    k2 = 0    
    for i in lfn1+lfn2 :
        for j in lvar+lwlf :
            func = func.subs({
                sp.Derivative(i, j): sym2[k1, k2]
            })
            k2 += 1
        k2 = 0
        k1 += 1
    
    #then replace the other derivatives to avoid conflict
    k1 = 0
    k2 = 0         
    for n in lwlf:
        for m in lvar:
            func = func.subs(
                sp.Derivative(n, m), sym1[k1, k2]
            )
            k2 += 1
        k2 = 0 
        k1 += 1
            
    #this order in particular is important because of dependencies
    #unsure of how it affects the next substitutions
    func = func.subs({
    τ:sp.symbols("τ"),
    ξ:sp.symbols("ξ"),
    γ:sp.symbols("γ"),
    ζ:sp.symbols("ζ"),
    φ:sp.symbols("φ"),
    χ:sp.symbols("χ"),
    ψ:sp.symbols("ψ"),
    ω:sp.symbols("ω")
    })         
    func = func.subs({
    u1:sp.symbols("u_1"),
    u2:sp.symbols("u_2"),
    w1:sp.symbols("w_1"),
    w2:sp.symbols("w_2")
    })        
    return func 

def subs_parts(func):
    #this is for the 4 weyl functions
    #substitute the time derivatives with the other parts a.k.a solve for and subs
    sym1 = symbols_1()
    func = func.subs( sym1[0, 0],  (-sym1[0, 3] - sym1[1, 1] - sym1[3, 2] ))
    func = func.subs( sym1[2, 0],  ( sym1[1, 2] - sym1[2, 3] - sym1[3, 1] ))
    func = func.subs( sym1[1, 0],  (-sym1[0, 1] + sym1[1, 3] + sym1[2, 2] ))
    func = func.subs( sym1[3, 0],  (-sym1[0, 2] - sym1[2, 1] + sym1[3, 3] ))
    return func

"""
array([[u_1t, u_1x, u_1y, u_1z],
       [u_2t, u_2x, u_2y, u_2z],
       [w_1t, w_1x, w_1y, w_1z],
       [w_2t, w_2x, w_2y, w_2z]], dtype=object)
"""

#takes in the expression and substitutes the derivatives that are zero
#by getting the first results we can see which components are supposed to be zero
#to clean up results, such like τ_u1 being = 0. 
def zero_coef(expr):
    sym = symbols_2()
    sym_zero = sym.tolist()
    for i in range(0, 4): # u1,u2,w1,w2
        for j in range(4, 8):
            sym_zero[i][j] = 0
    
    for i in range(0, 8):
        for j in range(0, 8):
            expr = expr.subs( sym[i, j], sym_zero[i][j])     
    return expr

"""
array([[τ_t, τ_x, τ_y, τ_z,  0  ,  0  ,  0  ,  0  ],
       [ξ_t, ξ_x, ξ_y, ξ_z,  0  ,  0  ,  0  ,  0  ],
       [γ_t, γ_x, γ_y, γ_z,  0  ,  0  ,  0  ,  0  ],
       [ζ_t, ζ_x, ζ_y, ζ_z,  0  ,  0  ,  0  ,  0  ],
       [φ_t, φ_x, φ_y, φ_z, φ_u1, φ_u2, φ_w1, φ_w2],
       [χ_t, χ_x, χ_y, χ_z, χ_u1, χ_u2, χ_w1, χ_w2],
       [ψ_t, ψ_x, ψ_y, ψ_z, ψ_u1, ψ_u2, ψ_w1, ψ_w2],
       [ω_t, ω_x, ω_y, ω_z, ω_u1, ω_u2, ω_w1, ω_w2]], dtype=object)

"""
#the prolongations function
def pr1(phi_a, ua, J):
    p1 = (phi_a - (τ*sp.diff(ua, t) + ξ*sp.diff(ua, x) + γ*sp.diff(ua, y) + ζ*sp.diff(ua, z) ))
    #can be written better
    p2 = p1.diff(J)
    p3 = τ*sp.diff(ua, J, t) + ξ*sp.diff(ua, J, x) + γ*sp.diff(ua, J, y) + ζ*sp.diff(ua, J, z)
    #pJa = (p2+p3)*sp.diff(eq, ua.diff(J))
    pJa = (p2+p3)
    return pJa

