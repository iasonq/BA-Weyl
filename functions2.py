import numpy as np
from numpy.core.numeric import True_ 
import sympy as sp 


#this is the second substitution function
#substitutes the terms that subs_it cant substitute
#placed here to avoid conflict with global variables
def subs_der(func):
    
    sym_der = [
        sp.symbols(f"{symbol}_{index}")
        for symbol in ("τ", "ξ", "γ", "ζ", "φ", "χ", "ψ", "ω")
        for index in ("t", "x", "y", "z" )
        ]
    sym_der = np.reshape(sym_der, (8,4))
    t, x, y, z, c = sp.symbols("t x y z c")
    u1 = sp.symbols("u_1")
    u2 = sp.symbols("u_2")
    w1 = sp.symbols("w_1")
    w2 = sp.symbols("w_2")
    τ = sp.Function("τ")(t, x, y, z, u1, u2, w1, w2)
    ξ = sp.Function("ξ")(t, x, y, z, u1, u2, w1, w2)
    γ = sp.Function("γ")(t, x, y, z, u1, u2, w1, w2)
    ζ = sp.Function("ζ")(t, x, y, z, u1, u2, w1, w2)
    φ = sp.Function("φ")(t, x, y, z, u1, u2, w1, w2)
    χ = sp.Function("χ")(t, x, y, z, u1, u2, w1, w2)
    ψ = sp.Function("ψ")(t, x, y, z, u1, u2, w1, w2)
    ω = sp.Function("ω")(t, x, y, z, u1, u2, w1, w2)
    
    lvar = [t, x, y, z]
    lfn1 = [τ, ξ, γ, ζ]
    lfn2 = [φ, χ, ψ, ω]
    
    k1 = 0
    k2 = 0    
    for i in lfn1+lfn2 :
        for j in lvar :
            func = func.subs( sp.Derivative(i, j), sym_der[k1, k2])
            k2 += 1
        k2 = 0
        k1 += 1
    
    return func 


def sort_coef(expr, sym1):
    
    last_eq = expr.copy()
    for k in range(0, 4):
        for l in range(1, 4):
            last_eq = last_eq.subs( sym1[k, l], 0 )
    last_eq

    lista = []
    for k in range(0, 4):
        for l in range(1, 4):
            #substract the last_eq every time because we dont need it sitting at the bac
            #every time
            expr_zer = expr.copy() - last_eq
            for i in range(0, 4):
                for j in range(1, 4):
                    #substitute all coefficient with 0
                    #except the one we are interested in
                    if sym1[i, j] != sym1[k, l]:
                        expr_zer = expr_zer.subs( sym1[i, j], 0 )
            #replace the coefficient we are interested in with 1
            #to obtain a clean final expression
            expr_zer = expr_zer.subs( sym1[k, l], 1 )
            #substitute the last fractal derivatives 
            expr_zer = subs_der(expr_zer)
            lista.append(expr_zer)
    #don't forget the last_eq!
    lista.append(subs_der(last_eq))
    #in one case scenario we generate a 0 because one coefficient doesnt show up at all
    #so 0 comes out
    lista.remove(0)
    return lista

#linearly combine elements
def lin_comb(exp, sym2, col1, col2):
    plus = []
    minus = []
    #counters 
    cp = 0
    cm = 0
    for eq1 in exp[0:, col1]:
        for eq2 in exp[0:, col2]:
            tempp = eq1 + eq2
            tempm = eq1 - eq2
            for i in range(0, 8):
                for check in sym2[i]:
                    if tempp.has(check):
                        cp = cp + 1
                    if tempm.has(check):
                        cm = cm + 1
            if (cp <= 3) and (tempp != 0): 
                plus.append(tempp)
            if (cm <= 3) and (tempm != 0):
                minus.append(tempm)
            cp = 0
            cm = 0
            
    total = np.array(plus + minus)
    return total 

#remove redundancies
def rm_red(exp):
    rows = np.shape(exp)[0]
    cols = np.shape(exp)[1]
    
    for l in range(0, cols):
        for eq in exp[0:, l]:
            for j in range(0, rows):
                for k in range(l+1, cols):
                    if eq == exp[j, k] or eq == -1*exp[j, k]:
                        exp[j, k] = 0
    return exp

#clear zeros

def clear_zer(exp):
    rows = np.shape(exp)[0]
    cols = np.shape(exp)[0]
    l = []
    for i in range(0, rows):
        for j in range(0, cols):
            if exp[i, j] != 0:
                l.append(exp[i, j])
    l = np.array(l)
    return l

#takes 1-Dim
def lin_comb_t(exp, sym2):
    s = np.size(exp)
    plus = []
    minus = []
    #counters 
    cp = 0
    cm = 0
    for i in range(0, s):
        for j in range(i+1, s):
            eq1 = exp[i]
            eq2 = exp[j]
            tempp = eq1 + eq2
            tempm = eq1 - eq2
            for k in range(0, 8):
                for check in sym2[k]:
                    if tempp.has(check):
                        cp = cp + 1
                    if tempm.has(check):
                        cm = cm + 1
            if (cp <= 4) and (tempp != 0): 
                plus.append(tempp)
            if (cm <= 4) and (tempm != 0):
                minus.append(tempm)
            cp = 0
            cm = 0
            
    total = np.array(plus + minus)
    return total 

#takes 1-Dim
#remove redundancies
def rm_red_t(exp):
    s = np.size(exp)
    for i in range(0, s):
        if exp[i] != 0:
            for j in range(i+1, s):
                if exp[i] == exp[j] or exp[i] == -1*exp[j]:
                    exp[j] = 0
    return exp

#takes 1d array
#clear Trues
def clear_true(exp):
    s = np.size(exp)
    l = []
    for i in range(0, s):
        if exp[i] != True :
            l.append(exp[i])
    return l

#takes 1-Dim
#clear zeros
def clear_zer_t(exp):
    s = np.size(exp)
    l = []
    for i in range(0, s):
        if exp[i] != 0 or exp[i] != True :
            l.append(exp[i])
    #l = np.array(l)
    return l

#make 1-Dim
def flatten(expr):
    rows = np.shape(expr)[0]
    cols = np.shape(expr)[1]
    expr = expr.reshape(1, rows*cols)
    return expr

#take only expressions with 2 or 3 terms
def sort_23(expr, sym2, num):
    sort = []
    c = 0
    for exp in expr:    
        for k in range(0, 8):
            for check in sym2[k]:
                if exp.has(check):
                    c = c + 1
        if (c == num): 
            sort.append(exp)   
        c = 0
    return sort 

#useless shit
def divide_coef(expr, sym2, num):
    s = np.size(expr)
    #counter symbol
    cs = 0
    #counter numbers
    cn = 0
    for i in range(0, s):    
        for k in range(0, 8):
            for check in sym2[k]:
                if expr[i].has(check):
                    cs = cs + 1
                if expr[i].has(num):
                    cn = cn + 1
        if (cs == cn): 
            expr[i] = expr[i]/num  
        cs = 0
        cn = 0
    return expr

#a bit of a fail, made a better one for p2
#helper function to make dictionaries
def make_dicti(expr, sym2):
    #load an array with 2-terms
    dicti = {}
    for exp in expr:    
        flag = False
        for k in range(0, 8):
            for check in sym2[k]:
                if exp.has(check) and (flag == False):
                    #check if it the negative version of number
                    if exp.has(-1*check):
                        dicti[ exp + check ] = check
                        flag = True
                    else:
                        dicti[ exp - check ] = -1*check
                        flag = True
    return dicti
