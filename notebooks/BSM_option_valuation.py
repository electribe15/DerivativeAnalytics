

import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['font.family'] = 'serif'
from scipy.integrate import quad


def dN(x):
 return math.exp(-0.5 * x ** 2) / math.sqrt(2 * math.pi)

def N(d):
 return quad(lambda x: dN(x), -20, d, limit=50)[0]

def d1f(St, K, t, T, r, sigma):
 d1 = (math.log(St / K) + (r + 0.5 * sigma ** 2)
* (T - t)) / (sigma * math.sqrt(T - t))
 return d1

def BSM_call_value(St, K, t, T, r, sigma):
 d1 = d1f(St, K, t, T, r, sigma)
 d2 = d1 - sigma * math.sqrt(T - t)
 call_value = St * N(d1) - math.exp(-r * (T - t)) * K * N(d2)
 return call_value

def BSM_put_value(St, K, t, T, r, sigma):
 put_value = BSM_call_value(St, K, t, T, r, sigma)
 - St + math.exp(-r * (T - t)) * K
 return put_value