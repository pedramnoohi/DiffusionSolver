import Solver
import math
import numpy as np
def f(x):return math.sin(2*(math.pi)*x)
f=np.vectorize(f)

#[tHist,uHist]=CrankNicholson([0,1],100,30,f,1,2)


Solver.CrankNicholson([0,1000],1000,30,f,1,0.0001).plot()
