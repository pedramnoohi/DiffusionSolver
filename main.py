import numpy as np
import scipy.optimize
import math
import matplotlib.pyplot as plt
class TimeDiscrete:
    def discrete_time(self):
        #h=(self.finT-self.initT)/self.nSteps
        #tHist= np.linspace(self.init,self.finT,self.nSteps+1)
        tHist=np.zeros(self.nSteps+1)
        return tHist
    def __init__(self,tspan,nSteps,**kwargs):
        self.init=tspan[0]
        self.finT=tspan[1]
        self.nSteps=nSteps
        super().__init__(**kwargs)
class SpaceDiscrete:
    '''
    Discretizing space
    '''
    def discrete_space(self):
        span=np.linspace(0,1,self.s_steps+2)
        span=span[1:self.s_steps+1]
        f=self.f_initial
        u0=f(span)
        return u0
    def __init__(self,s_steps,L,f_initial,**kwargs):
        self.s_steps=s_steps
        #L is length of x domain from 0 to L
        self.L=L
        #initital condition function, solving heat equation requires at t=0 to have space dimesnion to have assigned
        #values,
        self.f_initial=f_initial
        super().__init__(**kwargs)
class CrankNicholson(SpaceDiscrete,TimeDiscrete):
    def __init__(self,tspan,nSteps,s_steps,f_initial,L):

        super().__init__(tspan=tspan,nSteps=nSteps,s_steps=s_steps,f_initial=f_initial,L=L)


    def second_der(self):
        s_steps=self.s_steps

        D = np.zeros([s_steps, s_steps])
        for j in [i for i in range(0, s_steps)]:
            D[j, j] = -2
            if j == 0:
                D[j, j + 1] = 1
            elif j == s_steps-1:
                D[j, j - 1] = 1
            else:
                D[j, j - 1] = 1
                D[j, j + 1] = 1
        
        #For L= 1
        h=1/(s_steps +1)
        D=D*(1/(h**2))
        D[0, s_steps - 1] = 1
        D[s_steps - 1, 0] = 1
        return D
    def CrankFunction(self,t,U):
        self.t=t
        self.U=U

        return np.matmul(self.second_der(),U)
    def calculate(self):
        u0=self.discrete_space()
        print(u0.shape)
        p = len(u0)
        s_steps=self.s_steps
        # Pre-allocation
        tHist=self.discrete_time()

        nsteps=self.nSteps
        uHist = np.zeros((nsteps + 1, p))
        u = u0
        t = self.init  # Initial time
        h = (self.finT - self.init) / nsteps #time steps
        uHist[0, :] = (u)
        f=self.CrankFunction
        print(self.CrankFunction(t + h, u))
        for n in [i for i in range(0, nsteps)]:
            def g(v):
                l = u
                out = v - (h / 2) * f(t + h, v) - l - (h / 2) * (f(t, l))

                return out

            V = scipy.optimize.fsolve(g, u)


            u=V
            t += h

            tHist[n+1]=t
            uHist[n+1, :] = u

        return [tHist, uHist]
    def plot(self):
        ax = plt.axes(projection='3d')
        span = np.linspace(0, 1, self.s_steps + 2)
        span = span[1:self.s_steps + 1]
        [thists,UHist]=self.calculate()

        X,Y =np.meshgrid(thists,span)
        ax.plot_surface(X,Y,UHist.transpose())
        plt.title('My title')
        plt.xlabel('time')
        plt.ylabel('space')
        plt.show()

def f(x):return math.sin(2*(math.pi)*x)
f=np.vectorize(f)

[tHist,uHist]=CrankNicholson([0,1],100,30,f,1).calculate()

ax=plt.axes(projection='3d')
CrankNicholson([0,0.5],100,30,f,1).plot()
