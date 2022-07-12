import numpy as np
import scipy.optimize
import math
import matplotlib.pyplot as plt
class TimeDiscrete:
    '''
    Discretizes Time domain.
    Class gets inherited by CrankNicholson.

    Args:
        tspan(lst): includes initial and final time.
        t_steps(int): number of steps to break time domain up into.
        **kwargs: The keyword arguments are used fo CrankNicholson.

    Attributes:
        init(int): initial time
        finT(int): final time
    '''
    def discrete_time(self):
        '''
        Discretizes time, given certain number of steps.

        Returns:
            tHist(lst): list of of every time point in domain, from 0 to t_steps.
        '''

        #h=(self.finT-self.initT)/self.t_steps
        #tHist= np.linspace(self.init,self.finT,self.t_steps+1)
        tHist=np.zeros(self.t_steps+1)
        return tHist
    def __init__(self,tspan,t_steps,**kwargs):

        self.init=tspan[0]
        self.finT=tspan[1]
        self.t_steps=t_steps
        super().__init__(**kwargs)
class SpaceDiscrete:
    '''
    Discretize Space, and provides x values at inital time, given some initial condition function.

    This class is intended to be subclassed by CrankNicholson.

    Args:
        L(lst): Length of spatial domain.
        s_steps(int): Number of steps to discretize space domain (excluding boundary conditions).
        f_initial(fun):Initial condition function for time=0.

    Attributes:
        L(lst): This is where length is stored
        s_steps(int): This is where number of steps are stored.
        f_initial(fun): Where initial condition function is stored.

    '''
    def discrete_space(self):
        '''
        Discretizes space, given certain number of steps.

        Returns:
            u0(lst): list including every spatial point, given initial condition function f_initial
        '''

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
    '''
    Inherits all methods from SpaceDiscrete, Time Discrete. Applies cranknicholson scheme for diffusion problem,
    and plots it.Inherits all arguments and attributes from parent classes.

    Args:
        constant(float): Diffusion constant

    Attributes:
        constant(float): where diffusion constant is stored
    '''
    def __init__(self,tspan,t_steps,s_steps,f_initial,L,constant):
        self.constant=constant
        super().__init__(tspan=tspan,t_steps=t_steps,s_steps=s_steps,f_initial=f_initial,L=L)


    def second_der(self):
        '''
        Creates second differentiation matrix given s_steps, applies boundary conditions to them

        Returns:
            D(matrix): 2nd differentiation matrix including boundary conditions.

        '''
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
        '''
        Returns the next array of values in the CrankNicholson scheme.

        Args:
            t(int): takes time at a specific point in CrankNicholson scheme.
            U(int): current array of spatial values at a given time.
        Returns:
            v(array): Array v that is the next array of values in crank nicholson scheme iteration

        '''
        self.t=t
        self.U=U
        v=np.matmul(self.second_der(), self.constant * U)
        return v
    def calculate(self):
        '''
        Returns the next array of values in the CrankNicholson scheme.


        Returns:
            tHist(array): all time values from initial to final.
            uHist(matrix): matrix of values, where each column is one discrete point on the spatial domain,and each row
                        shows us its updated value at a given time.

        '''
        u0=self.discrete_space()
        print(u0.shape)
        p = len(u0)
        s_steps=self.s_steps
        # Pre-allocation
        tHist=self.discrete_time()

        t_steps=self.t_steps
        uHist = np.zeros((t_steps + 1, p))
        u = u0
        t = self.init  # Initial time
        h = (self.finT - self.init) / t_steps #time steps
        uHist[0, :] = (u)
        f=self.CrankFunction
        print(self.CrankFunction(t + h, u))
        for n in [i for i in range(0, t_steps)]:
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
        '''
        Plots the approximations.

        Returns:
            ax(plt): plot of diffusion equation
        '''
        ax = plt.axes(projection='3d')
        span = np.linspace(0, 1, self.s_steps + 2)
        span = span[1:self.s_steps + 1]
        [thists,UHist]=self.calculate()

        X,Y =np.meshgrid(thists,span)
        ax.plot_surface(X,Y,UHist.transpose())
        plt.title('Diffusion')
        plt.xlabel('time')
        plt.ylabel('space')
        plt.show()



