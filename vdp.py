import numpy as np

from casadi import Opti, vertcat


def get_next(X_k, u_prev=0, x_ref=0, dt=0.5, N=5):
    opti = Opti()  # optimization problem

    X = opti.variable(2,N+1)  # state variables for all k (including 0)
    opti.subject_to(opti.bounded(-3,X,3))

    U = opti.variable(1,N+1)  # control action
    del_U = opti.variable(1,N+1)
    opti.subject_to(opti.bounded(-1,U,1))

    # objective
    N_1 = 1
    N_2 = N
    N_u = N_2

    Q = 10
    R = 1

    error = X[0,N_1:N_2+1] - x_ref
    opti.minimize(  # 5a
        (error @ error.T) * Q + 
        (del_U[:N_u+1] @ del_U[:N_u+1].T) * R
    )

    # van der pol oscillator
    f = lambda x, u: vertcat(x[1], (1 - x[0]**2) * x[1] - x[0] + u)

    for k in range(N):  # 5b using Runge-Kutta
        k1 = f(X[:,k],U[:,k])
        k2 = f(X[:,k] + (dt / 2) * k1, U[:,k])
        k3 = f(X[:,k] + (dt / 2) * k1, U[:,k])
        k4 = f(X[:,k] + dt * k3, U[:,k])

        x_next = X[:,k] + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

        opti.subject_to(X[:,k+1] == x_next)

    for j in range(N_u+1):  # 5c
        opti.subject_to(
            U[j] == u_prev + del_U[:j+1] @ np.ones(j+1)
        )

    # 5d is not necessary as N_u == N_2

    # initial conditions
    opti.subject_to(X[:,0] == X_k)

    opti.solver(  # set numerical backend
        'ipopt',
        dict(verbose=False, print_time=False), dict(print_level=0)  # silence
    )

    sol = opti.solve()  # actual solve

    return sol.value(U), sol.value(X)