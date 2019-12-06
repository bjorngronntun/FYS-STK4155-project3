import numpy as np
import matplotlib.pyplot as plt

# Analytical solution
def analytic(t_end, Nx, Nt):
    x_np = np.linspace(0, 1, Nx)
    t_np = np.linspace(0, t_end, Nt)

    X, T = np.meshgrid(x_np, t_np)

    x, t = X.ravel(), T.ravel()
    solution = np.exp(-(np.pi**2)*t)*np.sin(np.pi*x)
    return solution.reshape((Nt, Nx))

if __name__ == '__main__':
    t_end = 1
    Nx = 11
    Nt = 200
    analytic = analytic(t_end, Nx, Nt).reshape((Nt, Nx))
    print('Analytic shape', analytic.shape)
    print(analytic)
    for no, t_value in enumerate(np.linspace(0, t_end, Nt)):

        plt.plot(np.linspace(0, 1, Nx), analytic[no, :], label='t={}'.format(t_value))
    plt.legend()
    plt.grid(True)
    plt.show()
