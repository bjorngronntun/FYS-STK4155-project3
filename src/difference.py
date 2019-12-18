import numpy as np
import matplotlib.pyplot as plt


def difference(t_end, Nx, Nt):
    """
    Computes solution of heat equation with predefined initial/boundary
    conditions, using forward-difference method
    Inputs:         t_end       float, max value of t
                    Nx          int, number of grid lines in x direction
                    Nt          int, number of grid lines in t direction

    Returns:        solution    np array, function values on grid
    """
    x_np = np.linspace(0, 1, Nx)
    t_np = np.linspace(0, t_end, Nt)

    X, T = np.meshgrid(x_np, t_np)

    x, t = X.ravel(), T.ravel()

    solution = np.zeros((Nt, Nx))

    h = 1/(Nx - 1)
    k = (t_end)/(Nt - 1)
    # Initial condition
    solution[0, :] = np.sin(np.pi*x_np)

    for i in range(0, Nt - 1):
        for j in range(1, Nx - 1):
            solution[i + 1, j] = solution[i, j] + (k/h**2)*(solution[i, j + 1] - 2*solution[i, j] + solution[i, j - 1])
    return solution

if __name__ == '__main__':
    t_end = 1
    Nx = 11
    Nt = 200
    solution = difference(t_end, Nx, Nt)
    print(solution)
    for no, t_value in enumerate(np.linspace(0, t_end, Nt)):
        print(no)
        plt.plot(np.linspace(0, 1, Nx), solution[no, :], label='t={}'.format(t_value))
    plt.legend()
    plt.grid(True)
    plt.show()
