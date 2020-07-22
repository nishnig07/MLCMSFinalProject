import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix


def gaussian_kernel(x, y, epsilon):
    distMatrix = distance_matrix(x, y)
    return np.exp(-np.square(distMatrix) / epsilon ** 2)


def kernel_gradient(x, y, epsilon):
    distMatrix = distance_matrix(x, y)
    ker_x = gaussian_kernel(x, y, epsilon)
    xmy = np.zeros((x.shape[1], x.shape[0], y.shape[0]))
    for k in range(x.shape[1]):
        xm = np.dot(x[:, k].reshape(-1, 1), np.ones((1, y.shape[0])))
        ym = np.dot(y[:, k].reshape(-1, 1), np.ones((1, x.shape[0]))).T
        xmy[k, :, :] = xm - ym

    res = np.zeros((x.shape[1], distMatrix.shape[0], distMatrix.shape[1]))
    for k in range(res.shape[0]):
        distMatrixK = xmy
        res[k, :, :] = (2 / epsilon ** 2) * ker_x * np.array(distMatrixK[k, :])
    return res


def equation_solve(x0, f0, x, y, gy, epsilon, L, noise=1e-5):
    ker_y = gaussian_kernel(y, y, epsilon) + noise ** 2 * np.identity(x.shape[0])
    ker_y_inv = np.linalg.pinv(ker_y)

    k_x0y = gaussian_kernel(x0, x, epsilon)
    ker_grad = kernel_gradient(x, x, epsilon)
    ker_grad = np.vstack(ker_grad)
    full_ker = np.dot(ker_grad, ker_y_inv)
    full_ker = np.vstack(full_ker)
    gy = gy.reshape((1250, 1))
    gy_new = np.row_stack([gy, f0])
    full_ker = np.row_stack([full_ker, k_x0y])
    Hy, _, _, _ = np.linalg.lstsq(full_ker, gy_new, rcond=None)

    phi = np.empty([1250, L])
    for eachpoint in range(L):
        phi_l = 2 * np.exp(-np.square((y - y[eachpoint]) / epsilon)) * (y - y[eachpoint]) / epsilon ** 2
        phi_l = phi_l.reshape((1250,))
        phi[:, eachpoint] = phi_l
    plt.show()
    approx_func_Ct = np.linalg.inv(phi.T @ phi) @ phi.T @ gy

    # Plot the approximated values from lstsq
    plt.figure()
    plt.contour(xx1, xx2, Hy.reshape((25, 25)))
    plt.show()

    # Plot the approximated values from RBF
    plt.figure()
    plt.contour(xx1, xx2, approx_func_Ct.reshape((25, 25)))
    plt.show()


def pendulum(yy1, yy2):
    return (yy2 ** 2) / 2 + (1 - np.cos(yy1))


def gradient(xx1, xx2):
    return np.array([-np.sin(xx1), xx2])


if __name__ == '__main__':
    Y_q, Y_p = np.linspace(-9, 9, 25), np.linspace(-1.5, 1.5, 25)
    X_q, X_p = np.linspace(-2 * np.pi, 2 * np.pi, 25), np.linspace(-1, 1, 25)
    xx1, xx2 = np.meshgrid(X_q, X_p)
    yy1, yy2 = np.meshgrid(Y_q, Y_p)
    x = np.column_stack([xx1.flatten(), xx2.flatten()])
    y = np.column_stack([yy1.flatten(), yy2.flatten()])

    x0 = np.array([.5, .4]).reshape(1, -1)
    f0 = pendulum(.4, .5)

    yr = np.zeros((625, 2))
    yr[:, 0] = np.random.uniform(low=np.min(Y_q), high=np.max(Y_q), size=(625))
    yr[:, 1] = np.random.uniform(low=np.min(Y_p), high=np.max(Y_p), size=(625))
    y = yr

    gx = gradient(xx1, xx2)

    equation_solve(x0, f0, x, y, gx, epsilon=2, noise=1e-5, L=625)
    data = pendulum(yy1, yy2)

    a, b = np.meshgrid(Y_q, Y_p)
    fig, ax = plt.subplots(1, 1, figsize=(7, 3), sharey=True)

    # Plot the original Hamiltonian
    plt.contour(a, b, data.reshape((25, 25)), linewidths=4)
    plt.show()
