import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.distance
import scipy
import scipy.interpolate
import scipy.signal


def image_contour(ax, field, extent,
                  label=None, levels=None, cmap='viridis', colors=None, aspect='auto',
                  do_imshow=False,
                  do_remove_extreme=False,
                  # Like (batch_qp[:, 0], batch_qp[:, 1]))
                  training_data=None,
                  scatter_kw=dict(color='black', s=24, label='sample training batch'),
                  **kw_imshow
                  ):
    fig = ax.figure
    if do_imshow:
        imshow = ax.imshow(field, extent=extent, origin='lower', cmap=cmap, aspect='auto', **kw_imshow)
    if colors is None:
        colors = 'white'
    contour = ax.contour(field, extent=extent, levels=levels, cmap=cmap, linewidths=4)
    if do_imshow:
        mappable = imshow
    else:
        mappable = contour
    fig.colorbar(mappable, label=label, ax=ax)
    if training_data is not None:
        ax.scatter(*training_data, **scatter_kw)

    if do_remove_extreme:
        kw = do_remove_extreme if isinstance(do_remove_extreme, dict) else {}
        field = do_remove_extreme(field, **kw)


class GaussianProcess:
    def gaussian_kernel(x, y, epsilon):
        distMatrix = scipy.spatial.distance.cdist(x, y, 'euclidean')
        return np.exp(-np.array(distMatrix) ** 2 / epsilon ** 2)  # added epsilon_sq

    def gaussian_kernel_grad(xx, yy, epsilon):
        """ computes the gradient of the kernel w.r.t. the first argument """

        x = xx.copy()
        y = yy.copy()
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        if len(y.shape) == 1:
            y = y.reshape(1, -1)

        distMatrix = scipy.spatial.distance.cdist(x, y, 'euclidean')
        kxy = np.exp(-np.array(distMatrix) ** 2 / epsilon ** 2)

        xmy = np.zeros((x.shape[1], x.shape[0], y.shape[0]))
        if x.shape[1] == 1:
            xm = np.dot(xx, np.ones(y.shape).T)
            ym = np.dot(yy, np.ones(x.shape).T).T
            xmy[0, :, :] = xm - ym
        else:
            for k in range(x.shape[1]):
                xm = np.dot(xx[:, k].reshape(-1, 1), np.ones((1, yy.shape[0])))
                ym = np.dot(yy[:, k].reshape(-1, 1), np.ones((1, xx.shape[0]))).T
                xmy[k, :, :] = xm - ym
        res = np.zeros((x.shape[1], distMatrix.shape[0], distMatrix.shape[1]))
        for k in range(res.shape[0]):
            distMatrixK = xmy
            res[k, :, :] = (-2 / epsilon ** 2) * kxy * np.array(distMatrixK[k, :])
        return res

    def __init__(self,
                 data: np.array,
                 fdata: np.array,
                 theta: np.array,
                 kernel=gaussian_kernel,
                 kernel_grad=gaussian_kernel_grad,
                 output_noise_std=1e-5,
                 rcond=1e-10,  # used in lstsq
                 verbose=False) -> None:
        """ sets up the gaussian process on a list of points """

        data = data.copy()
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        self.__data = data
        self.__fdata = fdata
        self.__theta = theta
        self.__kernel = kernel
        self.__kernel_grad = kernel_grad
        self.__initialized = True
        self.__output_noise_std = output_noise_std
        self.__rcond = rcond

        if verbose and output_noise_std < 1e-5:
            print('Output noise std set to', output_noise_std,
                  'which might cause matrices to become non - positive definite.')

        if not (self.__fdata is None):
            self.__alpha, self.__L, self.__ll = self.data_cov(output_noise_std, verbose)

    def data_cov(self, output_noise_std):
        if (self.__initialized is None):
            raise ValueError('Must initialize first')
        k_xx = self.__kernel(self.__data, self.__data, self.__theta) + \
               output_noise_std ** 2 * np.identity(self.__data.shape[0])
        k_grad = self.__kernel_grad(self.__data, self.__data, self.__theta)
        L = np.linalg.cholesky(k_xx)
        alpha0, res0, _, _ = np.linalg.lstsq(L, self.__fdata, rcond=self.__rcond)
        alpha, res, _, _ = np.linalg.lstsq(L.T, alpha0, rcond=self.__rcond)
        logLikelihood = -1 / 2 * np.dot(self.__fdata.T, alpha) \
                        - np.sum(np.log(np.diag(L))) \
                        - L.shape[0] / 2 * np.log(2 * np.pi)

        return alpha, L, np.array(logLikelihood).flatten()[0]

    def solve(self, x0, f0, xnew, gnew):
        """ solve d/dx[f](ynew)=g(ynew) for f """

        if len(xnew.shape) == 1:
            xnew = xnew.reshape(1, -1)
        if len(gnew.shape) == 1:
            gnew = gnew.reshape(1, -1)
        if len(x0.shape) == 1:
            x0 = x0.reshape(1, -1)

        k_xx = self.__kernel(self.__data, self.__data, self.__theta) + \
               self.__output_noise_std ** 2 * np.identity(self.__data.shape[0])
        k_xxinv = np.linalg.pinv(k_xx)
        k_x0x = self.__kernel(x0, self.__data, self.__theta)
        k_grad = self.__kernel_grad(xnew, self.__data, self.__theta)
        mean_grad = np.zeros((xnew.shape[0] * xnew.shape[1], self.__data.shape[0]))
        for k in range(k_grad.shape[0]):
            kxxx = np.dot(k_grad[k,:,:], k_xxinv)

        mean_grad[(k * xnew.shape[0]):((k + 1) * xnew.shape[0]), :] = kxxx
        gnew = gnew.reshape(-1, 1)
        kx0xkinv = np.dot(k_x0x, k_xxinv)
        mean_grad = np.row_stack([mean_grad, kx0xkinv])
        gnew = np.row_stack([gnew, f0])
        fx, residuals, rank, singularValues = np.linalg.lstsq(mean_grad, gnew, rcond=self.__rcond)
        return fx


def solution1(x):
    """
    Method with original pendulum equation
    :param x: point x with q and p value
    :return: return the H value
    """
    return (x[:, 1] ** 2) / 2 + (1 - np.cos(x[:, 0]))


def g1(x):
    """
    Method to calculate the gradient
    :param x: point x with q and p value
    :return: returns an array of \dot{p} and \dot{p}
    """
    return np.array([-np.sin(x[:, 0]), x[:, 1]])


if __name__ == '__main__':
    np.random.seed(123495)
    noise = 1e-5
    N1, N2 = 100, 40
    K = 25

    # Form the data points in X and Y
    train_q = np.linspace(-2 * np.pi, 2 * np.pi, K) + np.random.randn(K, ) * noise * 1e-3
    train_p = np.linspace(-1, 1, K) + np.random.randn(K, ) * noise * 1e-3
    test_q = np.linspace(-9, 9, N1) + np.random.randn(N1, ) * noise * 1e-3
    test_p = np.linspace(-1.5, 1.5, N2) + np.random.randn(N2, ) * noise * 1e-3

    xx1, xx2 = np.meshgrid(test_q, test_p)
    yy1, yy2 = np.meshgrid(train_q, train_p)
    x = np.column_stack([xx1.flatten(), xx2.flatten()])
    y = np.column_stack([yy1.flatten(), yy2.flatten()])

    yr = np.zeros((K * K, 2))
    yr[:, 0] = np.random.uniform(low=np.min(train_q), high=np.max(train_q), size=(K * K))
    yr[:, 1] = np.random.uniform(low=np.min(train_p), high=np.max(train_p), size=(K * K))
    y = yr

    # Calculate the gradients
    gy = g1(y)
    epsilon = 2
    x00 = np.array([.5, .4]).reshape(1, -1)
    f0 = solution1(x00)

    # solve the PDE
    gp = GaussianProcess(y, None, theta=epsilon, output_noise_std=noise)
    Hy = gp.solve(x00, f0, y, gy, solver_tolerance=1e-6)

    data = solution1(x).reshape((N2, N1))
    ex = [np.min(x[:, 1]), np.max(x[:, 1]), np.min(x[:, 0]), np.max(x[:, 0])]
    ey = [np.min(x[:, 0]), np.max(x[:, 0]), np.min(x[:, 1]), np.max(x[:, 1])]
    field1 = data
    extent = ey
    figsize = (7, 3)

    # Plot the original Hamiltonian pendulum
    fig, ax = plt.subplots(1, 1, figsize=figsize, sharey=True)
    image_contour(ax, field1, extent, label='H(q,p)', levels=None, cmap='viridis', colors=None,
                  aspect=2.5, do_imshow=False, do_remove_extreme=False, training_data=None,
                  scatter_kw=dict(color='black', s=12, label='sample training batch'))
    ax.set_ylabel('p')
    ax.set_xlabel('q')
    plt.show()

    # Plot the approximated solutions
    plt.figure()
    plt.contour(yy1, yy2, Hy.reshape((-1, yy1.shape[1])))
    plt.show()
