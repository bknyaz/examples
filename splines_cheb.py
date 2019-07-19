import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plot_basis(basis, title, ax):
    # Plot basis    
    x = np.linspace(-1, 1, basis.shape[1])
    for i in range(basis.shape[0]):
        y = np.zeros(len(x)) + i
        ax.plot(y, x, basis[i, :])
    ax.set_title(title, fontsize=16)
#     # Verify orthogonality
#     plt.show()
#     plt.imshow(np.dot(basis, basis.T))
#     plt.show()

def bspline_basis(n_splines, degree, n_points=100):
    # Modified from https://github.com/mdeff/cnn_graph/blob/master/lib/models.py
    # Create knot vector and a range of samples on the curve
    assert n_splines > degree
    kv = np.array([0] * degree + list(range(n_splines - degree + 1)) +
                  [n_splines - degree] * degree, dtype='int')  # knot vector
    u = np.linspace(0, n_splines - degree, n_points)  # samples range

    def coxDeBoor(k, d):
        # Test for end conditions
        if (d == 0):
            return ((u - kv[k] >= 0) & (u - kv[k + 1] < 0)).astype(int)

        denom1 = kv[k + d] - kv[k]
        term1 = 0
        if denom1 > 0:
            term1 = ((u - kv[k]) / denom1) * coxDeBoor(k, d - 1)

        denom2 = kv[k + d + 1] - kv[k + 1]
        term2 = 0
        if denom2 > 0:
            term2 = ((-(u - kv[k + d + 1]) / denom2) * coxDeBoor(k + 1, d - 1))

        return term1 + term2

    # Compute basis for each point
    b = np.column_stack([coxDeBoor(k, degree) for k in range(n_splines)])
    b[n_points - 1][-1] = 1
    
    return b.T


def chebyshev_basis(K, n_points=100):
    x = np.linspace(-1, 1, n_points)    
    # Create basis T
    T = np.zeros((K, len(x)))
    T[0,:] = 1
    T[1,:] = x
    for n in range(1, K-1):
        T[n+1, :] = 2*x*T[n, :] - T[n-1, :]    
    return T

fig = plt.figure(figsize=(12,5))
ax = fig.add_subplot(1, 2, 1, projection='3d')
basis = bspline_basis(4, 1)
plot_basis(basis, 'Spline basis (%d splines, degree=%d)' % (4, 1), ax)
ax = fig.add_subplot(1, 2, 2, projection='3d')
basis = bspline_basis(5, 3)
plot_basis(basis, 'Spline basis (%d splines, degree=%d)' % (5, 3), ax)
plt.tight_layout()   
plt.savefig('splines.png', transparent=True)
plt.show()

fig = plt.figure(figsize=(12,5))
ax = fig.add_subplot(1, 2, 1, projection='3d')
basis = chebyshev_basis(4)
plot_basis(basis, 'Chebyshev basis (K=%d)' % 4, ax)
basis = chebyshev_basis(7)
ax = fig.add_subplot(1, 2, 2, projection='3d')
plot_basis(basis, 'Chebyshev basis (K=%d)' % 5, ax)
plt.tight_layout()   
plt.savefig('cheb.png', transparent=True)
plt.show()
