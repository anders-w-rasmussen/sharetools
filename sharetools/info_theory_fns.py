import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import cKDTree
from scipy.special import gamma, digamma



def entropy_decomp_3(x, y, z, n_neighbors):

    n_samples = x.size

    # First Calculate H(x), H(y), H(z) i.e. entropy of x, y, z individually
    h_x = get_h(x, n_neighbors)
    h_y = get_h(y, n_neighbors)
    h_z = get_h(z, n_neighbors)

    # Now we get the mutual information values I(x,y), I(x,z), I(y,z)
    i_xy = _compute_mi_cc(x, y, n_neighbors)
    i_xz = _compute_mi_cc(x, z, n_neighbors)
    i_yz = _compute_mi_cc(y, z, n_neighbors)

    # Last thing we need is the three way mutual information
    i_xyz = mi_3(x, y, z, n_neighbors)

    # Using chain rule for mutual information (confirm this)
    i_xyGz = i_xy - i_xyz
    i_xzGy = i_xz - i_xyz
    i_yzGx = i_yz - i_xyz

    print("it worked")

def get_h(x, k=1, norm='max', min_dist=0.):
    """

    # TAKEN FROM https://github.com/paulbrodersen/entropy_estimators.

    Estimates the entropy H of a random variable x (in nats) based on
    the kth-nearest neighbour distances between point samples.
    @reference:
    Kozachenko, L., & Leonenko, N. (1987). Sample estimate of the entropy of a random vector. Problemy Peredachi Informatsii, 23(2), 9–16.
    Arguments:
    ----------
    x: (n, d) ndarray
        n samples from a d-dimensional multivariate distribution
    k: int (default 1)
        kth nearest neighbour to use in density estimate;
        imposes smoothness on the underlying probability distribution
    norm: 'euclidean' or 'max'
        p-norm used when computing k-nearest neighbour distances
    min_dist: float (default 0.)
        minimum distance between data points;
        smaller distances will be capped using this value
    Returns:
    --------
    h: float
        entropy H(X)
    """

    n, d = x.shape

    if norm == 'max': # max norm:
        p = np.inf
        log_c_d = 0 # volume of the d-dimensional unit ball
    elif norm == 'euclidean': # euclidean norm
        p = 2
        log_c_d = (d/2.) * np.log(np.pi) - np.log(gamma(d/2. +1))
    else:
        raise NotImplementedError("Variable 'norm' either 'max' or 'euclidean'")

    kdtree = cKDTree(x)

    # query all points -- k+1 as query point also in initial set
    # distances, _ = kdtree.query(x, k + 1, eps=0, p=norm)
    distances, _ = kdtree.query(x, k + 1, eps=0, p=p)
    distances = distances[:, -1]

    # enforce non-zero distances
    distances[distances < min_dist] = min_dist

    sum_log_dist = np.sum(log(2*distances)) # where did the 2 come from? radius -> diameter
    h = -digamma(k) + digamma(n) + log_c_d + (d / float(n)) * sum_log_dist

    return h


def _compute_mi_cc(x, y, n_neighbors):


    """Compute mutual information between two continuous variables.

    I lifted this from SKLEARN

    Parameters
    ----------
    x, y : ndarray, shape (n_samples,)
        Samples of two continuous random variables, must have an identical
        shape.

    n_neighbors : int
        Number of nearest neighbors to search for each point, see [1]_.

    Returns
    -------
    mi : float
        Estimated mutual information. If it turned out to be negative it is
        replace by 0.

    Notes
    -----
    True mutual information can't be negative. If its estimate by a numerical
    method is negative, it means (providing the method is adequate) that the
    mutual information is close to 0 and replacing it by 0 is a reasonable
    strategy. """


    n_samples = x.size

    x = x.reshape((-1, 1))
    y = y.reshape((-1, 1))
    xy = np.hstack((x, y))

    # Here we rely on NearestNeighbors to select the fastest algorithm.
    nn = NearestNeighbors(metric='chebyshev', n_neighbors=n_neighbors)

    nn.fit(xy)
    radius = nn.kneighbors()[0]
    radius = np.nextafter(radius[:, -1], 0)

    # Algorithm is selected explicitly to allow passing an array as radius
    # later (not all algorithms support this).
    nn.set_params(algorithm='kd_tree')

    nn.fit(x)
    ind = nn.radius_neighbors(radius=radius, return_distance=False)
    nx = np.array([i.size for i in ind])

    nn.fit(y)
    ind = nn.radius_neighbors(radius=radius, return_distance=False)
    ny = np.array([i.size for i in ind])

    mi = (digamma(n_samples) + digamma(n_neighbors) -
          np.mean(digamma(nx + 1)) - np.mean(digamma(ny + 1)))

    return max(0, mi)


def mi_3(x, y, z, n_neighbors):
    """Compute mutual information between three continuous variables.

    I lifted this from SKLEARN """

    n_samples = x.size

    x = x.reshape((-1, 1))
    y = y.reshape((-1, 1))
    z = y.reshape((-1, 1))
    xyz = np.hstack((x, y, z))

    # Here we rely on NearestNeighbors to select the fastest algorithm.
    nn = NearestNeighbors(metric='chebyshev', n_neighbors=n_neighbors)

    nn.fit(xyz)
    radius = nn.kneighbors()[0]
    radius = np.nextafter(radius[:, -1], 0)

    # Algorithm is selected explicitly to allow passing an array as radius
    # later (not all algorithms support this).
    nn.set_params(algorithm='kd_tree')

    nn.fit(x)
    ind = nn.radius_neighbors(radius=radius, return_distance=False)
    nx = np.array([i.size for i in ind])

    nn.fit(y)
    ind = nn.radius_neighbors(radius=radius, return_distance=False)
    ny = np.array([i.size for i in ind])

    nn.fit(z)
    ind = nn.radius_neighbors(radius=radius, return_distance=False)
    nz = np.array([i.size for i in ind])


    mi = (digamma(n_samples) + digamma(n_neighbors) -
          np.mean(digamma(nx + 1)) - np.mean(digamma(ny + 1)) - np.mean(digamma(nz + 1)))

    return max(0, mi)

