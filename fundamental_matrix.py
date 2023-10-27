import numpy as np
from utils import epipolar_lines, visualize_correspondences
import scipy

np.set_printoptions(precision=3, suppress=True)

def eight_point(pts1, pts2, K1=None, K2=None, ransac_filter=False, image1=None, image2=None):
    assert pts1.shape == pts2.shape, "Corrspondences must have the same shape, (N,2)"
    assert pts1.shape[0] >= 8, "Need at least 8 correspondences to compute fundamental matrix"

    pts1_, T1 = normalize_points(pts1) # normalized homogeneous coordinates
    pts2_, T2 = normalize_points(pts2) # normalized homogeneous coordinates

    npts = pts1.shape[0]
    A = np.zeros((npts, 9))

    for i in range(npts):
        A[i,0:3] = pts2_[i,0]*pts1_[i,:]
        A[i,3:6] = pts2_[i,1]*pts1_[i,:]
        A[i,6:9] = pts1_[i,:]
    _,D,vh = np.linalg.svd(A)
    f = vh[-1].reshape(3,3)
    F = singularizeF(f)

    # optional refinement
    # Fr = refineF_algebraic(F,A, method='lm')
    Fr = refineF_geometric(F, pts1_, pts2_, dist='symmetric')

    Fr = T2.T@Fr@T1
    Fr = Fr/Fr[-1,-1]
    print("F matrix:")
    print(Fr)
    print("Rank of F matrix: ", np.linalg.matrix_rank(Fr))

    # visualize_correspondences(image1, image2, pts1, pts2)
    epipolar_lines(image1, image2, Fr)

    return F

def seven_point(pts1, pts2, K1=None, K2=None, ransac_filter=False, image1=None, image2=None):
    print(pts1.shape)
    print(pts2.shape)
    pass

def normalize_points(pts):
    ptsh = np.hstack((pts, np.ones((pts.shape[0], 1))))
    m = np.mean(pts, axis=0)
    pts_ = pts - m
    s = np.sqrt(2) / np.max(np.linalg.norm(pts_, axis=1))
    T = np.array([[s, 0, -s*m[0]],
                  [0, s, -s*m[1]],
                  [0, 0, 1]])
    return (T@ptsh.T).T, T

def singularizeF(F):
    U,S,V = np.linalg.svd(F)
    S[-1] = 0
    return U@np.diag(S)@V

def epipoles(F):
    U,S,V = np.linalg.svd(F)
    assert S[-1] < 1e-9, "F must be singular"
    e1 = V[-1,:]
    U,S,V = np.linalg.svd(F.T)
    e2 = V[:,-1]
    return e1/e1[-1], e2

def sampson_error(f, hpts1, hpts2):
    F = singularizeF(f.reshape([3, 3]))
    Fp1 = F@hpts1.T
    FTp2 = F.T@hpts2.T
    r = 0
    for fp1, fp2, hp2 in zip(Fp1.T, FTp2.T, hpts2):
        r += (hp2.dot(fp1))**2 * (1/((fp1[0]**2 + fp1[1]**2) + (fp2[0]**2 + fp2[1]**2)))
    return r

def symmetric_epipolar_distance(f, hpts1, hpts2):
    F = singularizeF(f.reshape([3, 3]))
    Fp1 = F@hpts1.T
    FTp2 = F.T@hpts2.T
    r = 0
    for fp1, fp2, hp2 in zip(Fp1.T, FTp2.T, hpts2):
        r += (hp2.dot(fp1))**2 * (1/(fp1[0]**2 + fp1[1]**2) + 1/(fp2[0]**2 + fp2[1]**2))
    return r

def algebraic_error(e, A=None, method='lm'):
    ex = np.cross(e, np.identity(3)*-1)
    E = np.zeros((9,9))
    E[0:3,0:3] = ex
    E[3:6,3:6] = ex
    E[6:9,6:9] = ex
    U,D,Vh = np.linalg.svd(E)
    U_ = U[:,:6] # rank of E is 6
    A_ = A@U_
    _,_,Vh = np.linalg.svd(A_)
    x = Vh[-1].reshape(-1,1)
    f = U_@x
    # e_new,_ = epipoles(f.reshape(3,3))
    # need old e value from previous iteration to maintain smoothness and differentiability,
    # not possible with scipy least_squares, might be possible with scipy.minimize
    # if np.dot(e_old, e) < 0:
    #     f = -f
    err = A @ f
    r = np.sum(err.flatten()**2)/2

    if method=='lm':
        return err.flatten()
    else:
        return r


def refineF_algebraic(F, A, method='powell'):
    print(f"Running algebraic refinement...")
    e,_ = epipoles(F)
    if method=='lm':
        res = scipy.optimize.least_squares(
            lambda x: algebraic_error(x, A, method=method), e.flatten(),
            method='lm',
            verbose=1,
            loss='linear',
            xtol=1e-12
        )
        e_opt = res.x

    elif method=='powell':
        e_opt = scipy.optimize.fmin_powell(
            lambda x: algebraic_error(x, A, method=method), e.flatten(),
            xtol=1e-12,
        )
    # get F from optmized e_opt
    ex = np.cross(e_opt, np.identity(3)*-1)
    E = np.zeros((9,9))
    E[0:3,0:3] = ex
    E[3:6,3:6] = ex
    E[6:9,6:9] = ex
    U,D,Vh = np.linalg.svd(E)
    U_ = U[:,:6] # rank of E is 6
    A_ = A@U_
    _,_,Vh = np.linalg.svd(A_)
    x = Vh[-1].reshape(-1,1)
    f = U_@x
    Fopt = f.reshape(3,3)
    return Fopt/Fopt[-1,-1]


def refineF_geometric(F, pts1, pts2, dist='sampson'):
    print(f"Running geometric refinement with {dist} distance...")
    if dist=='sampson':
        f = scipy.optimize.fmin_powell(
            # lambda x: _objective_F(x, pts1, pts2), F.reshape([-1]),
            lambda x: sampson_error(x, pts1, pts2), F.reshape([-1]),
            maxiter=100000,
            maxfun=10000
        )
    elif dist=='symmetric':
        f = scipy.optimize.fmin_powell(
            lambda x: symmetric_epipolar_distance(x, pts1, pts2), F.reshape([-1]),
            maxiter=100000,
            maxfun=10000
        )
    return singularizeF(f.reshape([3, 3]))

def ransac():
    pass

def ransac_plots():
    pass