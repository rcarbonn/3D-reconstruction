import numpy as np
from utils import epipolar_lines, visualize_correspondences
import scipy

# np.set_printoptions(precision=3, suppress=True)

def eight_point(pts1, pts2, K1=None, K2=None, ransac_filter=False, image1=None, image2=None):
    assert pts1.shape == pts2.shape, "Corrspondences must have the same shape, (N,2)"
    assert pts1.shape[0] >= 8, "Need at least 8 correspondences to compute fundamental matrix"

    pts1_, T1 = normalize_points(pts1) # normalized homogeneous coordinates
    pts2_, T2 = normalize_points(pts2) # normalized homogeneous coordinates

    npts = pts1.shape[0]
    # npts = 100
    A = np.zeros((npts, 9))

    for i in range(npts):
        A[i,0:3] = pts2_[i,0]*pts1_[i,:]
        A[i,3:6] = pts2_[i,1]*pts1_[i,:]
        A[i,6:9] = pts1_[i,:]
    print("Rank of A matrix: ", np.linalg.matrix_rank(A))
    _,D,vh = np.linalg.svd(A)
    print("original D:", D[-5:]**2)
    f = vh[-1].reshape(3,3)
    print("Rank of f: ", np.linalg.matrix_rank(f))
    print("error_before:::::", np.sum((A@f.reshape(-1,1))**2))
    F = singularizeF(f)
    # F = f.reshape(3,3)
    print("error_after:::::", np.sum((A@F.reshape(-1,1))**2))
    print("Rank of F matrix: ", np.linalg.matrix_rank(F))
    # e,_  = epipoles(F)
    print(F/F[-1,-1])
    Fr = refineF(F,A)
    print(Fr)
    Fr2 = refineF2(F, pts1_[:,:2], pts2_[:,:2])
    print("Fr2 is : ")
    print(Fr2/Fr2[-1,-1])
    err2 = A @ Fr2.reshape(-1,1)
    print("error of fr2 is: ", np.sum(err2**2))
    # F = T2.T@F@T1
    Fr = T2.T@Fr@T1
    # # F = F/F[-1,-1]
    # print("F matrix:")
    # print(F)
    print(Fr)
    print("Rank of F matrix: ", np.linalg.matrix_rank(Fr))
    #  #TODO: iteratively minimize algebraic error to refine F

    # # visualize_correspondences(image1, image2, pts1, pts2)
    # epipolar_lines(image1, image2, Fr)

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
    # assert S[-1] < 1e-9, "F must be singular"
    e1 = V[-1,:]
    U,S,V = np.linalg.svd(F.T)
    e2 = V[:,-1]
    return e1, e2

def algebraic_error(e, A=None):
    # e,_ = epipoles(f.reshape(3,3))
    ex = np.cross(e, np.identity(3)*-1)
    E = np.zeros((9,9))
    E[0:3,0:3] = ex
    E[3:6,3:6] = ex
    E[6:9,6:9] = ex
    U,D,Vh = np.linalg.svd(E)
    print("rank of E is ", np.linalg.matrix_rank(E))
    # print(D)
    U_ = U[:,:6] # rank of E is 6
    A_ = A@U_
    _,_,Vh = np.linalg.svd(A_)
    x = Vh[-1].reshape(-1,1)
    f = U_@x
    e_new,_ = epipoles(f.reshape(3,3))
    # if np.dot(e_new, e) < 0:
    #     print('swithcing sign')
    #     f = -f
    err = A @ f
    r = np.sum(err.flatten()**2)/2
    # r = np.linalg.norm(err)/A.shape[0]
    print(r)
    # return (r,)
    return err.flatten()


def refineF(F, A):
    # e,_ = epipoles(F)
    e = np.array([1,1,1])
    print(e, F@e.reshape(-1,1))
    Aold = A.copy()
    res = scipy.optimize.least_squares(
        lambda x: algebraic_error(x, A), e.flatten(),
        method='lm',
        verbose=1,
        loss='linear',
        xtol=1e-12,
        # x_scale=np.array([1e9,1e9,1e9])
        # maxiter=100000,
        # maxfun=10000
    )
    print(res.x)
    ex = np.cross(res.x, np.identity(3)*-1)
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


def _objective_F(f, pts1, pts2):
    F = singularizeF(f.reshape([3, 3]))
    num_points = pts1.shape[0]
    hpts1 = np.concatenate([pts1, np.ones([num_points, 1])], axis=1)
    hpts2 = np.concatenate([pts2, np.ones([num_points, 1])], axis=1)
    Fp1 = F.dot(hpts1.T)
    FTp2 = F.T.dot(hpts2.T)

    r = 0
    for fp1, fp2, hp2 in zip(Fp1.T, FTp2.T, hpts2):
        r += (hp2.dot(fp1))**2 * (1/(fp1[0]**2 + fp1[1]**2) + 1/(fp2[0]**2 + fp2[1]**2))
    return r

def refineF2(F, pts1, pts2):
    f = scipy.optimize.fmin_powell(
        lambda x: _objective_F(x, pts1, pts2), F.reshape([-1]),
        maxiter=100000,
        maxfun=10000
    )
    return singularizeF(f.reshape([3, 3]))

def ransac():
    pass

def ransac_plots():
    pass