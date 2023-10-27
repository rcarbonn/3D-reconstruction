import numpy as np
from utils import epipolar_lines, visualize_correspondences, inlier_plot
import scipy
import matplotlib.pyplot as plt

np.random.seed(0)
np.set_printoptions(precision=3, suppress=True)

def eight_point(pts1, pts2, K1=None, K2=None, ransac_filter=False, image1=None, image2=None, viz_corr=False, viz_epi=True):
    assert pts1.shape == pts2.shape, "Corrspondences must have the same shape, (N,2)"
    assert pts1.shape[0] >= 8, "Need at least 8 correspondences to compute fundamental matrix"

    if ransac_filter:
        print("Running with RANSAC...")
        max_iter = 10000
        max_inliers = 0
        num_inliers = []
        for i in range(max_iter):
            choice = np.random.choice(pts1.shape[0], 8, replace=False)
            pts1_8 = pts1[choice,:]
            pts2_8 = pts2[choice,:]
            hpts1 = np.hstack((pts1, np.ones((pts1.shape[0], 1))))
            hpts2 = np.hstack((pts2, np.ones((pts2.shape[0], 1))))
            F = eight_point(pts1_8, pts2_8, ransac_filter=False, viz_corr=False, viz_epi=False)
            _,res = sampson_distance(F, hpts1, hpts2, residuals=True)
            inlier_mask = res < 0.5
            n_inliers = np.sum(inlier_mask)
            if n_inliers > max_inliers:
                max_inliers = n_inliers
                inlier_mask_best = inlier_mask
                Fbest = F
            num_inliers.append(max_inliers*100/pts1.shape[0])
        print("Maximum number of inliers: ", max_inliers)
        inliers1 = pts1[inlier_mask_best,:]
        inliers2 = pts2[inlier_mask_best,:]
        Fransac = eight_point(inliers1, inliers2, ransac_filter=False, image1=image1, image2=image2, viz_corr=False, viz_epi=True)
        inlier_plot(num_inliers)
        return Fransac

    pts1_, T1 = normalize_points(pts1) # normalized homogeneous coordinates
    pts2_, T2 = normalize_points(pts2) # normalized homogeneous coordinates

    A = computeA(pts1_, pts2_)
    _,D,vh = np.linalg.svd(A)
    f = vh[-1].reshape(3,3)
    F = singularizeF(f)

    if not ransac_filter and pts1_.shape[0] > 8:
        # optional refinement
        # Fr = refineF_algebraic(F,A, method='lm')
        F = refineF_geometric(F, pts1_, pts2_, dist='sampson')
        print("F matrix:")
        print(F/F[-1,-1])
        print("Rank of F matrix: ", np.linalg.matrix_rank(F))

    F = T2.T@F@T1
    F = F/F[-1,-1]

    if K1 is not None and K2 is not None:
        E = K2.T@F@K1
        E = E/E[-1,-1]
        print("E matrix:")
        print(E)
        print("Rank of E matrix: ", np.linalg.matrix_rank(E))

    if viz_corr:
        visualize_correspondences(image1, image2, pts1, pts2)
    if viz_epi:
        epipolar_lines(image1, image2, F)

    return F

def seven_point(pts1, pts2, pts1_extra=None, pts2_extra=None, K1=None, K2=None, ransac_filter=False, image1=None, image2=None, viz_corr=False, viz_epi=True):
    assert pts1.shape == pts2.shape, "Corrspondences must have the same shape, (N,2)"
    assert pts1.shape[0] == 7, "Need exactly 7 correspondences for the 7-point method"

    # if ransac_filter:
    #     print("Running with RANSAC...")
    #     max_iter = 10000
    #     max_inliers = 0
    #     num_inliers = []
    #     best_choice = None
    #     for i in range(max_iter):
    #         choice = np.random.choice(pts1_extra.shape[0], 7, replace=False)
    #         pts1_7 = pts1_extra[choice,:]
    #         pts2_7 = pts2_extra[choice,:]
    #         hpts1 = np.hstack((pts1_extra, np.ones((pts1_extra.shape[0], 1))))
    #         hpts2 = np.hstack((pts2_extra, np.ones((pts2_extra.shape[0], 1))))
    #         Farr = seven_point(pts1_7, pts2_7, ransac_filter=False, viz_corr=False, viz_epi=False)
    #         for f in Farr:
    #             _,res = sampson_distance(f, hpts1, hpts2, residuals=True)
    #             inlier_mask = res < 0.5
    #             n_inliers = np.sum(inlier_mask)
    #             if n_inliers > max_inliers:
    #                 max_inliers = n_inliers
    #                 inlier_mask_best = inlier_mask
    #                 best_choice = choice
    #                 Fbest = f
    #         num_inliers.append(max_inliers*100/pts1_extra.shape[0])
    #     print("Maximum number of inliers: ", max_inliers)
    #     inliers1 = pts1_extra[inlier_mask_best,:]
    #     inliers2 = pts2_extra[inlier_mask_best,:]
    #     Fransac = seven_point(pts1_extra[best_choice,:], pts2_extra[best_choice,:], inliers1, inliers2, ransac_filter=False, image1=image1, image2=image2, viz_corr=False, viz_epi=True)
    #     inlier_plot(num_inliers)
    #     return Fransac

    pts1_, T1 = normalize_points(pts1) # normalized homogeneous coordinates
    pts2_, T2 = normalize_points(pts2) # normalized homogeneous coordinates

    A = computeA(pts1_, pts2_)
    _,_,Vh = np.linalg.svd(A)
    F1 = Vh[-1].reshape(3,3)
    F2 = Vh[-2].reshape(3,3)

    df1 = np.linalg.det(F1)
    df2 = np.linalg.det(F2)
    # general 3d polynomial = p(x) = Ax^3 + Bx^2 + Cx + D
    D = df2
    B = df1/2 + np.linalg.det(2*F2-F1)/2 - df2
    A = (np.linalg.det(2*F1-F2) - 2*B - 2*df1 + df2)/6
    C = df1 - df2 - A - B
    roots = np.polynomial.polynomial.polyroots([D,C,B,A])
    print("Roots for lamdba: ", roots)
    Fsols = []
    for rt in roots:
        if not np.iscomplex(rt):
            F = np.real(rt)*F1 + (1-np.real(rt))*F2
            F = T2.T@F@T1
            F = F/F[-1,-1]
            Fsols.append(F)

    if pts1_extra is not None and pts2_extra is not None and not ransac_filter:
        pts1_extra_,_ = normalize_points(pts1_extra) # normalized homogeneous coordinates
        pts2_extra_,_ = normalize_points(pts2_extra) # normalized homogeneous coordinates
        Afull = computeA(pts1_extra_, pts2_extra_)
        err = []
        for f in Fsols:
            # f = T2.T@f@T1
            # f = f/f[-1,-1]
            err.append(np.sum((Afull@f.reshape(-1,1))**2))
        print(err)
        Fs = Fsols[np.argmin(err)]
        Fsols = [refineF_geometric(Fs, pts1_extra_, pts2_extra_, dist='sampson')]
        print("F matrix:")
        print(Fsols[0])
        print("Rank of F matrix: ", np.linalg.matrix_rank(Fsols[0]))

    Farr = []
    for f in Fsols:
        F7 = T2.T@f@T1
        F7 = F7/F7[-1,-1]
        Farr.append(F7)

    if K1 is not None and K2 is not None:
        E = K2.T@Fsols[0]@K1
        E = E/E[-1,-1]
        print("E matrix:")
        print(E)
        print("Rank of E matrix: ", np.linalg.matrix_rank(E))

    # F = eight_point(pts1_extra, pts2_extra, image1=image1, image2=image2, viz_corr=True)
    if viz_corr:
        visualize_correspondences(image1, image2, pts1, pts2)
    if viz_epi:
        epipolar_lines(image1, image2, Farr[0])

    return Farr

def ransac():
    pass

def ransac_plots():
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

def computeA(pts1, pts2):
    npts = pts1.shape[0]
    A = np.zeros((npts, 9))
    for i in range(npts):
        A[i,0:3] = pts2[i,0]*pts1[i,:]
        A[i,3:6] = pts2[i,1]*pts1[i,:]
        A[i,6:9] = pts1[i,:]
    return A

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

def sampson_distance(F, hpts1, hpts2, residuals=False):
    Fp1 = F@hpts1.T
    FTp2 = F.T@hpts2.T
    r = 0
    res = []
    for fp1, fp2, hp2 in zip(Fp1.T, FTp2.T, hpts2):
        d = (hp2.dot(fp1))**2 * (1/((fp1[0]**2 + fp1[1]**2) + (fp2[0]**2 + fp2[1]**2)))
        r+=d
        res.append(d)
    if residuals:
        return r, np.array(res)
    else:
        return r

def sampson_error(f, hpts1, hpts2):
    F = singularizeF(f.reshape([3, 3]))
    return sampson_distance(F, hpts1, hpts2)

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
