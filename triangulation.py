import numpy as np
from utils import plot3d
import scipy.optimize

def triangulate(pts1, pts2, P1, P2, image1=None, image2=None):
    assert pts1.shape[0] == pts2.shape[0], "pts1 and pts2 should have same number of points"
    pts1_ = np.hstack((pts1, np.ones((pts1.shape[0], 1))))
    pts2_ = np.hstack((pts2, np.ones((pts2.shape[0], 1))))

    A = np.array([np.vstack([(np.cross(p1, -np.identity(3))@P1)[:2], (np.cross(p2, -np.identity(3))@P2)[:2]]) for p1,p2 in zip(pts1_, pts2_)])
    X = np.array([np.linalg.svd(a)[2][-1] for a in A])
    X = X/X[:,None,-1]

    Xopt = bundle_adjustment(X[:,:3], P1, P2, pts1, pts2)[24:]
    X = Xopt.reshape((-1,3))

    img1 = np.array(image1)
    colors1 = img1[pts1[:,1], pts1[:,0],:]
    plot3d(X, colors1)
    return

def reprojection_error(x, pts1, pts2):
    P1 = x[:12].reshape((3,4))
    P2 = x[12:24].reshape((3,4))
    X = x[24:].reshape((-1,3))
    X = X.reshape((-1,3))
    X = np.hstack((X, np.ones((X.shape[0],1))))
    reproj1 = P1 @ X.T
    reproj1 = (reproj1 / reproj1[2,:])[:2,:]
    reproj2 = P2 @ X.T
    reproj2 = (reproj2 / reproj2[2,:])[:2,:]
    err = np.sum([np.square(reproj1.T - pts1), np.square(reproj2.T - pts2)])
    return err

def bundle_adjustment(X, P1, P2, pts1, pts2):
    x0 = np.hstack((P1.flatten(), P2.flatten(), X.flatten()))
    print("Reprojection error before bundle adjustment: ", reprojection_error(x0, pts1, pts2))
    f = scipy.optimize.fmin_powell(
        lambda x: reprojection_error(x, pts1, pts2), x0,
        maxiter=10000, disp=True
    )
    # f = scipy.optimize.fmin_powell(
    #     lambda x: reprojection_error(x, P1, P2, pts1, pts2), X.flatten(),
    #     maxiter=10000, disp=True
    # )
    return f

def colmap_reconstruct():
    pass