import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D as l2d
import cv2
import os


DATA_KEYS = {
    'q1a': ['pts1', 'pts2', 'ransac_filter', 'K1', 'K2', 'image1', 'image2'],
    'q1b': ['pts1', 'pts2', 'pts1_extra', 'pts2_extra', 'ransac_filter', 'K1', 'K2', 'image1', 'image2'],
    'q2a': ['pts1', 'pts2', 'ransac_filter', 'K1', 'K2', 'image1', 'image2'],
    'q2b': ['pts1', 'pts2', 'pts1_extra', 'pts2_extra', 'ransac_filter', 'K1', 'K2', 'image1', 'image2'],
    'q3' : ['pts1', 'pts2', 'P1', 'P2', 'image1', 'image2'],
    'q5' : ['pts1', 'pts2', 'ransac_filter', 'image1', 'image2']
}


def q1_data(question, object_name, data_dir='./data/', seven_pt=False, ransac_filter=False, debug=False):
    data = {}
    obj_path = os.path.join(data_dir, question, object_name)
    data['corr_raw'] = np.load(os.path.join(obj_path, f"{object_name}_corresp_raw.npz"))
    data['intrinsic_matrices'] = np.load(os.path.join(obj_path, f"intrinsic_matrices_{object_name}.npz"))
    data['pts1'] = data['corr_raw']['pts1']
    data['pts2'] = data['corr_raw']['pts2']
    data['K1'] = data['intrinsic_matrices']['K1']
    data['K2'] = data['intrinsic_matrices']['K2']
    data['ransac_filter'] = ransac_filter
    data['image1'] = Image.open(os.path.join(obj_path, f"image_1.jpg"))
    data['image2'] = Image.open(os.path.join(obj_path, f"image_2.jpg"))
    if seven_pt or 'b' in question:
        data['corr7'] = np.load(os.path.join(obj_path, f"{object_name}_7_point_corresp.npz"))
        data['pts1'] = data['corr7']['pts1']
        data['pts2'] = data['corr7']['pts2']
        data['pts1_extra'] = data['corr_raw']['pts1']
        data['pts2_extra'] = data['corr_raw']['pts2']
    qdata = {k:v for k,v in data.items() if k in DATA_KEYS[question]}
    if debug:
        for k,v in qdata.items():
            if isinstance(v, np.ndarray):
                print(f"{k}: {v.shape}")
    return qdata


def q3_data(data_dir='./data/', debug=False):
    data = {}
    data['pts1'] = np.load(os.path.join(data_dir, 'q3', 'pts1.npy'))
    data['pts2'] = np.load(os.path.join(data_dir, 'q3', 'pts2.npy'))
    data['P1'] = np.load(os.path.join(data_dir, 'q3', 'P1.npy'))
    data['P2'] = np.load(os.path.join(data_dir, 'q3', 'P2.npy'))
    data['image1'] = Image.open(os.path.join(data_dir, 'q3', 'img1.jpg'))
    data['image2'] = Image.open(os.path.join(data_dir, 'q3', 'img2.jpg'))
    qdata = {k:v for k,v in data.items() if k in DATA_KEYS['q3']}
    if debug:
        for k,v in qdata.items():
            if isinstance(v, np.ndarray):
                print(f"{k}: {v.shape}")
    return qdata


def q5_data(data_dir='./data/', ransac_filter = True, debug=False):
    data = {}
    data['image1'] = Image.open(os.path.join(data_dir, 'q5', 'image1.jpg'))
    data['image2'] = Image.open(os.path.join(data_dir, 'q5', 'image2.jpg'))
    data['ransac_filter'] = ransac_filter
    img1g = cv2.cvtColor(np.array(data['image1']), cv2.COLOR_RGB2GRAY)
    img2g = cv2.cvtColor(np.array(data['image2']), cv2.COLOR_RGB2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()
    kp1, desc1 = sift.detectAndCompute(img1g, None)
    kp2, desc2 = sift.detectAndCompute(img2g, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc1, desc2, k=2)

    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append(m)
    
    if debug:
        img3 = cv2.drawMatches(img1g, kp1, img2g, kp2, good, None, flags=2)
        plt.imshow(img3)
        plt.show()
    
    pts1 = []
    pts2 = []
    for mat in good:
        pts1.append(kp1[mat.queryIdx].pt)
        pts2.append(kp2[mat.trainIdx].pt)
    data['pts1'] = np.array(pts1)
    data['pts2'] = np.array(pts2)
    qdata = {k:data[k] for k in DATA_KEYS['q5']}
    if debug:
        for k,v in qdata.items():
            if isinstance(v, np.ndarray):
                print(f"{k}: {v.shape}")
    return qdata



def load_data(args):
    if 'q1' in args.question:
        data = q1_data(args.question, args.image, ransac_filter=args.ransac, debug=args.debug)
    if 'q2' in args.question:
        data = q1_data('q1'+args.question[-1], args.image, ransac_filter=True, debug=args.debug)
    if 'q3' in args.question:
        data = q3_data(debug=args.debug)
    if 'q5' in args.question:
        data = q5_data(debug=args.debug)
    return  data

def visualize_correspondences(image1, image2, pts1, pts2):
    img1 = np.array(image1)
    img2 = np.array(image2)

    h1,w1 = img1.shape[:2]
    h2,w2 = img2.shape[:2]

    cimage = np.zeros((max(h1,h2), w1+w2, 3), dtype=np.uint8)
    cimage[:h1, :w1, :] = img1
    cimage[:h2, w1:, :] = img2
    fig, ax = plt.subplots(1,1,figsize=(10,10))
    ax.imshow(cimage)
    for p1, p2 in zip(pts1,pts2):
        l = l2d([p1[0], p2[0]+w1], [p1[1], p2[1]], linewidth=1.0)
        ax.add_line(l)
    plt.show()

def epipolar_lines(image1, image2, F):
    img1 = np.array(image1)
    img2 = np.array(image2)
    h,w = img2.shape[:2]

    def onclick(event):
        try:
            u,v = event.xdata, event.ydata
            l = F.dot(np.array([u,v,1]))
            l = l/l[-1]
            # l = l/np.linalg.norm(l[:2])
            if np.linalg.norm(l[:2]) < 1e-5:
                print("Zero line! skipping")
                return
            if l[1] !=0:
                x1,y1 = 0, -l[2]/l[1]
                x2,y2 = w, -(l[0]*(w)+l[2])/l[1]
            else:
                x1,y1 = -l[2]/l[0], 0
                x2,y2 = -(l[1]*(h)+l[2])/l[0], h
            
            # y1,y2 = np.clip([y1,y2],0,h)
            # x1,x2 = np.clip([x1,x2],0,w)
            ax2.set_xlim(0,w)
            ax2.set_ylim(h,0)
            ax1.plot(u,v,'o', markersize=6, linewidth=2)
            ax2.plot([x1,x2], [y1,y2], linewidth=2)
            plt.draw()
        except:
            print("Invalid point! skipping")

    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(img1)
    ax1.set_axis_off()
    ax1.set_title("Selected Points")
    ax2.imshow(img2)
    ax2.set_title("Epipolar Lines")
    ax2.set_axis_off()
    _ = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    return

def inlier_plot(num_inliers):
    plt.figure()
    plt.plot(num_inliers)
    plt.show()
    return

def plot3d(X, colors):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:,0], X[:,1], X[:,2], c=colors/255.0, marker='o', s=5)
    plt.show()
    return