import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D as l2d
import os


DATA_KEYS = {
    'q1a': ['pts1', 'pts2', 'ransac_filter', 'K1', 'K2', 'image1', 'image2'],
    'q1b': ['pts1', 'pts2', 'ransac_filter', 'K1', 'K2', 'image1', 'image2'],
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
    if seven_pt or 'q1b' in question:
        data['corr7'] = np.load(os.path.join(obj_path, f"{object_name}_7_point_corresp.npz"))
        data['pts1'] = data['corr7']['pts1']
        data['pts2'] = data['corr7']['pts2']
    qdata = {k:v for k,v in data.items() if k in DATA_KEYS[question]}
    if debug:
        for k,v in qdata.items():
            if isinstance(v, np.ndarray):
                print(f"{k}: {v.shape}")
    return qdata

def load_data(args):
    if 'q1' in args.question:
        data = q1_data(args.question, args.image, ransac_filter=args.ransac, debug=args.debug)
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
        u,v = event.xdata, event.ydata
        # l = F@np.array([u,v,1]).reshape(3,1)
        l = F.dot(np.array([u,v,1]))
        l = l/l[-1]
        # l = l/np.linalg.norm(l[:2])
        print(l)
        if np.linalg.norm(l[:2]) < 1e-5:
            print("Zero line! skipping")
            return
        if l[1] !=0:
            x1,y1 = 0, -l[2]/l[1]
            x2,y2 = w-1, -(l[0]*(w-1)+l[2])/l[1]
        else:
            x1,y1 = -l[2]/l[0], 0
            x2,y2 = -(l[1]*(h-1)+l[2])/l[0], h-1
        
        ax1.plot(u,v,'*', markersize=6, linewidth=2)
        ax1.set_axis_off()
        ax2.plot([x1,x2], [y1,y2], linewidth=2)
        ax2.set_axis_off()
        plt.draw()

    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(img1)
    ax2.imshow(img2)
    _ = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    return