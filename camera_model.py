import numpy as np
from numpy import linalg as LA
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from scipy.optimize import minimize

####################  define Lab's k   #############################

pixel_size = 0.007  # pixel size in mm. in meters: 7*10**-6
W_pixel = 2304
H_pixel = 1720
f = 65  # focal length in mm. in meters: 0.065
c_x = W_pixel/2
c_y = H_pixel/2
f_x = f/pixel_size   # option 2:  W_pixel/f
#c_x = 300
#c_y = 300
#f_x = 400
f_y = f_x
k = np.array([[f_x, 0  , c_x],
              [0  , f_y, c_y],
              [0  , 0  , 1  ]])
#########################################################

rotate_angle = 0# np.pi/16  # pi/16 is 11.25 deg
a = np.cos(rotate_angle)
b = np.sin(rotate_angle)

#################  Rotation Matrix   ####################
#        R_x               R_y               R_z
#    [[1, 0, 0],   ||  [[a, 0, b],   ||  [[a, -b, 0],
#     [0, a, -b],  ||   [0, 1, 0],   ||   [b, a, 0],
#     [0, b, a]]   ||   [-b, 0, a]]  ||   [0, 0, 1]]

###############   camera 1  ############
R1 = np.array([[1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]])
t1 = np.array([[0], [0], [0]])
###############   camera 2  ############
R2 = np.array([[a, -b, 0],
               [b, a, 0],
               [0, 0, 1]])
t2 = np.array([[0], [10], [0]])
#########################################################


def get_random_points(num, z = 500):

    point_3D = []

    theta_x = np.tan((W_pixel*pixel_size)/(2*f))
    theta_y = np.tan((H_pixel*pixel_size)/(2*f))

    point_3D.append(np.array([z*np.tan(theta_x),0,z]))
    point_3D.append(np.array([-z*np.tan(theta_x),0,z]))
    point_3D.append(np.array([0,z*np.tan(theta_y),z]))
    point_3D.append(np.array([0,-z*np.tan(theta_y),z]))

    bonds_x = z*np.tan(theta_x)
    bonds_y = z*np.tan(theta_y)
    for i in range(num):
        P = np.random.uniform(-bonds_x,bonds_x,3)
        P[1] = np.random.uniform(-bonds_y,bonds_y)
        P[2] = np.random.normal(3)+z
        point_3D.append(P)

    return point_3D

def TransToFund(points,hom = 0):
  ret = []
  for mat in points:
    x = (float(mat[0]))
    y = (float(mat[1]))
    if (hom):
        ret.append([x, y, 1])
    else: #2D
        ret.append([x,y])
  return ret

def pointsHomDeleteLastComponent(pts):
    ret = []
    for p in pts:
        #p_ret = np.array([int(p[0]),int(p[1])])
        ret.append(np.array([int(p[0]),int(p[1])]))
    array_ret = np.array(ret)
    return array_ret.transpose()

def plot_3D(xyz1, xyz2, color1, color2): #format: [ [x1,y1,z1],... [x8,y8,z8] ]

    fig = plt.figure()
    ax = Axes3D(fig)

    for i in range(len(xyz1)): #plot each point + it's index as text above
        ax.scatter(xyz1[i][0],xyz1[i][1],xyz1[i][2],color=color1)
        ax.text(xyz1[i][0],xyz1[i][1],xyz1[i][2],  '%s' % (str(i+1)), size=15, zorder=5, color='k')

        ax.scatter(xyz2[i][0], xyz2[i][1], xyz2[i][2], color=color2)
        ax.text(xyz2[i][0], xyz2[i][1], xyz2[i][2], '%s' % (str(i + 1)), size=15, zorder=5, color='k')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.gcf().set_size_inches(8, 6)
    plt.show()

def plot_2D_camera(pts1, pts2): #format:

    pts1_2cords = pointsHomDeleteLastComponent(pts1)
    pts2_2cords = pointsHomDeleteLastComponent(pts2)

    x1 = np.array(pts1_2cords[0])
    y1 = np.array(pts1_2cords[1])
    x2 = np.array(pts2_2cords[0])
    y2 = np.array(pts2_2cords[1])
    x1norm = x1 / y1
    c1norm = 1 / y1
    x2norm = x2 / y2
    c2norm = 1 / y2
    samp = np.linspace(0, 10, 5000)

    plt.subplot(121)
    for i in range(len(pts1)):
        plt.scatter(x1[i], y1[i], marker='+')
        plt.text(x1[i]-70, y1[i]+30,  '%s' % (str(i+1)), size=10, zorder=5, color='k')
        plt.text(x1[i]+30, y1[i]+30, '%d, %d' % (int(x1[i]), int(y1[i])))

    plt.subplot(122)
    for i in range(len(pts1)):
        plt.scatter(x2[i], y2[i], marker='o')
        plt.text(x2[i]-70, y2[i]+30,  '%s' % (str(i+1)), size=10, zorder=5, color='k')
        plt.text(x2[i]+30, y2[i]+30, '%d, %d' % (int(x2[i]), int(y2[i])))

    plt.gcf().set_size_inches(15, 5)
    plt.show()

    for i in range(len(pts1)):
        plt.scatter(x1[i], y1[i], marker='+', color='r')
        plt.text(x1[i], y1[i],  '%s' % (str(i+1)), size=10, zorder=5, color='r')
        plt.scatter(x2[i], y2[i], marker='o', color='b')
        plt.text(x2[i], y2[i],  '%s' % (str(i+1)), size=10, zorder=5, color='b')

    plt.gcf().set_size_inches(12, 5)
    plt.show()

point_3D_Euclidean = get_random_points(3)
point_3D_Homogeneous = cv2.convertPointsToHomogeneous(np.asarray(point_3D_Euclidean))

M1 = np.matmul(k, np.c_[R1, t1])
M2 = np.matmul(k, np.c_[R2, t2])
points1 = []
points2 = []

for i, P in enumerate(point_3D_Homogeneous):
    p1 = np.matmul(M1, np.transpose(np.mat(P)))
    p1 = p1 / p1[2]
    points1.append(p1)
    p2 = np.matmul(M2, np.transpose(np.mat(P)))
    p2 = p2 / p2[2]
    points2.append(p2)


plot_2D_camera(points1, points2)

print('end test')
