################################################################################
################################################################################
##                   Bundle Adjustment With Known Positions                   ##
##                                                                            ##
##             FONCTIONS DE BASE UTILISEES DANS TOUS NOS SCRIPTS              ##
################################################################################
################################################################################

## packages

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from numpy.random import rand, normal
import scipy.linalg as sl
from scipy.linalg import rq
from math import atan2

## fonctions

def det(C1, C2, theta1, theta2, p1, p2, f):
    """ retourne la valeur du déterminant associé à un point et deux caméras
    et aux paramètres
    
    PARAMETRES
    C1, C2 [array (3,)] : coordonnées des centres des caméras, 
    theta1, theta2 [array (3,)] : angles de rotation associés aux caméras, 
    p1, p2 [array (2,)] : coordonnées des images des points sur les plans images des caméras
    
    """
   
    ca1, ca2 = np.cos(theta1[0]), np.cos(theta2[0])
    sa1, sa2 = np.sin(theta1[0]), np.sin(theta2[0])
    cb1, cb2 = np.cos(theta1[1]), np.cos(theta2[1])
    sb1, sb2 = np.sin(theta1[1]), np.sin(theta2[1])
    cg1, cg2 = np.cos(theta1[2]), np.cos(theta2[2])
    sg1, sg2 = np.sin(theta1[2]), np.sin(theta2[2])
    x1, x2 = p1[0], p2[0]
    y1, y2 = p1[1], p2[1]
    m1, m2, m3 = C2[0]-C1[0], C2[1]-C1[1], C2[2]-C1[2]
    
    return (
    m1 * ((ca1*(-sg1*x1+cg1*y1) + sa1*(sb1*(cg1*x1+sg1*y1) + f*cb1)) * (sa2*(sg2*x2-cg2*y2) + ca2*(sb2*(cg2*x2+sg2*y2) + f*cb2)) - (ca2*(-sg2*x2+cg2*y2) + sa2*(sb2*(cg2*x2+sg2*y2) + f*cb2)) * (sa1*(sg1*x1-cg1*y1) + ca1*(sb1*(cg1*x1+sg1*y1) + f*cb1)))
    - m2 * ((cb1*(cg1*x1+sg1*y1) - f*sb1) * (sa2*(sg2*x2-cg2*y2) + ca2*(sb2*(cg2*x2+sg2*y2) + f*cb2)) - (cb2*(cg2*x2+sg2*y2) - f*sb2) * (sa1*(sg1*x1-cg1*y1) + ca1*(sb1*(cg1*x1+sg1*y1) + f*cb1)))
    + m3 * ((cb1*(cg1*x1+sg1*y1) - f*sb1) * (ca2*(-sg2*x2+cg2*y2) + sa2*(sb2*(cg2*x2+sg2*y2) + f*cb2)) - (cb2*(cg2*x2+sg2*y2) - f*sb2) * ( ca1*(-sg1*x1+cg1*y1) + sa1*(sb1*(cg1*x1+sg1*y1) + f*cb1)))
    ) / f**2

def deriv_det(C1, C2, theta1, theta2, p1, p2, f):
    """ Retourne les dérivées partielles du déterminant associé à un point 
    et deux caméras et aux paramètres
    
    PARAMETRES
    C1, C2 [array (3,)] : coordonnées des centres des caméras, 
    theta1, theta2 [array (3,)] : angles de rotation associés aux caméras, 
    p1, p2 [array (2,)] : coordonnées des images des points sur les plans images des caméras
    
    SORTIE
    D [array (5, 2)] : retourne les dérivées partielles par rapport (dans l'ordre par ligne) aux alpha_i, beta_i, gamma_i, x_i et y_i. (colonne 0 : point 1, colonne 1 : point 2)
    
    """
    
    # notations
    ca1, ca2 = np.cos(theta1[0]), np.cos(theta2[0])
    sa1, sa2 = np.sin(theta1[0]), np.sin(theta2[0])
    cb1, cb2 = np.cos(theta1[1]), np.cos(theta2[1])
    sb1, sb2 = np.sin(theta1[1]), np.sin(theta2[1])
    cg1, cg2 = np.cos(theta1[2]), np.cos(theta2[2])
    sg1, sg2 = np.sin(theta1[2]), np.sin(theta2[2])
    x1, x2 = p1[0], p2[0]
    y1, y2 = p1[1], p2[1]
    m1, m2, m3 = C2[0]-C1[0], C2[1]-C1[1], C2[2]-C1[2]
    fk1 = cb1*(cg1*x1+sg1*y1) - f*sb1
    fl1 = cb2*(cg2*x2+sg2*y2) - f*sb2
    fk2 = ca1*(-sg1*x1+cg1*y1) + sa1*(sb1*(cg1*x1+sg1*y1) + f*cb1)
    fl2 = ca2*(-sg2*x2+cg2*y2) + sa2*(sb2*(cg2*x2+sg2*y2) + f*cb2)
    fk3 = sa1*(sg1*x1-cg1*y1) + ca1*(sb1*(cg1*x1+sg1*y1) + f*cb1)
    fl3 = sa2*(sg2*x2-cg2*y2) + ca2*(sb2*(cg2*x2+sg2*y2) + f*cb2)
    
    # remplissage de la matrice D
    D = np.zeros((5, 2))
    
    D[0, 0] = (m1 * ((-sa1*(-sg1*x1+cg1*y1) + ca1*(sb1*(cg1*x1+sg1*y1) + f*cb1)) * fl3 - fl2 * (ca1*(sg1*x1-cg1*y1) - sa1*(sb1*(cg1*x1+sg1*y1) + f*cb1)))
    - m2 * (- fl1 * (ca1*(sg1*x1-cg1*y1) - sa1*(sb1*(cg1*x1+sg1*y1) + f*cb1)))
    + m3 * (- fl1 * (-sa1*(-sg1*x1+cg1*y1) + ca1*(sb1*(cg1*x1+sg1*y1) + f*cb1))))
    
    D[0, 1] = (m1 * (fk2 * (ca2*(sg2*x2-cg2*y2) - sa2*(sb2*(cg2*x2+sg2*y2) + f*cb2)) - (-sa2*(-sg2*x2+cg2*y2) + ca2*(sb2*(cg2*x2+sg2*y2) + f*cb2)) * fk3)
    - m2 * (fk1 * (ca2*(sg2*x2-cg2*y2) - sa2*(sb2*(cg2*x2+sg2*y2) + f*cb2)))
    + m3 * (fk1 * (-sa2*(-sg2*x2+cg2*y2) + ca2*(sb2*(cg2*x2+sg2*y2) + f*cb2))))
    
    D[1, 0] = (m1 * (sa1*(cb1*(cg1*x1+sg1*y1) - f*sb1) * fl3 - fl2 * (ca1*(cb1*(cg1*x1+sg1*y1) - f*sb1)))
    - m2 * ((-sb1*(cg1*x1+sg1*y1) - f*cb1) * fl3 - fl1 * (ca1*(cb1*(cg1*x1+sg1*y1) - f*sb1)))
    + m3 * ((-sb1*(cg1*x1+sg1*y1) - f*cb1) * fl2 - fl1 * (sa1*(cb1*(cg1*x1+sg1*y1) - f*sb1))))
    
    D[1, 1] = (m1 * (fk2 * (ca2*(cb2*(cg2*x2+sg2*y2) - f*sb2)) - (sa2*(cb2*(cg2*x2+sg2*y2) - f*sb2)) * fk3)
    - m2 * (fk1 * (ca2*(cb2*(cg2*x2+sg2*y2) - f*sb2)) - (-sb2*(cg2*x2+sg2*y2) - f*cb2) * fk3)
    + m3 * (fk1 * (sa2*(cb2*(cg2*x2+sg2*y2) - f*sb2)) - (-sb2*(cg2*x2+sg2*y2) - f*cb2) * fk2))
    
    D[2, 0] = (m1 * ((ca1*(-cg1*x1-sg1*y1) + sa1*sb1*(-sg1*x1+cg1*y1)) * fl3 - fl2 * (sa1*(cg1*x1+sg1*y1) + ca1*sb1*(-sg1*x1+cg1*y1)))
    - m2 * (cb1*(-sg1*x1+cg1*y1) * fl3 - fl1 * (sa1*(cg1*x1+sg1*y1) + ca1*sb1*(-sg1*x1+cg1*y1)))
    + m3 * (cb1*(-sg1*x1+cg1*y1) * fl2 - fl1 * (ca1*(-cg1*x1-sg1*y1) + sa1*sb1*(-sg1*x1+cg1*y1))))
    
    D[2, 1] = (m1 * (fk2 * (sa2*(cg2*x2+sg2*y2) + ca2*sb2*(-sg2*x2+cg2*y2)) - (ca2*(-cg2*x2-sg2*y2) + sa2*sb2*(-sg2*x2+cg2*y2)) * fk3)
    - m2 * (fk1 * (sa2*(cg2*x2+sg2*y2) + ca2*sb2*(-sg2*x2+cg2*y2)) - cb2*(-sg2*x2+cg2*y2) * fk3)
    + m3 * (fk1 * (ca2*(-cg2*x2-sg2*y2) + sa2*sb2*(-sg2*x2+cg2*y2)) - cb2*(-sg2*x2+cg2*y2) * fk2))
    
    D[3, 0] = (m1 * ((ca1*-sg1 + sa1*sb1*cg1) * fl3 - fl2 * (sa1*sg1 + ca1*sb1*cg1))
    - m2 * (cb1*cg1 * fl3 - fl1 * (sa1*sg1 + ca1*sb1*cg1))
    + m3 * (cb1*cg1 * fl2 - fl1 * (ca1*-sg1 + sa1*sb1*cg1)))
    
    D[3, 1] = (m1 * (fk2 * (sa2*sg2 + ca2*sb2*cg2) - (ca2*-sg2 + sa2*sb2*cg2) * fk3)
    - m2 * (fk1 * (sa2*sg2 + ca2*sb2*cg2) - cb2*cg2 * fk3)
    + m3 * (fk1 * (ca2*-sg2 + sa2*sb2*cg2) - cb2*cg2 * fk2))
    
    D[4, 0] = (m1 * ((ca1*cg1 + sa1*sb1*sg1) * fl3 - fl2 * (sa1*-cg1 + ca1*sb1*sg1))
    - m2 * (cb1*sg1 * fl3 - fl1 * (sa1*-cg1 + ca1*sb1*sg1))
    + m3 * (cb1*sg1 * fl2 - fl1 * (ca1*cg1 + sa1*sb1*sg1)))
    
    D[4, 1] = (m1 * (fk2 * (sa2*-cg2 + ca2*sb2*sg2) - (ca2*cg2 + sa2*sb2*sg2) * fk3)
    - m2 * (fk1 * (sa2*-cg2 + ca2*sb2*sg2) - cb2*sg2 * fk3)
    + m3 * (fk1 * (ca2*cg2 + sa2*sb2*sg2) - cb2*sg2 * fk2))
    
    return D / f**2

def genere_liste_couples(K):
    """ crée la liste des couples (i1, i2) pour i1<i2 entre 0 et K-1 """
    
    liste_couples = []
    for i1 in range(K-1):
        for i2 in range(i1+1, K):
            liste_couples.append((i1, i2))
    
    return liste_couples

def matrices_Aj_Bj(C, theta, x, j, f, liste_couples):
    """ Calcule simultanné des matrices Aj et Bj associées au j-ième point de x
    
    PARAMETRES
    C [array (K, 3)] : ensemble des coordonnées des K caméras
    theta [array (K, 3)] : ensemble des angles de rotations des caméras
    x [array (N, K, 2)]: ensemble des coordonnées des images sur les K caméras des N points
    
    """
    
    K = C.shape[0]
    Aj, Bj = np.zeros((K*(K-1)//2, 3*K)), np.zeros((K*(K-1)//2, 2*K))
    for (l, (i1, i2)) in enumerate(liste_couples):
        derivees = deriv_det(C[i1], C[i2], theta[i1], theta[i2], x[j, i1], x[j, i2], f)
        Aj[l, 3*i1:3*i1+3] = derivees[:3, 0]
        Aj[l, 3*i2:3*i2+3] = derivees[:3, 1]
        Bj[l, 2*i1:2*i1+2] = derivees[3:, 0]
        Bj[l, 2*i2:2*i2+2] = derivees[3:, 1]
    
    return Aj, Bj

def matrice_Aj(C, theta, x, j, f, liste_couples):
    """ Calcule la matrice Aj associée au j-ième point de x
    
    PARAMETRES
    C [array (K, 3)] : ensemble des coordonnées des K caméras
    theta [array (K, 3)] : ensemble des angles de rotations des caméras
    x [array (N, K, 2)]: ensemble des coordonnées des images sur les K caméras des N points
    
    """
    
    K = C.shape[0]
    Aj = np.zeros((K*(K-1)//2, 3*K))
    for (l, (i1, i2)) in enumerate(liste_couples):
        derivees = deriv_det(C[i1], C[i2], theta[i1], theta[i2], x[j, i1], x[j, i2], f)
        Aj[l, 3*i1:3*i1+3] = derivees[:3, 0]
        Aj[l, 3*i2:3*i2+3] = derivees[:3, 1]
    
    return Aj

def matrice_Bj(C, theta, x, j, f, liste_couples):
    """ Calcule la matrice Bj associée au j-ième point de x 
    
    PARAMETRES
    C [array (K, 3)] : ensemble des coordonnées des K caméras
    theta [array (K, 3)] : ensemble des angles de rotations des caméras
    x [array (N, K, 2)]: ensemble des coordonnées des images sur les K caméras des N points
    
    """
    
    K = C.shape[0]
    Bj = np.zeros((K*(K-1)//2, 2*K))
    for (l, (i1, i2)) in enumerate(liste_couples):
        derivees = deriv_det(C[i1], C[i2], theta[i1], theta[i2], x[j, i1], x[j, i2], f)
        Bj[l, 2*i1:2*i1+2] = derivees[3:, 0]
        Bj[l, 2*i2:2*i2+2] = derivees[3:, 1]
    
    return Bj

def matrices_A_B(C, theta, x, f):
    """ Calcule les matrices A et B
    
    PARAMETRES
    C [array (K, 3)] : ensemble des coordonnées des K caméras
    theta [array (K, 3)] : ensemble des angles de rotations des caméras
    x [array (N, K, 2)] : ensemble des coordonnées des images sur les K caméras des N points
    
    """
    
    K, N = C.shape[0], x.shape[0]
    A, B = np.zeros((K*(K-1)//2*N, 3*K)), np.zeros((K*(K-1)//2*N, 2*K*N))
    for j in range(N):
        A[K*(K-1)//2*j:K*(K-1)//2*(j+1)], B[K*(K-1)//2*j:K*(K-1)//2*(j+1), 2*K*j:2*K*(j+1)] = matrices_Aj_Bj(C, theta, x, j, f, genere_liste_couples(K))
    
    return A, B

def matrice_A(C, theta, x, f):
    """ Calcule la matrice A
    
    PARAMETRES
    C [array (K, 3)] : ensemble des coordonnées des K caméras
    theta [array (K, 3)] : ensemble des angles de rotations des caméras
    x [array (N, K, 2)]: ensemble des coordonnées des images sur les K caméras des N points
    
    """
    
    K, N = C.shape[0], x.shape[0]
    A = np.zeros((K*(K-1)//2*N, 3*K))
    for j in range(N):
        A[K*(K-1)//2*j:K*(K-1)//2*(j+1)] = matrice_Aj(C, theta, x, j, f, genere_liste_couples(K))
    
    return A

def matrice_B(C, theta, x, f):
    """ Calcule la matrice B
    
    PARAMETRES
    C [array (K, 3)] : ensemble des coordonnées des K caméras
    theta [array (K, 3)] : ensemble des angles de rotations des caméras
    x [array (N, K, 2)]: ensemble des coordonnées des images sur les K caméras des N points
    
    """
    
    K, N = C.shape[0], x.shape[0]
    B = np.zeros((K*(K-1)//2*N, 2*K*N))
    for j in range(N):
        B[K*(K-1)//2*j:K*(K-1)//2*(j+1), 2*K*j:2*K*(j+1)] = matrice_Bj(C, theta, x, j, f, genere_liste_couples(K))
    
    return B

def F(C, theta, x, f):
    """ Calcule les valeurs de tous les déterminants du problème
    
    PARAMETRES
    C [array (K, 3)] : ensemble des coordonnées des K caméras
    theta [array (K, 3)] : ensemble des angles de rotations des caméras
    x [array (N, K, 2)]: ensemble des coordonnées des images sur les K caméras des N points
    
    SORTIE
    FF [list (N*K*(K-1)//2)] : valeurs de tous les déterminants dans l'ordre de nos conventions
    
    """
    
    N, K = x.shape[:-1]
    FF = []
    for j in range(N):
        for (i1, i2) in genere_liste_couples(K):
            FF.append(det(C[i1], C[i2], theta[i1], theta[i2], x[j, i1], x[j, i2], f))
    
    return np.array(FF)

def RX(t):
    """ matrice de rotation selon X d'angle t """
    
    return(np.array([[1, 0, 0], 
                     [0, np.cos(t), -np.sin(t)], 
                     [0, np.sin(t), np.cos(t)]]))

def RY(t):
    """ matrice de rotation selon Y d'angle t """
    
    return(np.array([[np.cos(t), 0, np.sin(t)], 
                     [0, 1, 0], 
                     [-np.sin(t), 0, np.cos(t)]]))

def RZ(t):
    """ matrice de rotation selon Z d'angle t """
    
    return(np.array([[np.cos(t), -np.sin(t), 0], 
                     [np.sin(t), np.cos(t), 0], 
                     [0, 0, 1]]))

def matrice_R(theta):
    """ matrice de rotation de la forme Rz Ry Zx 
    
    PARAMETRE
    theta [array (3,)] : angles de rotations
    
    """
    
    return np.dot(RZ(theta[2]), np.dot(RY(theta[1]), RX(theta[0])))

def matrice_P(C, theta, f):
    """ crée la matrice de projection d'une caméra
    
    PARAMETRES
    C [array (3,)] : coordonnées de la caméra
    theta [array (3,)] : angles de rotations de la caméra
    f : focale en pixels
    
    """
    
    R = matrice_R(theta)
    K = np.array([[f, 0, 0], [0, f, 0], [0, 0, 1]])
    P = np.zeros((3, 4))
    P[:, :3] = np.dot(K, R)
    P[:, 3] = np.dot(K, R).dot(-C)
    
    return P

def kr_from_p(P):
    """ Extract K, R and C from a camera matrix P, such that P = K*R*[eye(3) | -C]. 
    
    K is scaled so that K[2, 2] = 1 and K[0, 0] > 0. 
    
    """
    
    K, R = rq(P[:, :3])
    
    K /= K[2, 2]
    if K[0, 0] < 0:
        D = np.diag([-1, -1, 1])
        K = np.dot(K, D)
        R = np.dot(D, R)
    
    C = -np.linalg.solve(P[:, :3], P[:, 3])
    
    test = np.dot(np.dot(K, R), np.concatenate((np.eye(3), -np.array([C]).T), axis=1))
    np.testing.assert_allclose(test / test[2, 3], P / P[2, 3])
    
    return C, R, K

def theta_from_r(R):
    a = atan2(R[2, 1], R[2, 2])
    b = atan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))
    g = atan2(R[1, 0], R[0, 0])
    
    return a, b, g

def genere_donnees():
    """ crée aléatoirement des données 
    
    SORTIES
    C_reel [array (K, 3)] : coordonnées des caméras
    X_reel [array (N, 3)] : coordonnées des points
    theta_reel [array (K, 3)] : angles de rotation des caméras initiaux
    
    """
    
    C_reel = rand(K, 3)
    C_reel[:, 0] = (C_reel[:, 0] - 0.5) *2*X0
    C_reel[:, 1] *= (C_reel[:, 1] - 0.5) *2*Y0
    C_reel[:, 2] = C_reel[:, 2] * (Z2-Z1) + Z1
    
    theta_reel = np.zeros((K, 3))
    for i in range(K): # on fait pointer les caméras vers le point (0, 0, 0)
        X, Y, Z = -C_reel[i] / sl.norm(C_reel[i])
        beta = np.arcsin(-X)
        alpha = np.arcsin(Y/np.cos(beta))
        if np.cos(alpha)*np.cos(beta)*Z < 0: alpha = np.pi - alpha
        theta_reel[i, 0] = alpha
        theta_reel[i, 1] = beta
    
    X_reel = (rand(N, 3) - 0.5)
    X_reel[:, 0] *= X0
    X_reel[:, 1] *= Y0
    X_reel[:, 2] *= Z0
    
    return C_reel, X_reel, theta_reel

def scene_3D():
    """ crée une figure 3D correspondant aux données """
    
    Z_cam = []
    for theta in theta_reel:
        Z_cam.append(np.linalg.inv(matrice_R(theta)).dot(np.array([0, 0, 1])))
    
    plt.figure().gca(projection = "3d")
    plt.xlabel("X"), plt.ylabel("Y")
    plt.plot([0], [0], [0], color="k", marker="+", markersize=10)
    plt.plot(X_reel[:, 0], X_reel[:, 1], X_reel[:, 2], marker="+", markersize=3, color="b", label="points", linestyle="None")
    plt.plot(C_reel[:, 0], C_reel[:, 1], C_reel[:, 2], marker="+", markersize=10, color="r", label="cameras", linestyle="None")
    plt.legend(loc="best")
    plt.xlim(-X0, X0), plt.ylim(-Y0, Y0)
    
    t = 200000
    for C, Z in zip(C_reel, Z_cam):
        ZZ = C + t*Z
        plt.plot([C[0], ZZ[0]], [C[1], ZZ[1]], [C[2], ZZ[2]], color="k")    # trace les axes Z_cam