import os
import sys
import glob
import re
import multiprocessing
import math
import matplotlib.image as mpimg
import scipy.integrate as integrate
import scipy.special as special
from scipy.optimize import minimize_scalar
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
from scipy.optimize import minimize
matplotlib.use('TkAgg', force=True)
# matplotlib.use('qtagg') dont use qtagg
from matplotlib import pyplot as plt

print("Switched to:", matplotlib.get_backend())

np.set_printoptions(precision=64)
# np.set_default_dtype(np.float64)
num= lambda low_con, up_con: 1 if low_con == up_con else 100
def plot_data1(data_file, num):
    data = open(data_file, "r+").readlines()
    bitrate, psnr = [], []

    for i in data:
        x = i.strip().split(" ")
        bitrate.append(float(x[0]))
        psnr.append(float(x[1]))

    return bitrate, psnr
# def psi(sigma_X, sigma_X_hat):
#     # Compute psi according to the given formula
#     sigma_X_hat=np.clip(sigma_X_hat,0.000000001,1)
#     log_term = np.log(sigma_X / sigma_X_hat)
#     fraction_term = (sigma_X_hat ** 2 - sigma_X ** 2) / (2 * sigma_X ** 2)
#     return log_term + fraction_term
def tau(sigma_X_hat, R, R_c, P):
    # Compute tau according to the given formula
    term1 = 1 - np.exp(-2 * R)
    term2 = 1 - np.exp(-2 * (R + R_c + P - psi(1,sigma_X_hat)))
    return np.sqrt(term1 * term2)
#equation 123
def compute_expression(sigma_X, sigma_X_hat, R, R_c, P):

    # Compute the right hand side
    tau_value = tau(sigma_X_hat, R, R_c, P)
    term1 = 2 * sigma_X_hat - 2 * sigma_X * tau_value
    psi_value=psi(sigma_X, sigma_X_hat)
    numerator = 2 * (1 - np.exp(-2 * R)) * np.exp(-2 * (R + R_c + P -psi_value )) * (sigma_X ** 2 - sigma_X_hat ** 2)
    denominator = sigma_X * tau_value
    term2 = numerator / denominator
    # term2 = numerator / denominator if denominator != 0 else 0

    rhs = term1 - term2

    return  rhs

def find_sign_change(arr):
    # Convert the list to a numpy array
    arr = np.array(arr)

    # Create a boolean array where True represents positive numbers
    pos_mask = arr > 0

    # Find the indices where the sign changes (from positive to non-positive)
    sign_change_indices = np.where(np.diff(pos_mask))[0]

    return sign_change_indices
def find_sign_change_inv(arr):
    # Convert the list to a numpy array
    arr = np.array(arr)

    # Create a boolean array where True represents positive numbers
    pos_mask = arr <0

    # Find the indices where the sign changes (from positive to non-positive)
    sign_change_indices = np.where(np.diff(pos_mask))[0]

    return sign_change_indices
def plot_E123():
    # R=0.021556101633966512
    # highlight equation
    # R= 0.0215559
    R=0.023714783924290925
    Rc = 1
    sigmaX = 1
    P=1
    sigmaXH = sigmaP(P)
    # R_c = 0, R_c = 1, R_c =\infty
    Rc0_list = []  # d0
    # Rc1_list = []  # d1
    R_list = []  # di
    sigmax_hat__list = []
    start = 0
    # stop = 0.22906133994565517
    stop=1
    num = 1000000
    # overlineD(R,Rc,P,sigmaX,sigmaXH)  rc=1 P=1 ,sigmaX=1,sigmaXH=
    for sigma_X_hat in np.linspace(start, stop, num):

        # if sigmaXH-sigma_X_hat<=0.000001:
        #     print(f"sigmaXH---{sigmaXH}")
        R_list.append(sigma_X_hat)  #
        # compute_expression(sigma_X, sigma_X_hat, R, R_c, P)
        value=compute_expression(sigmaX, sigma_X_hat, R, Rc, P)
        Rc0_list.append(value)
        # if np.isclose(value, 0):
        #     print(f'value={value},sigma_X_hat={sigma_X_hat}')

        # Rc1_list.append(overlineD_hat(R, Rc, P, sigmaX, sigmaXH))
    sign_change_indices = find_sign_change(Rc0_list)[0]
    print('compare sigmas')
    print(f'R=={R},diffff={R_list[sign_change_indices]-sigmaXH},sigma_X_hat={R_list[sign_change_indices]}, F_value={Rc0_list[sign_change_indices]}')
    print(f'diffff={R_list[sign_change_indices+1]-sigmaXH},sigma_X_hat={R_list[sign_change_indices+1]}, F_value={Rc0_list[sign_change_indices+1]}')
    print('compare values')
    sign_change_indices_diff = find_sign_change(R_list-sigmaXH)[0]
    print(f'R=={R},diff={R_list[sign_change_indices_diff]-sigmaXH},sigma_X_hat={R_list[sign_change_indices_diff]}, F_value={Rc0_list[sign_change_indices_diff]}')
    print(f'diff={R_list[sign_change_indices_diff+1]-sigmaXH},sigma_X_hat={R_list[sign_change_indices_diff+1]}, F_value={Rc0_list[sign_change_indices_diff+1]}')
    print(f"P=={P},sigmaXH===sigmaP=={sigmaXH}")
    label1 = r'$\overline{D}^{\prime}$'
    label0 = r'$R=0.0237, R_c=1, P=1, \sigma^2_X=1$'

    # label2=r'$R=\infty$'
    # line3 = plt.plot(R_list,Rci_list, 'sienna', label=label2, linewidth=2.0)
    # line2 = plt.plot(R_list, Rc1_list, 'r', label=label1, linewidth=2.0)
    line1 = plt.plot(R_list, Rc0_list, 'b', label=label0, linewidth=2.0)


    plt.ylabel(r'$F$')
    plt.xlabel(r'$\sigma_{\hat{X}}$')

    plt.legend(loc='upper right')
    plt.savefig('E123_1.pdf')
    plt.show()

def plot_E123_2():
    # highlight equation??R=1, R_c=1, P=0.005957, sigma^2_X=1, ?\sigma_{\hat{X}}???????highlighted expression???
    R= 1
    Rc = 1
    sigmaX = 1
    P=0.005957
    sigmaXH = sigmaP(P)
    # R_c = 0, R_c = 1, R_c =\infty
    Rc0_list = []  # d0
    # Rc1_list = []  # d1
    R_list = []  # di
    sigmax_hat__list = []
    start = 0
    # stop = 0.923837196166139
    num = 100000
    stop=1
    # overlineD(R,Rc,P,sigmaX,sigmaXH)  rc=1 P=1 ,sigmaX=1,sigmaXH=
    for sigma_X_hat in np.linspace(start, stop, num):
        #
        R_list.append(sigma_X_hat)  #
        # compute_expression(sigma_X, sigma_X_hat, R, R_c, P)
        Rc0_list.append(compute_expression(sigmaX, sigma_X_hat, R, Rc, P))
    print(f"P=={P},sigmaXH===sigmaxP=={sigmaXH}, value={Rc0_list[-1]}")

        # Rc1_list.append(overlineD_hat(R, Rc, P, sigmaX, sigmaXH))
    sign_change_indices = find_sign_change(Rc0_list)[0]
    print(f'value={Rc0_list[sign_change_indices]},sigma_X_hat={R_list[sign_change_indices]}')
    print(f'value={Rc0_list[sign_change_indices+1]},sigma_X_hat={R_list[sign_change_indices+1]}')
    label1 = r'$\overline{D}^{\prime}$'
    label0 = r'$R=1, R_c=1, P=0.005957, sigma^2_X=1$'

    # label2=r'$R=\infty$'
    # line3 = plt.plot(R_list,Rci_list, 'sienna', label=label2, linewidth=2.0)
    # line2 = plt.plot(R_list, Rc1_list, 'r', label=label1, linewidth=2.0)
    line1 = plt.plot(R_list, Rc0_list, 'r', label=label0, linewidth=2.0)


    plt.ylabel(r'$F$')
    plt.xlabel(r'$\sigma_{\hat{X}}$')

    plt.legend(loc='upper right')
    plt.savefig('E123_2.pdf')
    plt.show()
    # print("sigmaXH====", sigmaXH)


# binary entropy function
def Hb(x):
    if x == 0:
        return 0
    else:
        Hb = (-x * math.log2(x)) - ((1 - x) * math.log2((1 - x)))
        return Hb


# ternary entropy function
def Ht(x, y):
    if x == 0 or y == 0:
        return 0
    else:
        Ht = (-x * math.log2(x)) - (y * math.log2(y)) - ((1 - x - y) * math.log2(1 - x - y))
        return Ht


# if p < (sigma_x_value - sqrt(sigma_x_value^2 - sigma_x_value^2*2^(-2*r)))^2
#             DPR_values(i,j) = sigma_x_value^2 + (sigma_x_value-sqrt(p))^2 - 2*sigma_x_value*(sigma_x_value-sqrt(p))*sqrt(1-2^(-2*r));
#             P_active(i,j) = 1;
#         else
#             DPR_values(i,j) = sigma_x_value^2*2^(-2*r);
#         end
# def RcrDP(P,D,sigma=1):
#     T=sigma-math.sqrt(abs(math.pow(sigma, 2)-D))
#
#     if math.sqrt(P) <=T:
#         a=math.pow(sigma, 2)*math.pow(sigma-math.sqrt(P), 2)
#         c=math.pow(sigma, 2)+math.pow(sigma-math.sqrt(P), 2)
#         d=math.pow((c-D)/2, 2)
#         # c=(math.pow(sigma, 2)+math.pow(sigma-math.sqrt(P), 2))-D))/2
#         e=a-d
#         return  (1/2)*math.log2(a/e)
#     else:
#         return max((1/2)*math.log2(math.pow(sigma, 2)/D),0)
def RcrDP(P, D, sigma=1):
    T = 1 - math.sqrt(abs(1 - D))

    if math.sqrt(P) <= T:
        a = 1 * math.pow(1 - math.sqrt(P), 2)
        c = 1 + math.pow(1 - math.sqrt(P), 2)
        d = math.pow((c - D) / 2, 2)
        # c=(math.pow(sigma, 2)+math.pow(sigma-math.sqrt(P), 2))-D))/2
        e = a - d
        return (1 / 2) * math.log2(a / e)
    else:
        return max((1 / 2) * math.log2(math.pow(1, 2) / D), 0)


def RD(R, Rc, sigma=1):
    a = math.sqrt((1 - math.pow(2, (-2 * R))) * (1 - math.pow(2, (-2 * (R + Rc)))))
    # a=(1-math.pow(2,-2*R)) #Rc=0
    D = 2 * (math.pow(sigma, 2)) - 2 * (math.pow(sigma, 2)) * a
    # D=2-2*a
    # D=2-2*(1-math.pow(2,-2*R))#Rc=0

    return D




def R_cr1(D, P=0, theta=1 / 4, theta_bar=3 / 4):
    if D >= 0 and D <= P:
        return (theta - D) / theta * Hb(theta)
    elif D >= (2 * theta * theta_bar - (theta_bar - theta) * P):
        return 0
    else:
        x = (2 * theta * theta_bar - (theta_bar - theta) * P - D) / (2 * theta * theta_bar)
        return x * Hb(theta)


def R_pr1(D, P=0, theta=1 / 4, theta_bar=3 / 4):
    if D >= 0 and D <= P:
        return Hb(theta - D)
    elif D >= (2 * theta * theta_bar - (theta_bar - theta) * P):
        return 0
    else:
        x = (2 * theta * theta_bar - (theta_bar - theta) * P - D) / (2 * theta_bar + P - D)
        return Hb(x)


def R_crinfi(D, P=0, theta=1 / 4, theta_bar=3 / 4):
    if D >= 0 and D <= P / (1 - 2 * (theta - P)):
        return Hb(theta) - Hb(D)
    elif D >= (2 * theta * theta_bar - (theta_bar - theta) * P):
        return 0
    else:
        return 2 * Hb(theta) + Hb(theta - P) - Ht((D - P) / 2, theta) - Ht((D + P) / 2, theta_bar)


# slide 37
def GaussianRDP():
    directory = "../results/"
    theta = 1 / 4
    theta_bar = 3 / 4
    P = 0
    cnt = 0
    # R_c = 0, R_c = 1, R_c =\infty
    Rc0_list = []
    Rc1_list = []
    Rci_list = []
    d_list = []
    start = 0.05
    stop = 2
    num = 1000
    y_list = []
    for d in np.linspace(start, stop, num):
        d_list.append(d)
        Rc1_list.append(RcrDP(0.05, d, sigma=1))
        # # Rcr1=R_cr1(d)
        Rc0_list.append(RcrDP(0, d, sigma=1))
        Rci_list.append(RcrDP(math.inf, d, sigma=1))
        y_list.append(0)
    label0 = r'$R^{(\infty)}_{\mathrm{cr}}(D,0.05)$'
    label1 = r'$R^{(\infty)}_{\mathrm{cr}}(D,0)$'
    label2 = r'$R^{(\infty)}_{\mathrm{cr}}(D,\infty)$'
    line3 = plt.plot(d_list, Rci_list, 'sienna', label=label2, linewidth=2.0)
    line1 = plt.plot(d_list, Rc1_list, 'r', label=label0, linewidth=2.0)
    line2 = plt.plot(d_list, Rc0_list, 'b', label=label1, linewidth=2.0)

    # line4 = plt.plot(d_list, y_list, 'dot', label=label2, linewidth=2.0)
    # line1 = plt.plot(R_list, Rc0_list, 'r', label=label0, linewidth=2.0)
    # line2 = plt.plot(R_list, Rc1_list, 'b', label=label1, linewidth=2.0)
    # line3 = plt.plot(R_list, R_list, 'sienna', label=label2, linewidth=2.0)
    plt.xlabel(r'$D$')
    plt.ylabel(r'$R$')
    plt.legend(loc='upper right')
    # bbox_to_anchor = (0., 1.02, 1., .102),
    plt.savefig(directory + "Rcr(DR)" + 'R_D.pdf')
    plt.show()
    # plt.clf()
    # cnt += 1


# slide26 Guassian RD
def GuassianRD():
    directory = "../results/"
    theta = 1 / 4
    theta_bar = 3 / 4
    P = 0
    cnt = 0
    # R_c = 0, R_c = 1, R_c =\infty
    Rc0_list = []
    Rc1_list = []
    Rci_list = []
    R_list = []
    start = 0.00001
    stop = 4
    num = 1000
    for r in np.linspace(start, stop, num):
        R_list.append(r)
        Rc0_list.append(RD(r, 0, sigma=1))
        # Rcr1=R_cr1(d)
        Rc1_list.append(RD(r, 1, sigma=1))
        Rci_list.append(RD(r, math.inf, sigma=1))
    label0 = r'$R_{\mathrm{c}}=0$'
    label1 = r'$R_{\mathrm{c}}=1$'
    label2 = r'$R_{\mathrm{c}}=\infty$'
    line3 = plt.plot(Rci_list, R_list, 'sienna', label=label2, linewidth=2.0)
    line2 = plt.plot(Rc1_list, R_list, 'r', label=label1, linewidth=2.0)
    line1 = plt.plot(Rc0_list, R_list, 'b', label=label0, linewidth=2.0)

    # line1 = plt.plot(R_list, Rc0_list, 'r', label=label0, linewidth=2.0)
    # line2 = plt.plot(R_list, Rc1_list, 'b', label=label1, linewidth=2.0)
    # line3 = plt.plot(R_list, R_list, 'sienna', label=label2, linewidth=2.0)
    plt.xlabel(r'$D$')
    plt.ylabel(r'$R$')
    plt.legend(loc='upper right')
    # bbox_to_anchor = (0., 1.02, 1., .102),
    plt.savefig(directory + 'Gaussian R_D.pdf')
    plt.show()
    # plt.clf()
    # cnt += 1


# Binary Case slide page 34
def BinaryCase():
    directory = "../results/"
    theta = 1 / 4
    theta_bar = 3 / 4
    P = 0
    cnt = 0
    Rcr1_list = []
    Rpr1_list = []
    Rcrinfi_list = []
    d_list = []
    start = 0.00001
    stop = 3 / 8
    num = 1000
    for d in np.linspace(start, stop, num):
        d_list.append(d)
        Rcr1_list.append(R_cr1(d))
        # Rcr1=R_cr1(d)
        Rpr1_list.append(R_pr1(d))
        Rcrinfi_list.append(R_crinfi(d))
    label0 = r'$R^{(1)}_{\mathrm{cr}}(D,0)$'
    label1 = r'$R^{(1)}_{\mathrm{pr}}(D,0)$'
    # label2=r'$R^{(\infty)}_{\mathrm{cr}}(D,0)$'
    label2 = r'$R(D,0)$'
    line3 = plt.plot(d_list, Rcrinfi_list, 'sienna', label=label2, linewidth=2.0)
    line1 = plt.plot(d_list, Rcr1_list, 'r', label=label0, linewidth=2.0)
    # line2 = plt.plot(d_list, Rpr1_list, 'b', label=label1, linewidth=2.0)

    plt.xlabel(r'$D$')
    plt.ylabel(r'$R$')
    plt.legend(loc='upper right')
    # bbox_to_anchor = (0., 1.02, 1., .102),
    plt.savefig(directory + 'binary RD cr pr .pdf')
    plt.show()
    # plt.clf()
    # cnt += 1


def custom_function(d, p):
    if p > d:
        p = d
    x = p + 1 - np.sqrt(p ** 2 + 1 - 2 * d)

    return 1 - Hb(x / 2)


def pdCase():
    directory = "../results/"
    # Generate values for d and p
    d_values = np.linspace(0.000001, 0.5, 100)
    p_values = np.linspace(0.000001, d_values[-1], 100)

    # Create a meshgrid for d and p
    D, P = np.meshgrid(d_values, p_values)
    # Apply the condition P < D
    mask = P < D
    D = np.ma.masked_where(~mask, D)
    P = np.ma.masked_where(~mask, P)
    Z = np.ma.masked_where(~mask, custom_function(D, P))
    # Calculate the function values for each combination of d and p
    # Z = custom_function(D, P)

    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(D, P, Z, cmap='viridis')

    ax.set_xlabel('d')
    ax.set_ylabel('p')
    ax.set_zlabel('1 - H(p + (1 - sqrt(1 - 2d))/2)')
    ax.set_title('Function: 1 - H(p + (1 - sqrt(1 - 2d))/2)')

    # plt.show()
    # plt.savefig(directory + 'binary RD cr pr .pdf')
    plt.show()
    # plt.clf()
    # cnt += 1


#######
def ThreeDplot():
    # def custom_function(d, p):
    #     if p > d:
    #         p = d
    #     x = p + 1 - np.sqrt(p ** 2 + 1 - 2 * d)
    #
    #     return 1 - Hb(x / 2)
    # Function to be plotted
    def custom_function(d, p):
        x = p + 1 - np.sqrt(p ** 2 + 1 - 2 * d)
        return 1 - binary_entropy(x / 2)

    # Generate values for d and p
    d_values = np.linspace(0.001, 0.499, 100)  # Range for d: [0, 1/2)
    p_values = np.linspace(0.001, d_values[-1], 100)  # Range for p: [0, d_max)

    # Create a meshgrid for d and p values
    D, P = np.meshgrid(d_values, p_values)

    # Apply the condition P < D
    mask = P < D
    # Apply the mask to Z
    Z = custom_function(D, P)
    Z = np.where(mask, Z, np.nan)
    # # Apply the mask to D, P, and Z
    # D_masked = np.ma.masked_where(~mask, D)
    # P_masked = np.ma.masked_where(~mask, P)
    # Z_masked = np.ma.masked_where(~mask, custom_function(D, P))

    # Plotting a 3D surface plot for P < D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(D, P, Z, cmap='viridis', alpha=0.7)

    ax.set_xlabel('d')
    ax.set_ylabel('p')
    ax.set_zlabel('1 - H_b(p + sqrt(1 - 2d^2)/2)')

    plt.title('Custom Function 3D Plot for P < D')
    plt.show()

    # Binary entropy function
    def binary_entropy(p):
        return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


##############  paper  ##################
#### Theorem 2 GAUSSIAN CASE ####
def epsilon(R, Rc):
    epsilon = math.sqrt((1 - math.exp(-2 * R)) * (1 - math.exp(-2 * (R + Rc))))
    return epsilon


def RD1(R, Rc, muX, muXH, sigmaX, sigmaXH):
    epsilon1 = epsilon(R, Rc)
    D = math.pow((muX - muXH), 2) + math.pow(sigmaX, 2) + math.pow(sigmaXH, 2) - 2 * sigmaX * sigmaXH * epsilon1
    # D = math.pow((muX - muXH),2) + math.pow(sigmaX,2)+ math.pow(sigmaXH,2)-2*sigmaX*sigmaXH*epsilon
    return D


def plotTheorem2():
    # theta = 1 / 4
    # theta_bar = 3 / 4
    # P = 0
    # cnt = 0
    # R_c = 0, R_c = 1, R_c =\infty
    Rc0_list = []  # d0
    Rc1_list = []  # d1
    Rci_list = []  # di
    R_list = []
    start = 0.00001
    stop = 4
    num = 1000
    for r in np.linspace(start, stop, num):
        R_list.append(r)
        Rc0_list.append(RD1(r, Rc=0, muX=0, muXH=1, sigmaX=1, sigmaXH=math.sqrt(4)))
        # Rcr1=R_cr1(d)
        Rc1_list.append(RD1(r, Rc=1, muX=0, muXH=1, sigmaX=1, sigmaXH=math.sqrt(4)))
        Rci_list.append(RD1(r, Rc=math.inf, muX=0, muXH=1, sigmaX=1, sigmaXH=math.sqrt(4)))
    label0 = r'$R_{\mathrm{c}}=0$'
    label1 = r'$R_{\mathrm{c}}=1$'
    label2 = r'$R_{\mathrm{c}}=\infty$'
    line3 = plt.plot(R_list, Rci_list, 'sienna', label=label2, linewidth=2.0)
    line2 = plt.plot(R_list, Rc1_list, 'r', label=label1, linewidth=2.0)
    line1 = plt.plot(R_list, Rc0_list, 'b', label=label0, linewidth=2.0)

    # line1 = plt.plot(R_list, Rc0_list, 'r', label=label0, linewidth=2.0)
    # line2 = plt.plot(R_list, Rc1_list, 'b', label=label1, linewidth=2.0)
    # line3 = plt.plot(R_list, R_list, 'sienna', label=label2, linewidth=2.0)
    # plt.xlabel(r'$D$')
    # plt.ylabel(r'$R$')
    plt.ylabel(r'$D$')
    plt.xlabel(r'$R$')
    plt.legend(loc='upper right')
    # bbox_to_anchor = (0., 1.02, 1., .102),
    # plt.savefig(directory  + 'paper R_D.pdf')
    plt.savefig('Theorem2_R1_4.pdf')
    plt.show()


#### Corollary 2  ####
def sigmaRRcP(R, Rc, P, sigmaX):
    a = math.sqrt(1 - math.exp(-2 * R))
    b = math.sqrt(1 - math.exp(-2 * (R + Rc)))
    c = sigmaX * b

    beta1 = a * math.exp(-2 * (R + Rc))
    beta1 = beta1 / c

    beta2 = sigmaX * math.sqrt((1 - math.exp(-2 * R)) * (1 - math.exp(-2 * (R + Rc + P))))
    d = sigmaX * a * (math.exp(-2 * (R + Rc)))
    beta2 = beta2 + d / b

    if R == 0:
        return 0
    else:
        sigma = math.sqrt(1 + 4 * beta1 * beta2) - 1
        sigma = sigma / (2 * beta1)
        return sigma


# def sigmaP(P):
#     from sympy import symbols, Eq, log, solve

#     y = symbols('x')
#     eq1 = Eq(log(1/y) + (y**2 - 1)/2-P)
#     sol = solve(eq1)
#     if len(sol)==0:
#         sol=[0]
#     sigmaP=sol[0]
#     # print(f"sigmaP====={sigmaP}")

#     return sigmaP
def sigmaP(P):
    from scipy.optimize import root_scalar
    def equation(y, P):
        return 1 / np.exp(P - (y ** 2 - 1) / 2) - y

    # Initial guess
    initial_guess = 0.0

    # Value for P
    P_value = P  # You need to specify the value of P

    # Call the solver function
    result = root_scalar(equation, args=(P_value), x0=initial_guess, x1=1, method='secant')

    # Get the solution
    solution = result.root

    print("Numerical Solution for y:", solution)

    return solution


def psi(sigmaX, sigmaXH):
    # if sigmaXH==0:
    #     sigmaXH=0.0001

    if sigmaXH == 0:

        return math.inf
    else:

        psi = math.log(sigmaX / sigmaXH) + (math.pow(sigmaXH, 2) - math.pow(sigmaX, 2)) / (2 * math.pow(sigmaX, 2))
        # print(f"sigmaX={sigmaX},sigmaXH={sigmaXH},psi={psi}")
        # psi=np.clip(psi,0,0.1) #P=0.1
        # print(f"sigmaX={sigmaX},sigmaXH={sigmaXH},psi={psi}")
        return psi

    # def sigmaP(P):


#     from sympy import symbols, Eq, log, solve
#     # math.log(sigmaX/sigmaXH)+(math.pow(sigmaXH,2)-math.pow(sigmaX,2))/(2*math.pow(sigmaX,2))-x
#     y = symbols('x')
#     eq1 = Eq(log(1/y) + (y**2 - 1)/2-P)
#     eq1=Eq
#     sol = solve(eq1)
#     if len(sol)==0:
#         sol=[0]
#     sigmaP=sol[0]
#     print(f"sigmaP====={sigmaP}")

#     return sigmaP

# def sigmaP(P):
#     from sympy import symbols, Eq, log, solve
#     from scipy.optimize import fsolve
#     def equation(y):
#         return log(1/y) + (y**2 - 1)/2-1
#     # math.log(sigmaX/sigmaXH)+(math.pow(sigmaXH,2)-math.pow(sigmaX,2))/(2*math.pow(sigmaX,2))-x
#     args=(1)
#     x=fsolve(equation,0)
#     print("sigmap==={x[0]}")
#     sigmaP=x[0]

#     return sigmaP

# Rc=math.inf   #Corollary moreover
def C2_DRRcP(R, sigmaXH):
    sigmaX = 1
    # P=0, sigmaP=1;P=1,sigmaP=0.23, P=inf, sigmaP=0
    # sigmaXH=sigmaP
    # slowly when interate P
    # sigmaXH=sigmaP(P)
    # ###faster
    # if P==0:
    #     sigmaXH=1
    # elif P==math.inf:
    #     sigmaXH=0
    # elif P==1:
    #     sigmaXH=0.23
    # elif P==0.1:
    #     sigmaXH=0.702310062419124

    threshold = sigmaX * math.sqrt(1 - math.exp(-2 * R))
    if sigmaXH <= threshold:
        D = math.pow(sigmaX, 2) * math.exp(-2 * R)
    else:
        d = 2 * sigmaX * sigmaXH * math.sqrt(1 - math.exp(-2 * R))
        D = math.pow(sigmaX, 2) + math.pow(sigmaXH, 2) - d
    return D


# Rc=math.inf   #Corollary moreover
def C2_DRRcP2(R, Rc, P, sigmaX, sigmaXH):
    # sigmaX=1
    # sigmaXH=sigmaP(P)
    # ###faster
    if P == 0:
        sigmaXH = 1
    elif P == math.inf:
        sigmaXH = 0
    elif P == 1:
        sigmaXH = 0.23
    elif P == 0.1:
        sigmaXH = 0.702310062419124
    if P == 0:
        return 2 * math.pow(sigmaX, 2) * (1 - epsilon(R, Rc))
    elif Rc == math.inf:
        threshold = sigmaX * math.sqrt(1 - math.exp(-2 * R))
        if (math.sqrt(P)) <= threshold:
            return math.pow(sigmaX, 2) * math.exp(-2 * R)
        else:
            d = 2 * sigmaX * sigmaXH * math.sqrt(1 - math.exp(-2 * R))
            return math.pow(sigmaX, 2) + math.pow(sigmaXH, 2) - d


# RC=math.inf, interate R set P=0,==0.1,p=math.inf
def plotC2moreover():
    # R_c = 0, R_c = 1, R_c =\infty
    Rc0_list = []  # d0
    Rc1_list = []  # d1
    Rci_list = []  # di
    R_list = []
    start = 0.0
    stop = 2
    num = 3000
    Rc = math.inf
    sigmaXH0 = sigmaP(P=0)
    sigmaXH1 = sigmaP(P=0.1)
    sigmaXHi = sigmaP(P=math.inf)
    for r in np.linspace(start, stop, num):
        print(f"r==={r}")
        # RRcPD3(R,Rc,P,sigmaX=1)
        R_list.append(r)  # P list
        Rc0_list.append(C2_DRRcP(r, sigmaXH0))
        # Rcr1=R_cr1(d)s
        Rc1_list.append(C2_DRRcP(r, sigmaXH1))
        Rci_list.append(C2_DRRcP(r, sigmaXHi))
    # label0=r'$P_{\mathrm{c}}=0$'
    # label1 = r'$P_{\mathrm{c}}=1$'
    # label2=r'$P_{\mathrm{c}}=2$'
    label0 = r'$P=0$'
    label1 = r'$P=0.1$'
    label2 = r'$P=\infty$'
    line3 = plt.plot(R_list, Rci_list, 'sienna', label=label2, linewidth=2.0)
    line2 = plt.plot(R_list, Rc1_list, 'r', label=label1, linewidth=2.0)
    line1 = plt.plot(R_list, Rc0_list, 'b', label=label0, linewidth=2.0)

    plt.ylabel(r'$D$')
    plt.xlabel(r'$R$')
    plt.legend(loc='upper right')
    plt.savefig('Corollary2case1DR.pdf')
    plt.show()


def plotC2moreoverDP():
    # Corollary 2
    # RC=math.inf, interate P set R=0,R=0.1,R=math.inf

    Rc0_list = []  # d0
    Rc1_list = []  # d1
    Rci_list = []  # di
    R_list = []
    start = 0.0
    stop = 0.3
    num = 10000
    Rc = math.inf
    for p in np.linspace(start, stop, num):
        print(f"p==={p}")
        sigmaXH = sigmaP(p)
        print(f"sigmaXH==={sigmaXH}")
        R_list.append(p)  # P list
        Rc0_list.append(C2_DRRcP(R=0, sigmaXH=sigmaXH))
        # Rcr1=R_cr1(d)
        Rc1_list.append(C2_DRRcP(R=0.1, sigmaXH=sigmaXH))
        Rci_list.append(C2_DRRcP(R=0.5, sigmaXH=sigmaXH))
    # label0=r'$P_{\mathrm{c}}=0$'
    # label1 = r'$P_{\mathrm{c}}=1$'
    # label2=r'$P_{\mathrm{c}}=2$'
    label0 = r'$R=0$'
    label1 = r'$R=0.1$'
    label2 = r'$R=0.5$'
    # label2=r'$R=\infty$'
    line3 = plt.plot(R_list, Rci_list, 'sienna', label=label2, linewidth=2.0)
    line2 = plt.plot(R_list, Rc1_list, 'r', label=label1, linewidth=2.0)
    line1 = plt.plot(R_list, Rc0_list, 'b', label=label0, linewidth=2.0)

    plt.ylabel(r'$D$')
    plt.xlabel(r'$P$')
    plt.legend(loc='upper right')
    plt.savefig('Corollary2moreoverDP.pdf')
    plt.show()


##### in particular P=0 ###
def C2_DR(R, Rc):
    sigmaX = 1
    D = 2 * math.pow(sigmaX, 2) * (1 - epsilon(R, Rc))
    return D


def plotC2C2():
    # DP when R=0 R=1 R= mathi.inf
    Rc0_list = []  # d0
    Rc1_list = []  # d1
    Rci_list = []  # di
    R_list = []
    start = 0.0
    stop = 50
    num = 50
    R = 0
    sigmaX = 1
    # C2_DRRcP2(R,Rc,P,sigmaX,sigmaXH)
    # sigmaXH0=sigmaP(P=0)
    # sigmaXH1=sigmaP(P=0.1)
    # sigmaXHi=sigmaP(P=math.inf)
    for r in np.linspace(start, stop, num):
        print(f"r==={r}")
        sigmaXH = sigmaP(r)
        # RRcPD3(R,Rc,P,sigmaX=1)
        R_list.append(r)  # P list
        Rc0_list.append(C2_DRRcP2(R, 0, r, sigmaX, sigmaXH))
        # Rcr1=R_cr1(d)s
        Rc1_list.append(C2_DRRcP2(R, 1, r, sigmaX, sigmaXH))
        Rci_list.append(C2_DRRcP2(R, math.inf, r, sigmaX, sigmaXH))
    # label0=r'$P_{\mathrm{c}}=0$'
    # label1 = r'$P_{\mathrm{c}}=1$'
    # label2=r'$P_{\mathrm{c}}=2$'
    label0 = r'$R=0$'
    label1 = r'$R=1$'
    label2 = r'$R=\infty$'
    line3 = plt.plot(R_list, Rci_list, 'sienna', label=label2, linewidth=2.0)
    line2 = plt.plot(R_list, Rc1_list, 'r', label=label1, linewidth=2.0)
    line1 = plt.plot(R_list, Rc0_list, 'b', label=label0, linewidth=2.0)

    plt.ylabel(r'$D$')
    plt.xlabel(r'$P$')
    plt.legend(loc='upper right')
    plt.savefig('Corollary2case2.pdf')
    plt.show()


#### Theorem 3  ####  computr sigmaXH(sigmaP) based on P value
# def psi(sigmaX, sigmaXH):
#     if sigmaXH == 0:
#         return math.inf
#     else:
#         psi = math.log(sigmaX / sigmaXH) + (math.pow(sigmaXH, 2) - math.pow(sigmaX, 2)) / (2 * math.pow(sigmaX, 2))
#         return psi
# def overlineD(R, Rc, P, sigmaX, sigmaXH):
#     # P=0, sigmaP=1;P=1,sigmaP=0.23, P=inf, sigmaP=0
#     threshold = sigmaX * epsilon(R, Rc)
#     if sigmaXH <= threshold:
#         return math.pow(sigmaX, 2) - math.pow(sigmaX, 2) * math.pow(epsilon(R, Rc), 2)
#     else:
#         d = 2 * sigmaX * sigmaXH * epsilon(R, Rc)
#         D = math.pow(sigmaX, 2) + math.pow(sigmaXH, 2) - d
#         return D
def overlineDinfty(R, sigmaX):
    d= (sigmaX**2)*math.exp(-2*R)
    return d
def overlineD(R, Rc, P, sigmaX, sigmaXH):#r'$\overline{D}(R,R_c,v(P)|\phi_{KL})$'  r'$\overline{D}(R,R_c,P|\phi_{KL})$'
    # P=0, sigmaP=1;P=1,sigmaP=0.23, P=inf, sigmaP=0
    threshold = sigmaX * epsilon(R, Rc)
    if sigmaXH <= threshold:
        return math.pow(sigmaX, 2) - math.pow(sigmaX, 2) * math.pow(epsilon(R, Rc), 2)
    else:
        d = 2 * sigmaX * sigmaXH * epsilon(R, Rc)
        D = math.pow(sigmaX, 2) + math.pow(sigmaXH, 2) - d
        return D

def overlineDpart2_11(R, Rc, P, sigmaX, sigmaXH):
    # sigmaX=1
    # Rc=0
    # P=0.1
    # sigmaXH=sigmaP(P) #ouyside out of function
    # P=0, sigmaP=1;P=1,sigmaP=0.23, P=inf, sigmaP=0
    threshold = sigmaX * epsilon(R, Rc)
    if sigmaXH <= threshold:
        return math.pow(sigmaX, 2) - math.pow(sigmaX, 2) * math.pow(epsilon(R, Rc), 2)
    else:
        d = 2 * sigmaX * sigmaXH * epsilon(R, Rc)
        D = math.pow(sigmaX, 2) + math.pow(sigmaXH, 2) - d
        return D

def overlineD_hat(R, Rc, P, sigmaX, sigmaXH):# r'$\overline{D}(R,R_c,P|W^2_2)$'
    # P=0, sigmaP=1;P=1,sigmaP=0.23, P=inf, sigmaP=0
    threshold = sigmaX * epsilon(R, Rc)
    if sigmaX - math.sqrt(P) <= threshold:
        return math.pow(sigmaX, 2) - math.pow(sigmaX, 2) * math.pow(epsilon(R, Rc), 2)
    else:
        d = 2 * sigmaX * (sigmaX - math.sqrt(P)) * epsilon(R, Rc)
        D = math.pow(sigmaX, 2) + math.pow(sigmaX - math.sqrt(P), 2) - d
        return D


def make_minD1(args):
    R, Rc, P, sigmaX = args
    print(f'{math.exp(-2 * (R + Rc + P - 0.1))}')
    minD = lambda sigmaX_H: math.pow(sigmaX, 2) + math.pow(sigmaX_H, 2) - 2 * sigmaX * sigmaX_H * math.sqrt(
        1 - math.exp(-2 * R)) * math.sqrt((1 - math.exp(-2 * (R + Rc + np.clip(P - psi(sigmaX, sigmaX_H),0,None)))))
    # minD = lambda sigmaX_H: math.pow(sigmaX, 2) + math.pow(sigmaX_H, 2) - 2 * sigmaX * sigmaX_H * math.sqrt(
    #     1 - math.exp(-2 * R)) * math.sqrt(max(0,1 - math.exp(-2 * (R + Rc + P - psi(sigmaX, sigmaX_H)))))
    return minD

def make_minDW(args):
    R, Rc, P, sigmaX = args
    print(f'{math.exp(-2 * (R + Rc + P - 0.1))}')
    minD = lambda sigmaX_H: math.pow(sigmaX, 2) + math.pow(sigmaX_H, 2) - 2 * sigmaX * sigmaX_H * math.sqrt(
        1 - math.exp(-2 * R)) * math.sqrt(sigmaX_H**2-np.clip(math.pow(sigmaX*math.exp(-(R+Rc))-math.sqrt(P),2), 0, None))

    # minD = lambda sigmaX_H: math.pow(sigmaX, 2) + math.pow(sigmaX_H, 2) - 2 * sigmaX * sigmaX_H * math.sqrt(
    #     1 - math.exp(-2 * R)) * math.sqrt(max(0,1 - math.exp(-2 * (R + Rc + P - psi(sigmaX, sigmaX_H)))))
    return minD

def constraint(args):
    sigmaXH_min, sigmaXH_max = args
    cons = ({'type': 'ineq', 'fun': lambda x: x - sigmaXH_min}, \
            {'type': 'ineq', 'fun': lambda x: -x + sigmaXH_max})
    return cons

def make_minD1_hat_T6(args):
    R, Rc, P, sigmaX,a= args
    # low_con=sigmaX-math.sqrt(P)
    # when a=1

    # aa = math.pow(sigmaX, 2) - a * np.clip((sigmaX ** 2 +  sigmaX_H** 2 - P), 0, None) + (a ** 2) * (sigmaX_H ** 2)
    # # aa=math.sqrt(np.clip(math.pow(sigmaX, 2) - 2 * k * j * sigmaX * i + math.pow(k, 2) * math.pow(i, 2),0,None))
    # cc = math.pow(np.clip((sigmaX*math.exp(-(R+Rc)  ) - math.sqrt(aa)), 0, None), 2)
    # if cc <= 1e-10:
    #     cc = 0
    # ga = np.clip(cc, 0, None) / math.pow(a, 2)

    # max_a= np.clip(math.pow(np.clip((sigmaX*math.exp(-(R+Rc)  ) - math.sqrt(math.pow(sigmaX, 2) - a * np.clip((sigmaX ** 2 +  sigmaX_H** 2 - P), 0, None) + (a ** 2) * (sigmaX_H ** 2))), 0, None), 2), 0, None) / math.pow(a, 2)
    if a < 0:
        a = 0
    # print(f"r====a==========={a}")

    minD = lambda sigmaX_H: (math.pow(sigmaX, 2) + math.pow(sigmaX_H, 2) - 2 * sigmaX * math.sqrt(
        (1 - math.exp(-2 * R))) * math.sqrt((math.pow(sigmaX_H, 2) - np.clip(math.pow(np.clip((sigmaX*math.exp(-(R+Rc)  ) - math.sqrt(math.pow(sigmaX, 2) - a * np.clip((sigmaX ** 2 +  sigmaX_H** 2 - P), 0, None) + (a ** 2) * (sigmaX_H ** 2))), 0, None), 2), 0, None) / math.pow(a, 2))))
    return minD
def minimaxFunction(a, sigmaX_H, R, Rc,P, sigmaX):
    # a, sigmaxH = x[0], x[1]
    minD = (math.pow(sigmaX, 2) + math.pow(sigmaX_H, 2) - 2 * sigmaX * math.sqrt(
        (1 - math.exp(-2 * R))) * math.sqrt((math.pow(sigmaX_H, 2) - np.clip(math.pow(np.clip((sigmaX*math.exp(-(R+Rc)  ) - math.sqrt( np.clip( math.pow(sigmaX, 2) - a * np.clip((sigmaX ** 2 +  sigmaX_H** 2 - P), 0, None) + (a ** 2) * (sigmaX_H ** 2),0,None) )), 0, None), 2), 0, None) / math.pow(a, 2))))
    return minD

import numpy as np
from scipy.optimize import minimize
import math

def make_minD1_hat_T6_minimax(R, Rc, P, sigmaX):
    def inner_max_problem(sigmaX_H, R, Rc, P, sigmaX):
        # Inner maximization objective function for 'a'
        def obj_max(a):
            # Correct the calculation for max_a
            aa = np.clip(math.pow(sigmaX, 2) - a[0] * np.clip(sigmaX ** 2 + sigmaX_H ** 2 - P, 0, None) + (a[0] ** 2) * (sigmaX_H ** 2),0,None)
            delta = np.clip(sigmaX * math.exp(-(R + Rc)) - math.sqrt(aa), 0, None)
            max_a = np.clip(math.pow(delta, 2), 0, None) / (a[0] ** 2) if a[0] != 0 else np.inf  # Avoid division by zero
            return -max_a  # Maximizing by minimizing the negative

        # Initial guess and bounds for 'a'
        a0 = np.array([0.1])
        bounds_a = [(0.0001, None)]  # a must be greater than zero

        # Maximization as minimization of the negative of the objective
        result = minimize(obj_max, a0, bounds=bounds_a, method='L-BFGS-B')
        return -result.fun, result.x[0]

    def outer_min_problem(R, Rc, P, sigmaX):
        # Outer minimization objective function for 'sigmaX_H'
        def obj_min(sigmaX_H):
            max_value, _ = inner_max_problem(sigmaX_H[0], R, Rc, P, sigmaX)
            return max_value

        # Initial guess and bounds for 'sigmaX_H'
        sigmaX_H0 = np.array([0.1])
        bounds_sigmaX_H = [(0.0001, None)]  # sigmaX_H must be non-negative

        # Minimization of the outer function
        result = minimize(obj_min, sigmaX_H0, bounds=bounds_sigmaX_H, method='L-BFGS-B')
        _, optimal_a = inner_max_problem(result.x[0], R, Rc, P, sigmaX)
        return result.fun, result.x[0], optimal_a

    # Perform the outer minimization
    min_value, optimal_sigmaX_H, optimal_a = outer_min_problem(R, Rc, P, sigmaX)

    print("Optimal values:")
    print("sigma_X_H:", optimal_sigmaX_H)
    print("a:", optimal_a)
    print("Minimum of the maximum value of the objective:", min_value)
    return min_value

# Example usage:
# make_minD1_hat_T6_minimax(1.0, 0.5, 2.0, 3.0)
# def make_minD1_hat_T6_minimax( R, Rc,P, sigmaX):
#     # R, Rc, P, sigmaX,sigmaX_H,a= args
#     # low_con=sigmaX-math.sqrt(P)
#     # when a=1
#     #
#     # aa = math.pow(sigmaX, 2) - a * np.clip((sigmaX ** 2 +  sigmaX_H** 2 - P), 0, None) + (a ** 2) * (sigmaX_H ** 2)
#     # # aa=math.sqrt(np.clip(math.pow(sigmaX, 2) - 2 * k * j * sigmaX * i + math.pow(k, 2) * math.pow(i, 2),0,None))
#     # cc = math.pow(np.clip((sigmaX*math.exp(-(R+Rc)  ) - math.sqrt(aa)), 0, None), 2)
#     # if cc <= 1e-10:
#     #     cc = 0
#     # ga = np.clip(cc, 0, None) / math.pow(a, 2)
#     #
#     # max_a= np.clip(math.pow(np.clip((sigmaX*math.exp(-(R+Rc)  ) - math.sqrt(math.pow(sigmaX, 2) - a * np.clip((sigmaX ** 2 +  sigmaX_H** 2 - P), 0, None) + (a ** 2) * (sigmaX_H ** 2))), 0, None), 2), 0, None) / math.pow(a, 2)
#     def inner_max_problem(a, sigmaX_H, R, Rc,P, sigmaX):
#         def obj_max(a):
#             max_a= np.clip(math.pow(np.clip((sigmaX*math.exp(-(R+Rc)  ) - math.sqrt(math.pow(sigmaX, 2) - a * np.clip((sigmaX ** 2 +  sigmaX_H** 2 - P), 0, None) + (a ** 2) * (sigmaX_H ** 2))), 0, None), 2), 0, None) / math.pow(a, 2)
#             return max_a
#         # if a < 0:
#         #     a = 0
#         a0=np.ndarray((1))
#         low_con=sigmaX-math.sqrt(P)
#         up_con=sigmaX
#         bounds=[(low_con,up_con)]
#         result = minimize(obj_max, a0, bounds=bounds, method='L-BFGS-B')
#         return -result.x
#
#     def outer_min_problem(R, Rc, sigmaX):
#         # Define the overall problem as a function of sigma_X_H
#         def obj_min(sigmaX_H):
#             return inner_max_problem(a, sigmaX_H, R, Rc,P, sigmaX)
#
#         # Initial guess for sigma_X_H
#         sigmaX_H0 = np.asarray((0.0001))
#         bounds=[(0.00001,None)]
#
#         # Minimization of the outer function
#         result = minimize(obj_min, sigmaX_H0,bounds=bounds, method='L-BFGS-B')
#         return result
#
#     # Perform the outer minimization
#     result = outer_min_problem(R, Rc, sigmaX)
#
#     print("Optimal values:")
#     print("sigma_X_H:", result.x[0])
#     print("Minimum of the maximum value of the objective:", result.fun)
#     return result.x
#     # minD = (math.pow(sigmaX, 2) + math.pow(sigmaX_H, 2) - 2 * sigmaX * math.sqrt(
#     #     (1 - math.exp(-2 * R))) * math.sqrt((math.pow(sigmaX_H, 2) - np.clip(math.pow(np.clip((sigmaX*math.exp(-(R+Rc)  ) - math.sqrt(math.pow(sigmaX, 2) - a * np.clip((sigmaX ** 2 +  sigmaX_H** 2 - P), 0, None) + (a ** 2) * (sigmaX_H ** 2))), 0, None), 2), 0, None) / math.pow(a, 2))))
#     # return minD
def make_minD1_hat(args):
    R, Rc, P, sigmaX = args
    # low_con=sigmaX-math.sqrt(P)
    # sigmaX_H=sigmaP(P)
    a = (sigmaX * math.exp(-(R + Rc)) - math.sqrt(P))
    if a < 0:
        a = 0
    # print(f"r========={R}")

    minD = lambda sigmaX_H: (math.pow(sigmaX, 2) + math.pow(sigmaX_H, 2) - 2 * sigmaX * math.sqrt(
        (1 - math.exp(-2 * R))) * math.sqrt(np.clip((math.pow(sigmaX_H, 2) - math.pow(a, 2)),0,None)))
    return minD

def constraint_hat(args):
    sigmaXH_min, sigmaXH_max = args
    cons = ({'type': 'ineq', 'fun': lambda x: x - sigmaXH_min}, \
            {'type': 'ineq', 'fun': lambda x: -x + sigmaXH_max})
    return cons


def underlineD(R, Rc, P, sigmaX, sigmaXH):
    # sigmaXH:sigmaP
    # sigmaX_H: sigmax^
    if P==math.inf:
        res=sigmaX**2 * math.exp(-2*R)
        return res
    else:

        from scipy.optimize import minimize
        # from scipy.optimize import minimize_scalar,Bounds

        args = (R, Rc, P, sigmaX)

        args1 = (sigmaXH, sigmaX)
        cons = constraint(args1)
        x0 = sigmaXH

        # print(f'R==={R},P==={P},psi==={psi(sigmaX, sigmaXH)}')
        # res = minimize(make_minD1(args), x0, method='L-BFGS-B',
        #                    options={'xatol': 1e-8, 'disp': True}, constraints=cons)
        res = minimize(make_minD1(args), x0, method='SLSQP',
                       options={'xatol': 1e-8, 'disp': True}, constraints=cons)
        # print(f'res.===={res.x[0]}')
        # print(f'res.fun===={res.fun}')


        return res.fun

def underlineDpart2_12(R, Rc, P, sigmaX, sigmaXH):
    # sigmaXH:sigmaP
    # sigmaX_H: sigmax^
    from scipy.optimize import minimize
    # from scipy.optimize import minimize_scalar,Bounds

    args = (R, Rc, P, sigmaX)

    args1 = (sigmaXH, sigmaX)
    cons = constraint(args1)
    x0 = sigmaXH
    res = minimize(make_minD1(args), x0, method='SLSQP',
                   options={'xatol': 1e-8, 'disp': True}, constraints=cons)
    print(f'res.===={res.x[0]}')
    print(f'res.fun===={res.fun}')

    return res.fun

def underlineD_hat(R, Rc, P, sigmaX, sigmaXH):
    # sigmaXH:sigmaP
    # sigmaX_H: sigmax^
    from scipy.optimize import minimize
    # from scipy.optimize import minimize_scalar,Bounds

    args = (R, Rc, P, sigmaX)

    low_con =np.clip(sigmaX - math.sqrt(P),0,None)
    if low_con < 0:
        low_con = 0

    args1 = (low_con, sigmaX)
    cons = constraint(args1)
    x0 = sigmaXH
    res = minimize(make_minD1_hat(args), x0, method='SLSQP',
                   options={
                       'xatol': 1e-8,
                            'disp': True}, constraints=cons)
    print(f"low and up {low_con},{sigmaX}")
    print(f'hat  blue line res.===={res.x[0]}')
    print(f'res.fun===={res.fun}')

    return res.fun

def underlineD_hat_T6(R, Rc, P, sigmaX, sigmaXH,a):
    # sigmaXH:sigmaP
    # sigmaX_H: sigmax^
    from scipy.optimize import minimize
    # from scipy.optimize import minimize_scalar,Bounds

    args = (R, Rc, P, sigmaX,a)

    low_con =np.clip(sigmaX - math.sqrt(P),0,None)
    if low_con < 0:
        low_con = 0

    args1 = (low_con, sigmaX)
    cons = constraint(args1)
    x0 = sigmaXH
    res = minimize(make_minD1_hat_T6(args), x0, method='SLSQP',
                   options={'xatol': 1e-8, 'disp': True}, constraints=cons)
    print(f"low and up {low_con},{sigmaX}")
    print(f'hat  blue line res.===={res.x[0]}')
    print(f'res.fun===={res.fun}')

    return res.fun
def underlineD_hat_T6_2(R, Rc, P, sigmaX, sigmaXH,a):
    # sigmaXH:sigmaP
    # sigmaX_H: sigmax^
    alist=[]
    from scipy.optimize import minimize
    # from scipy.optimize import minimize_scalar,Bounds
    for i in range(a,100000):
        args = (R, Rc, P, sigmaX,i)

        low_con =np.clip(sigmaX - math.sqrt(P),0,None)
        if low_con < 0:
            low_con = 0

        args1 = (low_con, sigmaX)
        cons = constraint(args1)
        x0 = sigmaXH
        res = minimize(make_minD1_hat_T6(args), x0, method='SLSQP',
                       options={'xatol': 1e-8, 'disp': True}, constraints=cons)
        print(f"low and up {low_con},{sigmaX}")
        print(f'hat  blue line res.===={res.x[0]}')
        print(f'res.fun===={res.fun}')
        if res.fun =='nan':
            alist.append(0)
        else:
            alist.append(res.fun)
    amax=max(alist[1:])
    return amax
def underlineD_hat_T6_minimax(R, Rc, P, sigmaX, sigmaXH):
    # sigmaXH:sigmaP
    # sigmaX_H: sigmax^

    # Define the inner function to minimiz

    alist=[]

    # from scipy.optimize import minimize_scalar,Bounds
    # for i in (0.0001,100):
    for i in np.linspace(0, 100000, 100):
        args = (R, Rc, P, sigmaX,i)

        low_con =np.clip(sigmaX - math.sqrt(P),0,None)
        if low_con < 0:
            low_con = 0

        args1 = (low_con, sigmaX)
        cons = constraint(args1)
        x0 = sigmaXH
        bounds = [(0, None), (low_con, sigmaX)]  # set bounds of a and sigma_xH
        res = minimize(make_minD1_hat_T6(args), x0, method='SLSQP',
                       options={'xatol': 1e-8, 'disp': True}, constraints=cons)
        print(f"low and up {low_con},{sigmaX}")
        print(f'hat  blue line res.===={res.x[0]}')
        print(f'res.fun===={res.fun}')
        if res.fun =='nan':
            alist.append(0)
        else:
            alist.append(res.fun)
    amax=max(alist[1:])
    return amax

def varsigma(R, Rc,  sigmaX):
    # P=1
    # sigmaX=1
    # sigmaXH=1
    """
    Calculates the value of the expression s(R, Rc) given the input parameters R and Rc.

    Parameters:
    R (float): The first parameter.
    Rc (float): The second parameter.

    Returns:
    float: The calculated value of the expression s(R, Rc).
    """
    numerator = sigmaX * np.sqrt(1 - np.exp(-2 * (R + Rc)) + 4 * (1 - np.exp(-2 * R)) * np.exp(-2 * (R + Rc)))
    denominator = 2 * np.sqrt(1 - np.exp(-2 * R)) * np.exp(-2 * (R + Rc))
    denominator=np.clip(denominator,0.0001,None)
    first_term = numerator / denominator
    second_term = -sigmaX * np.sqrt(1 - np.exp(-2 * (R + Rc))) / denominator
    # second_term = -sigmaX * np.sqrt(1 - np.exp(-2 * (R + Rc))) / (
    #             2 * np.sqrt(1 - np.exp(-2 * R)) * np.exp(-2 * (R + Rc)))

    return first_term + second_term
    # return first_term - second_term
def plotTheorem3DP():
    #   \mu_X=0, \sigma^2_X=1, R_c=1, P=1, 画\overline{D}和R的关系，以及\underline{D}和R关系。这两条曲线应该部分重合，如果不明显的话，把P改成0.5或0.1试试
    R= 1
    # P=1

    # sigmaXH = 0.2290613399456898
    Rc = 1
    sigmaX = 1
    # R_c = 0, R_c = 1, R_c =\infty
    Rc0_list = []  # d0
    Rc1_list = []  # d1
    sigmas = []  # di
    R_list = []
    start = 0
    stop = 0.02
    num = 1000001
    sigmaXHs = []
    varsigmas=varsigma(1, Rc, sigmaX=1)
    # overlineD(R,Rc,P,sigmaX,sigmaXH)  rc=1 P=1 ,sigmaX=1,sigmaXH=
    for P in np.linspace(start, stop, num):
        sigmaXH = sigmaP(P)
        R_list.append(P)  #

        Rc0_list.append(underlineD(R, Rc, P, sigmaX, sigmaXH))

        Rc1_list.append(overlineD(R, Rc, P, sigmaX, sigmaXH))
        sigmaXHs.append(sigmaXH)
        # sigmas.append(sigmaXH)
        # VarsigmaXHs=[]
        # RR=[]
        # for r in np.linspace(0.0000001, 1, 10000):
        #     VarsigmaXHs.append(varsigma(r, Rc, sigmaX=1))
        #     RR.append(r)
        # # Find common elements and their indices
        # # common_sigmaP_varsigma=[(i, elem) for i, elem in enumerate(varsigmas) if math.fabs(elem-sigmaP(P))<0.0001]
    common_sigmaP_varsigma = [(i, sigmaXHs[i], varsigmas) for i in range(len(sigmaXHs)) if
                              math.fabs(sigmaXHs[i]- varsigmas) < 0.00001]
    # common_sigmaP_varsigm = min(common_sigmaP_varsigma, key=lambda x: x[1])
    min_diff_element = min(common_sigmaP_varsigma, key=lambda x: abs(x[1] - x[2]))
    data_as_lists = [min_diff_element]

    common_elements_with_indices = [(i, Rc0_list[i], Rc1_list[i]) for i in range(min(len(Rc0_list), len(Rc1_list))) if
                                    math.fabs(Rc0_list[i] - Rc1_list[i]) < 0.000001]
    # common_elements_with_indices = [(i, elem) for i, elem in enumerate(Rc1_list) if elem in Rc0_list]
    # Print the result
    # print("Common elements with indices:", common_elements_with_indices)
    # Save the common elements and their indices to a text file
    with open('common_elementsFig6.txt', 'w') as file:
        file.write("P , underlineD, overlineD \n")
        for i, elem1, elem2 in common_elements_with_indices:
            file.write(f"{R_list[i]}, {elem1}, {elem2}\n")
        file.write("P,sigmaP,Varsigma\n")
        # for item in data_as_lists:
        #     file.write(f"{item}\\")
        for i, elem1, elem2 in data_as_lists:
            file.write(f"{R_list[i]}, {elem1} {elem2}\n")

    label0 = r'$\underline{D}$'
    label1 = r'$\overline{D}$'
    # label2=r'$R=\infty$'
    # line3 = plt.plot(R_list,Rci_list, 'sienna', label=label2, linewidth=2.0)
    line2 = plt.plot(R_list, Rc1_list, 'r', label=label1, linewidth=2.0)
    line1 = plt.plot(R_list, Rc0_list, 'b', label=label0, linewidth=2.0)


    plt.ylabel(r'$D$')
    plt.xlabel(r'$P$')

    plt.legend(loc='upper right')
    plt.savefig('theroem3PD_fig5PD1.pdf')
    plt.savefig('theroem3PD_fig5PD1.eps',format='eps')
    plt.show()
    # print("sigmaXH====", sigmaXH)


# fig 5 P=1
#fig5 revision P=\infinty
def find_closest_value(varsigmas, sigmaP):
    varsigmas_array = np.array(varsigmas)

    differences = np.abs(varsigmas_array - sigmaP)
    min_index = np.argmin(differences)

    return min_index,varsigmas_array[min_index]

from scipy.optimize import fsolve


def plotTheorem3():
    #   \mu_X=0, \sigma^2_X=1, R_c=1, P=1, 画\overline{D}和R的关系，以及\underline{D}和R关系。这两条曲线应该部分重合，如果不明显的话，把P改成0.5或0.1试试
    P = 1
    # sigmaXH = 0.2290613399456898 #sigmaP


    # P=math.inf
    sigmaXH = sigmaP(P)

    Rc = 1
    sigmaX = 1
    # R_c = 0, R_c = 1, R_c =\infty
    Rc0_list = []  # d0
    Rc1_list = []  # d1
    varsigmas = []  # di
    R_list = []
    start = 0.0
    stop = 0.03# stop =2 for paper fig5
    num = 10

    # overlineD(R,Rc,P,sigmaX,sigmaXH)  rc=1 P=1 ,sigmaX=1,sigmaXH=
    for r in np.linspace(start, stop, num):
        R_list.append(r)  #


        Rc0_list.append(underlineD(r, Rc, P, sigmaX, sigmaXH))

        Rc1_list.append(overlineD(r, Rc, P, sigmaX, sigmaXH))
        varsigmas.append(varsigma(r,Rc,sigmaX=1))

    label0 = r'$\underline{D}$'
    label1 = r'$\overline{D}$'
    label2=r'varsigmas'
    line3 = plt.plot(R_list,varsigmas, 'sienna', label=label2, linewidth=2.0)
    line2 = plt.plot(R_list, Rc1_list, 'r', label=label1, linewidth=2.0)
    line1 = plt.plot(R_list, Rc0_list, 'b', label=label0, linewidth=2.0)
    print('compare varsigmas[sign_change_indices]-sigmaXH')
    diff=varsigmas-sigmaXH
    sign_change_indices = find_sign_change(varsigmas-sigmaXH)[0]
    print(f'R=={R_list[sign_change_indices]},varsigmas={varsigmas[sign_change_indices]},sigmaXH={sigmaXH},diff={varsigmas[sign_change_indices]-sigmaXH}')
    print(f'R={R_list[sign_change_indices+1]},varsigmas={varsigmas[sign_change_indices+1]},diff={varsigmas[sign_change_indices+1]-sigmaXH}')

    sign_change_indices = find_sign_change_inv(varsigmas-sigmaXH)[0]
    print(f'R=={R_list[sign_change_indices]},varsigmas={varsigmas[sign_change_indices]},sigmaXH={sigmaXH},diff={varsigmas[sign_change_indices]-sigmaXH}')
    print(f'R={R_list[sign_change_indices+1]},varsigmas={varsigmas[sign_change_indices+1]},diff={varsigmas[sign_change_indices+1]-sigmaXH}')
    # # print(f"P=={P},sigmaXH===sigmaP=={sigmaXH}, value={Rc0_list[-1]}")
    # # print('compare F_values')
    # diff = sigmaXH - np.array(varsigmas)
    # min_idx = np.argmin(diff[diff >= 0])
    # print(f"The minimum value of sigmaXH - varsigmas >= 0 is: {varsigmas[min_idx]}")
    # sign_change_indices1 = np.argmin(sigmaXH-varsigmas)
    # print(f'R=={R_list[sign_change_indices1]},varsigmas={varsigmas[sign_change_indices1]},sigmaXH={sigmaXH},diff={varsigmas[sign_change_indices1]-sigmaXH}')
    # print(f'R={R_list[sign_change_indices1+1]},varsigmas={varsigmas[sign_change_indices1+1]},diff={varsigmas[sign_change_indices1+1]-sigmaXH}')
    # #

    # Find common elements and their indices
    index,closest_value = find_closest_value(varsigmas, sigmaXH)
    print(f'R_list[index]=={R_list[index]},closest_value==={closest_value},sigmaXH={sigmaXH}')
    # common_sigmaP_varsigma=[(i, elem) for i, elem in enumerate(varsigmas) if math.fabs(elem-sigmaP(P))<0.0001]
    common_sigmaP_varsigma=[(i, varsigmas[i], sigmaXH) for i in range( len(varsigmas)) if
     math.fabs(varsigmas[i]- sigmaXH) < 0.001]
    min_diff_element = min(common_sigmaP_varsigma, key=lambda x: abs(x[1] - x[2]))
    data_as_lists = [min_diff_element]

    common_elements_with_indices = [(i, Rc0_list[i], Rc1_list[i]) for i in range(min(len(Rc0_list), len(Rc1_list))) if
                       math.fabs(Rc0_list[i] - Rc1_list[i]) < 0.001]
    # common_elements_with_indices = [(i, elem) for i, elem in enumerate(Rc1_list) if elem in Rc0_list]
    # Print the result
    print("Common elements with indices:", common_elements_with_indices)
    # Save the common elements and their indices to a text file
    with open('common_elementsFig5.txt', 'w') as file:
        file.write("R , underlineD, overlineD \n")
        for i,  elem1, elem2 in common_elements_with_indices:
            file.write(f"{R_list[i]}, {elem1}, {elem2}\n")
        file.write("R,Varsigma , SigmaP\n")
        for i, elem1,elem2 in data_as_lists:
            file.write(f"{R_list[i]}, {elem1} {elem2}\n")
    plt.ylabel(r'$D$')
    plt.xlabel(r'$R$')

    plt.legend(loc='upper right')
    plt.savefig('theroem3RD_fig5.pdf')
    plt.savefig('theroem3RD_fig5.eps',format='eps')
    plt.show()
    print("sigmaXH====", sigmaXH)
def plotoverlap():
    import numpy as np
    import matplotlib.pyplot as plt

    # Define the range of x values
    x = np.linspace(-10, 10, 400)

    # Define two functions
    def func1(x):
        return np.sin(x)

    def func2(x):
        return np.cos(x)

    # Compute the function values
    y1 = func1(x)
    y2 = func2(x)

    # Plot the functions
    plt.figure(figsize=(10, 6))
    plt.plot(x, y1, label='sin(x)', color='blue')
    plt.plot(x, y2, label='cos(x)', color='red')

    # Highlight the overlap area
    overlap_x = x[(y1 < y2) & (y2 < y1)]
    overlap_y1 = y1[(y1 < y2) & (y2 < y1)]
    overlap_y2 = y2[(y1 < y2) & (y2 < y1)]

    plt.fill_between(overlap_x, overlap_y1, overlap_y2, color='black', alpha=0.5, label='Overlap Area')

    # Add labels and legend
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Overlap of sin(x) and cos(x)')
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()
def plotTheoremT5part2_12():
    #   \mu_X=0, \sigma^2_X=1, R_c=1, P=1, 画\overline{D}和R的关系，以及\underline{D}和R关系。这两条曲线应该部分重合，如果不明显的话，把P改成0.5或0.1试试
    P = 0.1
    sigmaXH = sigmaP(P)
    # sigmaXH = 0.2290613399456898 ###P=1
    # sigmaXH = 0.7023100624191236

    Rc = 0.00001
    # Rc = 1
    sigmaX = 1
    # R_c = 0, R_c = 1, R_c =\infty
    Rc0_list = []  # $\overline{D}(R,R_c,v(P)|\phi_{KL})$
    Rc1_list = []  # $\overline{D}(R,R_c,P|W^2_2)$
    Rc2_list = []  # di
    Rc3_list = []  # di
    R_list = []
    start = 0
    stop = 2
    num = 1000
    a=2*math.pow(sigmaX,2)
    vp=a/np.clip((a-P),0,None)
    vp=math.log(vp)
    Pb=sigmaP(vp)
    # overlineD(R,Rc,P,sigmaX,sigmaXH)  rc=1 P=1 ,sigmaX=1,sigmaXH=
    for r in np.linspace(start, stop, num):
        R_list.append(r)  #

        Rc0_list.append(overlineD(r, Rc, vp, sigmaX, Pb))

        Rc1_list.append(overlineD_hat(r, Rc, P, sigmaX, sigmaXH))

        # Rc2_list.append(underlineD_hat(r, Rc, P, sigmaX, sigmaXH))
        # Rc3_list.append(underlineD_hat_T6(r, Rc, P, sigmaX, sigmaXH,1))
        # Rc2_list.append(min_underDstar2forA(r, Rc, P, sigmaX)) # inaccurate
        # Rc2_list.append(underlineD_hat_T6(r, Rc, P, sigmaX, sigmaXH,1))
        # Rc3_list.append(min_underDstar2forA_Sup(r, Rc, P, sigmaX))

        Rc2_list.append(underlineD_hat_T6(r, Rc, P, sigmaX, sigmaXH, 1))

    label0 = r'$\overline{D}(R,R_c,v(P)|\phi_{KL})$'
    label1 = r'$\overline{D}(R,R_c,P|W^2_2)$'
    label2=r'$\underline{D}(R,R_c,P|W^2_2)$'
    label3=r'$for function$'
    line1 = plt.plot(R_list, Rc0_list, 'b', label=label0, linestyle='--', linewidth=2.0)

    line3 = plt.plot(R_list,Rc1_list, 'sienna', label=label1,linestyle='-', linewidth=2.0)
    line2 = plt.plot(R_list, Rc2_list, 'r', label=label2, linestyle=':', linewidth=2.0)
    # line2 = plt.plot(R_list, Rc3_list, 'g', label=label3, linestyle=':', linewidth=2.0)

    # # reordering the labels
    # handles, labels = plt.gca().get_legend_handles_labels()
    #
    # # specify order
    # order = [2,  1, 0]
    #
    # # pass handle & labels lists along with order as below
    # plt.legend([handles[i] for i in order], [labels[i] for i in order])


    plt.ylabel(r'$D$')
    plt.xlabel(r'$R$')

    plt.legend(loc='upper right')
    plt.savefig('correspondence_upper.pdf')
    plt.show()
    print("sigmaXH====", sigmaXH)
def plotTheoremT5part2_11():
    #   \mu_X=0, \sigma^2_X=1, R_c=1, P=1, 画\overline{D}和R的关系，以及\underline{D}和R关系。这两条曲线应该部分重合，如果不明显的话，把P改成0.5或0.1试试
    P = 0.1
    sigmaXH = sigmaP(P)
    # sigmaXH = 0.2290613399456898 ###P=1
    # sigmaXH = 0.7023100624191236

    Rc = 0
    sigmaX = 1
    # R_c = 0, R_c = 1, R_c =\infty
    Rc0_list = []  # r'$\underline{D}(R,R_c,P|\phi_{KL})$'
    Rc1_list = []  # r'$\underline{D}(R,R_c,2\sigma^2_X(1-e^{-P})|W^2_2)$'
    Rc2_list = []  # r'$\overline{D}(R,R_c,v(P)|\phi_{KL})$'
    R_list = []
    start = 0
    stop = 2
    num = 1000
    Pa=2*math.pow(sigmaX,2)*(1-math.exp(-P))
    Pb=sigmaP(Pa)
    # overlineD(R,Rc,P,sigmaX,sigmaXH)  rc=1 P=1 ,sigmaX=1,sigmaXH=
    for r in np.linspace(start, stop, num):
        R_list.append(r)  #

        Rc0_list.append(underlineD(r, Rc, P, sigmaX, sigmaXH))

        Rc1_list.append(underlineD_hat(r, Rc, Pa, sigmaX, Pb))
        Rc2_list.append(overlineD(r, Rc, P, sigmaX, sigmaXH))

    label0 = r'$\underline{D}(R,R_c,P|\phi_{KL})$'
    label1 = r'$\underline{D}(R,R_c,2\sigma^2_X(1-e^{-P})|W^2_2)$'
    label2=r'$\overline{D}(R,R_c,P|\phi_{KL})$'
    line2 = plt.plot(R_list,Rc2_list, 'b', label=label2,linestyle='--', linewidth=2.0)
    line0 = plt.plot(R_list, Rc0_list, 'sienna', label=label0, linestyle='-',linewidth=2.0)

    line1 = plt.plot(R_list, Rc1_list, 'r', label=label1, linestyle=':',linewidth=2.0)

    # line3 = plt.plot(R_list,Rc2_list, 'sienna', label=label2,linestyle='-', linewidth=2.0)
    # line2 = plt.plot(R_list, Rc1_list, 'r', label=label1, linestyle=':',linewidth=2.0)
    # line1 = plt.plot(R_list, Rc0_list, 'b', label=label0,linestyle='--', linewidth=2.0)



    plt.ylabel(r'$D$')
    plt.xlabel(r'$R$')

    plt.legend(loc='upper right')
    plt.savefig('correspondence_lower.pdf')
    plt.show()
    print("sigmaXH====", sigmaXH)


#### Corollary 3  ####
def RRcPD3(R, Rc, P, sigmaX=1):
    if P == 0:
        return 2 * math.pow(sigmaX, 2) * (1 - epsilon(R, Rc))
    elif Rc == math.inf:
        threshold = sigmaX * math.sqrt(1 - math.exp(-2 * R))
        if (sigmaX - math.sqrt(P)) <= threshold:
            return math.pow(sigmaX, 2) * math.exp(-2 * R)
        else:
            d = 2 * sigmaX * (sigmaX - math.sqrt(P)) * math.sqrt(1 - math.exp(-2 * R))

            return math.pow(sigmaX, 2) + math.pow((sigmaX - math.sqrt(P)), 2) - d


def plotC3C1():
    # R_c = 0, R_c = 1, R_c =\infty
    Rc0_list = []  # d0
    Rc1_list = []  # d1
    Rci_list = []  # di
    R_list = []
    start = 0.0
    stop = 2
    num = 1000
    Rc = math.inf
    for r in np.linspace(start, stop, num):
        # RRcPD3(R,Rc,P,sigmaX=1)
        R_list.append(r)
        Rc0_list.append(RRcPD3(r, Rc, P=0))
        # Rcr1=R_cr1(d)
        Rc1_list.append(RRcPD3(r, Rc, P=0.1))
        Rci_list.append(RRcPD3(r, Rc, P=1))
    # label0=r'$P_{\mathrm{c}}=0$'
    # label1 = r'$P_{\mathrm{c}}=1$'
    # label2=r'$P_{\mathrm{c}}=2$'
    label0 = r'$P=0$'
    label1 = r'$P=0.1$'
    label2 = r'$P=1$'
    line3 = plt.plot(R_list, Rci_list, 'sienna', label=label2, linewidth=2.0)
    line2 = plt.plot(R_list, Rc1_list, 'r', label=label1, linewidth=2.0)
    line1 = plt.plot(R_list, Rc0_list, 'b', label=label0, linewidth=2.0)

    # line1 = plt.plot(R_list, Rc0_list, 'r', label=label0, linewidth=2.0)
    # line2 = plt.plot(R_list, Rc1_list, 'b', label=label1, linewidth=2.0)
    # line3 = plt.plot(R_list, R_list, 'sienna', label=label2, linewidth=2.0)
    # plt.xlabel(r'$D$')
    # plt.ylabel(r'$R$')
    # plt.rc('text', usetex=True)
    plt.ylabel(r'$D$')
    plt.xlabel(r'$R$')
    plt.legend(loc='upper right')
    # bbox_to_anchor = (0., 1.02, 1., .102),
    # plt.savefig(directory  + 'paper R_D.pdf')
    plt.savefig('Corollary3Case1DR.eps',format='eps')
    plt.show()


def plotC3C2():
    # R_c = 0, R_c = 1, R_c =\infty
    Rc0_list = []  # d0
    Rc1_list = []  # d1
    Rci_list = []  # di
    R_list = []
    start = 0.0
    stop = 1
    num = 1000
    Rc = math.inf
    for P in np.linspace(start, stop, num):
        # RRcPD3(R,Rc,P,sigmaX=1)
        R_list.append(P)  # P list
        Rc0_list.append(RRcPD3(0, Rc, P))
        # Rcr1=R_cr1(d)
        Rc1_list.append(RRcPD3(0.1, Rc, P))
        Rci_list.append(RRcPD3(0.5, Rc, P))
    # label0=r'$P_{\mathrm{c}}=0$'
    # label1 = r'$P_{\mathrm{c}}=1$'
    # label2=r'$P_{\mathrm{c}}=2$'
    label0 = r'$R=0$'
    label1 = r'$R=0.1$'
    label2 = r'$R=0.5$'
    line3 = plt.plot(R_list, Rci_list, 'sienna', label=label2, linewidth=2.0)
    line2 = plt.plot(R_list, Rc1_list, 'r', label=label1, linewidth=2.0)
    line1 = plt.plot(R_list, Rc0_list, 'b', label=label0, linewidth=2.0)

    # line1 = plt.plot(R_list, Rc0_list, 'r', label=label0, linewidth=2.0)
    # line2 = plt.plot(R_list, Rc1_list, 'b', label=label1, linewidth=2.0)
    # line3 = plt.plot(R_list, R_list, 'sienna', label=label2, linewidth=2.0)
    # plt.xlabel(r'$D$')
    # plt.ylabel(r'$R$')
    # plt.rc('text', usetex=True)
    plt.ylabel(r'$D$')
    plt.xlabel(r'$P$')
    plt.legend(loc='upper right')
    # bbox_to_anchor = (0., 1.02, 1., .102),
    # plt.savefig(directory  + 'paper R_D.pdf')
    plt.savefig('Corollary3Case2DP.pdf')
    plt.savefig('Corollary3Case2DP.eps',format='eps')
    plt.show()


# P=0
def plotC3C3():
    # R_c = 0, R_c = 1, R_c =\infty
    Rc0_list = []  # d0
    Rc1_list = []  # d1
    Rci_list = []  # di
    R_list = []
    start = 0.0
    stop = 2
    num = 1000
    P = 0
    # Rc=math.inf
    for r in np.linspace(start, stop, num):
        # RRcPD3(R,Rc,P,sigmaX=1)
        R_list.append(r)
        Rc0_list.append(RRcPD3(r, Rc=0, P=0))
        # Rcr1=R_cr1(d)
        Rc1_list.append(RRcPD3(r, Rc=1, P=0))
        Rci_list.append(RRcPD3(r, Rc=math.inf, P=0))
    label0 = r'$R_{\mathrm{c}}=0$'
    label1 = r'$R_{\mathrm{c}}=1$'
    label2 = r'$R_{\mathrm{c}}=\infty$'

    line3 = plt.plot(R_list, Rci_list, 'sienna', label=label2, linewidth=2.0)
    line2 = plt.plot(R_list, Rc1_list, 'r', label=label1, linewidth=2.0)
    line1 = plt.plot(R_list, Rc0_list, 'b', label=label0, linewidth=2.0)

    # line1 = plt.plot(R_list, Rc0_list, 'r', label=label0, linewidth=2.0)
    # line2 = plt.plot(R_list, Rc1_list, 'b', label=label1, linewidth=2.0)
    # line3 = plt.plot(R_list, R_list, 'sienna', label=label2, linewidth=2.0)
    # plt.xlabel(r'$D$')
    # plt.ylabel(r'$R$')
    # plt.rc('text', usetex=True)
    plt.ylabel(r'$D$')
    plt.xlabel(r'$R$')
    plt.legend(loc='upper right')
    # bbox_to_anchor = (0., 1.02, 1., .102),
    # plt.savefig(directory  + 'paper R_D.pdf')
    plt.savefig('Corollary3Case3DR_rc1.pdf')
    plt.savefig('Corollary3Case3DR_rc1.eps',format='eps')
    plt.show()


# #sigmaXH=sigmaP
#     sigmaRRcP=sigmaRRcP(R,Rc,P,sigmaX)
#     epsilon1 =epsilon(R, Rc)
#     D =  math.pow(sigmaX,2)+ math.pow(sigmaXH,2)-2*sigmaX*sigmaXH*epsilon1
#     # D = math.pow((muX - muXH),2) + math.pow(sigmaX,2)+ math.pow(sigmaXH,2)-2*sigmaX*sigmaXH*epsilon
#     return D
def f1(x):
    return math.exp(-(x ** 2) / 2)


def T4DR(a):#
    # a>=0
    # result = integrate.quad(lambda x: special.jv(2.5,x), 0, 4.5)
    I, err = integrate.quad(f1, (-math.inf), a)
    # y=f1(a)
    # I =integrate.simpson(y, a)
    Qa = 1 / (math.sqrt(2 * math.pi)) * I
    D = 1 - math.exp(-(a ** 2)) / (2 * math.pi * Qa * (1 - Qa))
    R = Qa * math.log(1 / Qa) + (1 - Qa) * math.log(1 / (1 - Qa))
    return R, D



def baseline(R):
    D = 2 * math.exp(-2 * R) - math.exp(-4 * R)
    return D

def baseline2(R):
    sigmaX=1
    a= (1- math.exp(-2 * R))
    b=math.pow(a,2)
    D = math.pow(sigmaX,2)-math.pow(sigmaX,2)* b
    return D

def plotT4DR():
    # underlineD and baselineD
    Rc0_list = []  # d0
    Rc1_list = []  # d1
    Rci_list = []  # di
    R_list = []
    start = 0.0
    stop = 5
    stopR=math.log(2)
    num = 100
    for a in np.linspace(start, stop, num):
        R, D = T4DR(a)
        print(f'R====={R}')
        if R<=stopR:
            pass

        R_list.append(R)  # P list
        Rc0_list.append(D)
        # Rcr1=R_cr1(d)s
        Rc1_list.append(baseline(R))
        # Rci_list.append(C2_DRRcP2(math.inf,sigmaXH))
    # label0=r'$P_{\mathrm{c}}=0$'
    # label1 = r'$P_{\mathrm{c}}=1$'
    # label2=r'$P_{\mathrm{c}}=2$'
    label0 = r'$\overline{D}_s$'
    label1 = r'$\overline{D}$'
    # label2=r'$R=\infty$'
    # line3 = plt.plot(R_list,Rci_list, 'sienna', label=label2, linewidth=2.0)
    line2 = plt.plot(R_list, Rc1_list, 'r', label=label1, linewidth=2.0)
    line1 = plt.plot(R_list, Rc0_list, 'b', label=label0, linewidth=2.0)

    plt.ylabel(r'$D$')
    plt.xlabel(r'$R$')
    plt.legend(loc='upper right')
    plt.savefig('T4DR1-log2.pdf')
    plt.show()
def plotPart2T8DR():
    # underlineD and baselineD
    Rc0_list = []  # d0
    Rc1_list = []  # d1
    Rc2_list = []  # di
    R_list = []
    start = 0.0
    sigmaX=1
    stopR = math.log(2)
    num = 100
    stop=5
    for a in np.linspace(start, stop, num):
        R, D = T4DR(a)

        R_list.append(R)  # P list
        Rc0_list.append(D)
        # Rcr1=R_cr1(d)s
        Rc1_list.append(baseline2(R))
        Rc2_list.append(overlineDinfty(R, sigmaX))
    # label0=r'$P_{\mathrm{c}}=0$'
    # label1 = r'$P_{\mathrm{c}}=1$'
    label2=r'$\underline{D}(R,R_c,\infty)$'
    label1 = r'$\overline{D}(R,R_c,\infty)$'
    label0 = r'$\overline{D}_e(R,R_c)$'
    # label2=r'$R=\infty$'
    line1 = plt.plot(R_list, Rc1_list, 'b',  linestyle='--',label=label1, linewidth=2.0)
    line0 = plt.plot(R_list, Rc0_list, 'sienna',linestyle='-', label=label0, linewidth=2.0)
    line2= plt.plot(R_list,Rc2_list, 'r', linestyle=':',label=label2, linewidth=2.0)



    plt.ylabel(r'$D$')
    plt.xlabel(r'$R$')
    plt.legend(loc='upper right')
    plt.savefig('plotPart2T8DR-log2-3lines.pdf')
    plt.show()


def plotT4case2DR():
    # underlineD_hat and overlineD_hat
    P = 1
    sigmaXH = sigmaP(P)
    sigmaXH = 0.2290613399456898
    Rc = 1
    sigmaX = 1
    # R_c = 0, R_c = 1, R_c =\infty
    Rc0_list = []  # d0
    Rc1_list = []  # d1
    Rci_list = []  # di
    R_list = []
    start = 0
    stop = 2
    num = 100
    # overlineD(R,Rc,P,sigmaX,sigmaXH)  rc=1 P=1 ,sigmaX=1,sigmaXH=
    for r in np.linspace(start, stop, num):
        R_list.append(r)  #

        Rc0_list.append(underlineD_hat(r, Rc, P, sigmaX, sigmaXH))

        Rc1_list.append(overlineD_hat(r, Rc, P, sigmaX, sigmaXH))

    label0 = r'$\underline{D}^{\prime}$'
    label1 = r'$\overline{D}^{\prime}$'
    # label2=r'$R=\infty$'
    # line3 = plt.plot(R_list,Rci_list, 'sienna', label=label2, linewidth=2.0)
    line2 = plt.plot(R_list, Rc1_list, 'r', label=label1, linewidth=2.0)
    line1 = plt.plot(R_list, Rc0_list, 'b', label=label0, linewidth=2.0)


    plt.ylabel(r'$D$')
    plt.xlabel(r'$R$')

    plt.legend(loc='upper right')
    plt.savefig('theroem4Case2RD_fig9.eps',format='eps')
    plt.savefig('theroem4Case2RD_fig9.pdf')
    plt.show()
    print("sigmaXH====", sigmaXH)
def plotT4case2DP():
    # underlineD_hat and overlineD_hat
    R= 1

    # sigmaXH = 0.2290613399456898
    Rc = 1
    sigmaX = 1
    # R_c = 0, R_c = 1, R_c =\infty
    Rc0_list = []  # d0
    Rc1_list = []  # d1
    Rci_list = []  # di
    R_list = []
    start = 0
    stop = 0.02
    num = 100
    # overlineD(R,Rc,P,sigmaX,sigmaXH)  rc=1 P=1 ,sigmaX=1,sigmaXH=
    for P in np.linspace(start, stop, num):
        sigmaXH = sigmaP(P)
        R_list.append(P)  #

        Rc0_list.append(underlineD_hat(R, Rc, P, sigmaX, sigmaXH))

        Rc1_list.append(overlineD_hat(R, Rc, P, sigmaX, sigmaXH))
    label1 = r'$\overline{D}^{\prime}$'
    label0 = r'$\underline{D}^{\prime}$'

    # label2=r'$R=\infty$'
    # line3 = plt.plot(R_list,Rci_list, 'sienna', label=label2, linewidth=2.0)
    line2 = plt.plot(R_list, Rc1_list, 'r', label=label1, linewidth=2.0)
    line1 = plt.plot(R_list, Rc0_list, 'b', label=label0, linewidth=2.0)


    plt.ylabel(r'$D$')
    plt.xlabel(r'$P$')

    plt.legend(loc='upper right')
    plt.savefig('theroem4Case2PD_fig9_2.eps',format='eps')
    plt.savefig('theroem4Case2PD_fig9_2.pdf')
    plt.show()
    print("sigmaXH====", sigmaXH)

def consD(args):
    min, max = args
    cons = ({'type': 'ineq', 'fun': lambda x: x -min}, \
            {'type': 'ineq', 'fun': lambda x: -x + max})
    return cons
def consG(args):
    min, max = args
    cons = ({'type': 'ineq', 'fun': lambda x: -x +min}, \
            {'type': 'ineq', 'fun': lambda x: x - max})
    return cons
def cons_beta_alpha(X):
    cons = ({'type': 'ineq', 'fun': lambda x: -x +min}, \
            {'type': 'ineq', 'fun': lambda x: x - max})
    return cons

def a(beta,args):
    R,Rc=args
    print(f'beta={beta}')
    # beta = np.clip(beta, 0, 1)
    b=np.clip((math.exp(-(R + Rc)) - math.sqrt(1 - beta ** 2)),0,None)
    a=b/ (beta ** 2)
    return a
def minDstar(sigmaXH,args):
    sigmaX, R,mina= args
    minid=math.pow(sigmaX, 2) + math.pow(sigmaXH, 2) - 2 * sigmaX *sigmaXH* math.sqrt(
        (1 - math.exp(-2 * R)) * (1-mina))
    return minid
def min_underDstar(R, Rc, P, sigmaX):
    from scipy.optimize import minimize

    low_conD = np.clip(sigmaX - math.sqrt(P), 0, None)
    up_conD = sigmaX

    argsD=R, Rc, P, sigmaX

    argsD=(low_conD ,up_conD)
    cons_D = consD(argsD)
    mina_list=[]

    low_cona=(math.pow(sigmaX, 2) + math.pow(low_conD, 2)-P)/(2*sigmaX*low_conD)
    up_cona = 1

    argsA = (low_cona, up_cona)
    cons_A=consD(argsA)
    a_args=(R,Rc)
    if low_cona ==up_cona:
        mina=a(low_cona,a_args)
    else:
        cons_A = [(low_cona, up_cona)]
        mina=minimize(a,low_cona,args=(a_args,),method='SLSQP',options={'xatol': 1e-8, 'disp': True}, constraints=cons_A).fun
        # mina_list.append(mina)
    # mina=min(mina_list)
    # mina = minimize(a, low_cona,args=(a_args,), method='trust-constr', bounds=boundsA)
    d_args=(sigmaX,R,mina)
    res = minimize(minDstar, low_conD,args=(d_args,), method='SLSQP',
                   options={'xatol': 1e-8, 'disp': True}, constraints=cons_D)
    print(f"low and up  SIGMAXH== {low_conD},{up_conD}")
    print(f'mini a==={mina}')
    print(f'**when sigmax=***{res.x[0]},*miniD****{res.fun}')

    return res.fun
def minDstar2(sigmaXH,args):
    sigmaX,R, ming = args

    d2=math.pow(sigmaX, 2) + math.pow(sigmaXH, 2) - 2 * sigmaX * math.sqrt(
                (1 - math.exp(-2* R))) * math.sqrt(np.clip(math.pow(sigmaXH, 2) - ming,0,None))
    return d2
def Ga(beta,args):
    sigmaX,sigmaXH,alpha,R,Rc=args
    bb = sigmaX * math.exp(-(R + Rc))
    aa = math.sqrt(np.clip(math.pow(sigmaX, 2) - 2 * alpha * beta * sigmaX * sigmaXH + math.pow(alpha, 2) * math.pow(sigmaXH, 2), 0, None))
    cc = math.pow(np.clip((bb - aa), 0, None), 2)
    if cc <= 1e-10:
        cc = 0
    ga = np.clip(cc, 0, None) / math.pow(alpha, 2)
    return ga
def min_underDstar2(R, Rc, P, sigmaX, sigmaX_H):
    from scipy.optimize import minimize

    argsD=R, Rc, P, sigmaX
    # args = (R, Rc, P, sigmaX)
    low_conD= np.clip(sigmaX-math.sqrt(P),0,None)
    up_conD = sigmaX
    argsD=(low_conD ,up_conD)
    cons_D = consD(argsD)

    # argsa = (R, Rc, P, sigmaX)
    low_cona= (math.pow(sigmaX, 2) + math.pow(low_conD, 2)-P)/(2*sigmaX*low_conD)
    up_cona = 1
    # beta=low_cona
    if low_cona==0:
        low_cona=0.00001
    argsA = (low_cona, up_cona)
    cons_A=consD(argsA)

    a_args=(R,Rc)
    minG=minimize(a,1,args=(a_args,),method='SLSQP',options={'xatol': 1e-8, 'disp': True}, constraints=cons_A)
    argsg = (low_cona, up_cona)
    cons_g=consD(argsg)

    g_args=(R,Rc)
    minG=minimize(g,1,args=(g_args,),method='SLSQP',options={'xatol': 1e-8, 'disp': True}, constraints=cons_g)
    x0 = 0

    d_args=(sigmaX,R,minG.fun)
    res = minimize(minDstar2, x0,args=d_args, method='SLSQP',
                   options={'xatol': 1e-8, 'disp': True}, constraints=cons_D)
    print(f"low and up {low_conD},{up_conD}")
    print(f'*****{res.x[0]}')
    print(f'*****{res.fun}')

    return res.fun






    for i in np.linspace(low_conD, up_conD, numd): #g
        #(\sigma ^ 2_X+\sigma ^ 2_{\hat{X}}-P} / (2\sigma_X\sigma_{\hat{X}})
        aa=math.sqrt(1 - math.exp(-2*(R + Rc)))
        bb=(math.pow(sigmaX, 2) + math.pow(i, 2) - P) / (2 * sigmaX * i)
        low_beta = max(aa,bb)
        b_list=[]
        # if low_beta == up_beta:
        #     numb=1
        # low_beta=0
        numb=num(low_beta, up_beta)
        for j in np.linspace(low_beta, up_beta, numb):# \beta always equals to low_bound
            # print(f'j==={j},r==={R},rC===={Rc}')
            ###DEBUG
            # j=bb
            # low_alpha = j * sigmaX - sigmaX * math.sqrt(j ** 2 - 1 + math.exp(-2 * (R + Rc)))

            low_alpha=j*sigmaX-sigmaX*math.sqrt(np.clip(j**2-1+math.exp(-2*(R+Rc)),0,None))
            low_alpha=low_alpha/i
            up_alpha= j*sigmaX/i
            numa=num(low_alpha,up_alpha)
            if low_alpha==0:
                low_alpha=0.00000001
            a_list=[]
            for k in np.linspace(0.1, 1, numa):  # a
            # for k in np.linspace(low_alpha, up_alpha, numa): # a
                # k=up_alpha
                # k=low_beta
                # k=1

                bb=sigmaX * math.exp(-(R + Rc))
                aa=math.sqrt(np.clip(math.pow(sigmaX, 2) - 2 * k * j * sigmaX * i + math.pow(k, 2) * math.pow(i, 2),0,None))
                cc=math.pow(np.clip((bb-  aa),0,None),2)
                if cc<=1e-10:
                    cc=0
                ga=np.clip(cc,0,None)/ math.pow(k, 2)
                a_list.append(ga)

            b_list.append(max(a_list))
        ming=min(b_list)
        dg=math.pow(sigmaX, 2) + math.pow(i, 2) - 2 * sigmaX * math.sqrt(
                (1 - math.exp(-2* R))) * math.sqrt(np.clip(math.pow(i, 2) - ming,0,None))
        D_list.append(dg)
    D_min=min(D_list)
    print(f'Dstar2===={D_min}')
    return D_min

def G(x_set,args):
    R, Rc, sigmaX_H, sigmaX=args
    alpha,beta=x_set
    G=np.clip(sigmaX * math.exp(-(R + Rc)) - math.sqrt(
           math.pow(sigmaX, 2) - 2 * alpha * beta * sigmaX * sigmaX_H + math.pow(alpha * sigmaX, 2)), 0,
               None) / math.pow(alpha, 2)
    return G


def min_underDstarfor(R, Rc, P, sigmaX):

    low_conD = np.clip(sigmaX - np.sqrt(P), 0, None)
    up_conD = sigmaX
    up_beta=1
    d_list=[]

    numd= num(low_conD,up_conD)
    for i in np.linspace(low_conD, up_conD, numd): #sigmaxH
        low_beta = np.clip((sigmaX ** 2 + i ** 2 - P) / (2 * sigmaX * i), -1, 1)  # Ensure domain validity for sqrt
        # low_beta = (np.pow(sigmaX, 2) + np.pow(i, 2) - P) / (2 * sigmaX * i)
        # low_beta=0.001
        numb =num(low_beta, up_beta)
        a_l = []

        for j in np.linspace(low_beta, up_beta, numb):
            ppp = np.power(np.exp(-(R + Rc)) - np.sqrt(np.clip(1 - j ** 2, 0, None)), 2)
            a = np.clip(ppp, 0, None) / (j ** 2 if j != 0 else 1)
            a_l.append(a)
        mina=min(a_l)

        AA = np.sqrt(np.clip((1 - np.exp(-2 * R)) * (1 - mina), 0, None))
        d = sigmaX**2 + i**2 - 2 * sigmaX * i * AA
        d_list.append(d)
    dmin=min(d_list)
    print(f'MINAA===={mina},Dstar===={dmin}')
    return dmin

def min_underDstar2for(R, Rc, P, sigmaX):

    low_conD = np.clip(sigmaX - math.sqrt(P), 0, None)
    up_conD = sigmaX
    up_beta=1

    D_list=[]
    numd= num(low_conD,up_conD)
    for i in np.linspace(low_conD, up_conD, numd): #g
        #(\sigma ^ 2_X+\sigma ^ 2_{\hat{X}}-P} / (2\sigma_X\sigma_{\hat{X}})
        aa=math.sqrt(1 - math.exp(-2*(R + Rc)))
        bb=(math.pow(sigmaX, 2) + math.pow(i, 2) - P) / (2 * sigmaX * i)
        low_beta = max(aa,bb)
        b_list=[]

        numb=num(low_beta, up_beta)
        for j in np.linspace(low_beta, up_beta, numb):# \beta always equals to low_bound

            low_alpha=j*sigmaX-sigmaX*math.sqrt(np.clip(j**2-1+math.exp(-2*(R+Rc)),0,None))
            low_alpha=low_alpha/i
            up_alpha= j*sigmaX/i
            numa=num(low_alpha,up_alpha)
            if low_alpha==0:
                low_alpha=0.00000001
            a_list=[]
            k_list=[]
            for k in np.linspace(low_alpha, up_alpha, numa): # a
                # k=1
                k_list.append(k)
                bb=sigmaX * math.exp(-(R + Rc))
                aa=math.sqrt(np.clip(math.pow(sigmaX, 2) - 2 * k * j * sigmaX * i + math.pow(k, 2) * math.pow(i, 2),0,None))
                cc=math.pow(np.clip((bb-  aa),0,None),2)
                if cc<=1e-10:
                    cc=0
                ga=np.clip(cc,0,None)/ math.pow(k, 2)
                a_list.append(ga)
            index_max=np.argmax(a_list)
            # print(f'aaaa======={k_list[index_max]}')
            b_list.append(max(a_list))
        ming=min(b_list)
        dg=math.pow(sigmaX, 2) + math.pow(i, 2) - 2 * sigmaX * math.sqrt(
                (1 - math.exp(-2* R))) * math.sqrt(np.clip(math.pow(i, 2) - ming,0,None))
        D_list.append(np.float64(dg))
    D_min=min(D_list)
    # print(f'Dstar2===={D_min}')
    return D_min

# def min_underDstar2forA(R, Rc, P, sigmaX,low_alpha,up_alpha): #$\underline{D}(R,R_c,P|W^2_2)$ T6
#
#     low_conD = np.clip(sigmaX - math.sqrt(P), 0, None)
#     up_conD = sigmaX
#     low_alpha=0
#     up_alpha=1000
#     D_list=[]
#     numa=num(low_alpha,up_alpha)
#     numd= num(low_conD,low_conD)
#     for i in np.linspace(low_conD, up_conD, numd): #g
#             a_list=[]
#             k_list=[]
#             for k in np.linspace(low_alpha, up_alpha, numa): # a
#                 # k=1
#                 k_list.append(k)
#                 bb=sigmaX * math.exp(-(R + Rc))
#                 aa=math.sqrt(np.clip(math.pow(sigmaX, 2) - k * np.clip((sigmaX**2+  i**2-P),0,None)+ math.pow(k, 2) * math.pow(i, 2),0,None))
#                 cc=math.pow(np.clip((bb-  aa),0,None),2)
#                 if cc<=1e-10:
#                     cc=0
#                 ga=np.clip(cc,0,None)/ math.pow(k, 2)
#                 a_list.append(ga)
#             index_max=np.argmax(a_list)
#             print(f'aaaa======={k_list[index_max]}')
#             b_list.append(max(a_list))
#         ming=min(b_list)
#         dg=math.pow(sigmaX, 2) + math.pow(i, 2) - 2 * sigmaX * math.sqrt(
#                 (1 - math.exp(-2* R))) * math.sqrt(np.clip(math.pow(i, 2) - ming,0,None))
#         D_list.append(dg)
#     D_min=min(D_list)
#     print(f'Dstar2===={D_min}')
#     return D_min
def min_underDstar2forA(R, Rc, P, sigmaX):#$\underline{D}^{\prime}(R,R_c,P|W^2_2)$
    # return a_value
    alist_value=[]
    low_conD = np.clip(sigmaX - math.sqrt(P), 0, None)
    up_conD = sigmaX
    up_beta=1  #sigmax_H

    D_list=[]
    optimal_a=[]
    numd= num(low_conD,up_conD)
    for i in np.linspace(low_conD, up_conD, numd): #g
        #(\sigma ^ 2_X+\sigma ^ 2_{\hat{X}}-P} / (2\sigma_X\sigma_{\hat{X}})
        aa=math.sqrt(1 - math.exp(-2*(R + Rc)))
        bb=(math.pow(sigmaX, 2) + math.pow(i, 2) - P) / (2 * sigmaX * i)
        low_beta = max(aa,bb)
        b_list=[]

        numb=num(up_conD, up_conD)
        for j in np.linspace(low_beta, up_beta, numb):# \beta always equals to low_bound

            low_alpha=j*sigmaX-sigmaX*math.sqrt(np.clip(j**2-1+math.exp(-2*(R+Rc)),0,None))
            low_alpha=low_alpha/i
            up_alpha= j*sigmaX/i
            # print(f'up_alpha======{up_alpha}\n')
            # up_alpha=1000

            #a=\sup
            # low_alpha=up_alpha
            numa=num(low_alpha,up_alpha)
            if low_alpha==0:
                low_alpha=0.00000001
            a_value=[]
            a_list=[]
            k_list=[]
            for k in np.linspace(low_alpha, up_alpha, numa): # a
                # k=up_alpha
                k_list.append(k)
                bb=sigmaX * math.exp(-(R + Rc))
                aa = math.sqrt(np.clip(
                    math.pow(sigmaX, 2) - k * np.clip((sigmaX ** 2 + i ** 2 - P), 0, None) + math.pow(k, 2) * math.pow(
                        i, 2), 0, None))
                # aa=math.pow(sigmaX, 2)-k*np.clip((sigmaX**2+i**2-P),0,None)+(k**2)*(i**2)
                # # aa=math.sqrt(np.clip(math.pow(sigmaX, 2) - 2 * k * j * sigmaX * i + math.pow(k, 2) * math.pow(i, 2),0,None))
                # cc=math.pow(np.clip((bb-  math.sqrt(aa)),0,None),2)
                # aa=math.pow(sigmaX, 2)-k*np.clip((sigmaX**2+i**2-P),0,None)+(k**2)*(i**2)
                # aa=math.sqrt(np.clip(math.pow(sigmaX, 2) - 2 * k * j * sigmaX * i + math.pow(k, 2) * math.pow(i, 2),0,None))
                cc=math.pow(np.clip((bb-  aa),0,None),2)
                if cc<=1e-10:
                    cc=0
                ga=np.clip(cc,0,None)/ math.pow(k, 2)
                a_list.append(ga)
                a_value.append(k)
            index_max=np.argmax(a_list)
            alist_value.append(a_value[index_max])
            # print(f'aaaa=={k}=====index={a_value[index_max]}\n')
            # optimal_a.append(a_list[index_max])
            b_list.append(max(a_list))
        ming=min(b_list)
        dg=math.pow(sigmaX, 2) + math.pow(i, 2) - 2 * sigmaX * math.sqrt(
                (1 - math.exp(-2* R))) * math.sqrt(np.clip(math.pow(i, 2) - ming,0,None))
        D_list.append(dg)


    D_min=min(D_list)
    index_dmin = np.argmin(D_list)
    print(f'a_ index={index_dmin},D_min={D_min}')

    return D_min
    # return alist_value[index_dmin]


import numpy as np

import math


def calculate_alpha_hat_plot(sigma_X, sigma_X_hat, P, R, Rc):
    denominator = (sigma_X ** 2 + sigma_X_hat ** 2 - P) ** 2 - 4 * sigma_X ** 2 * sigma_X_hat ** 2 * math.exp(
        -2 * (R + Rc))

    if denominator == 0:  # 130
        print(f'13000000======r==={R}')
        alpha_hat = sigma_X ** 2 / (sigma_X ** 2 + sigma_X_hat ** 2 - P)
        return alpha_hat


    else:  # 131
        print(f'131======r==={R}')
        numerator = 2 * sigma_X ** 2 * (sigma_X ** 2 + sigma_X_hat ** 2 - P) * (1 - math.exp(-2 * (R + Rc)))

        sqrt_term = math.sqrt((4 * sigma_X ** 2 * sigma_X_hat ** 2 - (sigma_X ** 2 + sigma_X_hat ** 2 - P) ** 2) * (
                1 - math.exp(-2 * (R + Rc))))
        alpha_hat = (numerator / denominator) - (sqrt_term / denominator) * 2 * sigma_X ** 2 * math.exp(-(R + Rc))



        return R,alpha_hat

def calculate_alpha_hat(sigma_X, sigma_X_hat, P, R, Rc):
    denominator = (sigma_X ** 2 + sigma_X_hat ** 2 - P) ** 2 - 4 * (sigma_X ** 2 )* (sigma_X_hat ** 2 )* math.exp(
        -2 * (R + Rc))
    Rlist = []
    alpha_hat_list = []
    if denominator == 0: #130
        # print(f'13000000======r==={R}')
        alpha_hat = (sigma_X ** 2) / (sigma_X ** 2 + sigma_X_hat ** 2 - P)
        return alpha_hat


    else:#131
        # print(f'131======r==={R}')
        numerator = 2 * (sigma_X ** 2 )* (sigma_X ** 2 + sigma_X_hat ** 2 - P) * (1 - math.exp(-2 * (R + Rc)))
        aa=4 * (sigma_X ** 2) * (sigma_X_hat ** 2) - (sigma_X ** 2 + sigma_X_hat ** 2 - P) ** 2
        aa=aa* (1 - math.exp(-2 * (R + Rc)))
        aa=np.clip(aa,0,None)
        # print(f'ssssss={aa}')
        sqrt_term = math.sqrt(aa)
        alpha_hat = (numerator / denominator) - (sqrt_term / denominator) * 2 * (sigma_X ** 2) * math.exp(-(R + Rc))

        return alpha_hat


    # print("Alpha Hat:", alpha_hat)
def calculate_delta_plus(R, Rc, P, alpha, sigmaX,sigmaXH):
    """
    Calculates the value of δ⁺(σₓ, α) as per the equation (33) in the image.

    Parameters:
    sigma_x (float): The value of σₓ.
    alpha (float): The value of α.

    Returns:
    float: The calculated value of δ⁺(σₓ, α).
    """
    # if alpha==0:
    #     alpha=0.000000000001
    aa=sigmaX ** 2 - alpha * (sigmaX ** 2 + sigmaXH ** 2 - P) + alpha ** 2 * sigmaXH ** 2
    aa=np.clip(aa,0,None)
    term1 = (sigmaX * np.exp(-(R + Rc))) - np.sqrt(aa
        )
    term1=np.clip(term1,0,None)
    bb=term1 / alpha
    # print(f'term1 / alpha====={aa}')
    return bb
def min_underDstar2forA_Sup_hat(R, Rc, P, sigmaX):
    """
    Calculates the value of D'(R, Rc, P, W2) as per the equations in the image.

    Parameters:
    R (float): The first parameter.
    Rc (float): The second parameter.
    P (float): The third parameter.
    W2 (float): The fourth parameter.

    Returns:
    float: The calculated value of D'(R, Rc, P, W2).
    """
    # sigmaXH = lambda sigmaX: np.maximum(sigmaX - np.sqrt(P), 0)  # constraint sigmaX
    low_conD=np.clip(sigmaX - np.sqrt(P), 0,None)
    D_list = []
    sigmaXHlist=[]
    Alphalist=[]
    for sigmaXH in np.linspace(low_conD, sigmaX, 2000):

        thereshold= sigmaX ** 2+sigmaXH ** 2-2*sigmaX*sigmaXH* np.sqrt(1 - np.exp(-2 * (R+Rc)))
        if P>=thereshold: # equation 121
            delta_plus=0
        else: #equation 122
            alpha = calculate_alpha_hat(sigmaX, sigmaXH, P, R, Rc)
            # rlist,alpha_hat_list=calculate_alpha_hat_plot(sigmaX, sigmaXH, P, R, Rc)
            # sigmaXHlist.append(sigmaXH)
            # Alphalist.append(alpha)
            alpha=np.clip(alpha,0,None) #alpha>0
            # print(f'alpha======{alpha}')
            delta_plus=calculate_delta_plus(R, Rc, P, alpha, sigmaX,sigmaXH)
            # print(f'delta_plus======{delta_plus}')


        minD=sigmaX**2+sigmaXH**2 -2 * sigmaX * np.sqrt((1 - np.exp(-2 * R)) * (sigmaXH ** 2 - delta_plus ** 2))
        # print(f'minD======{minD}')
        D_list.append(minD)
    # plt.plot(sigmaXHlist,Alphalist,'r', linestyle=':', label="alpha", linewidth=2.0)
    # # line3 = plt.plot(R_list, Rc3_list, 'r', linestyle=':', label=label3, linewidth=2.0)
    # # line2 = plt.plot(R_list, Rc1_list, 'r', label=label1, linewidth=2.0)
    #
    # plt.xlabel(r'$sigmaxh$')
    # plt.ylabel(r'$alpha$')
    #
    # plt.legend(loc='upper right')
    # # Adjust the subplot layout
    # plt.gcf().subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.3)
    # # plt.savefig('theroem6RD_prime_P0.1Rc0.1.pdf')
    # plt.show()

    return min(D_list)



def min_underDstar2forA_Sup(R, Rc, P, sigmaX):#$\underline{D}^{\prime}(R,R_c,P|W^2_2)$
    # return a_value
    alist_value=[]
    low_conD = np.clip(sigmaX - math.sqrt(P), 0, None)
    up_conD = sigmaX
    up_beta=1  #sigmax_H

    D_list=[]
    optimal_a=[]
    numd= num(low_conD,up_conD)
    for i in np.linspace(low_conD, up_conD, numd): #g
        #(\sigma ^ 2_X+\sigma ^ 2_{\hat{X}}-P} / (2\sigma_X\sigma_{\hat{X}})
        aa=math.sqrt(1 - math.exp(-2*(R + Rc)))
        bb=(math.pow(sigmaX, 2) + math.pow(i, 2) - P) / (2 * sigmaX * i)
        low_beta = max(aa,bb)
        b_list=[]

        # numb=100
        numb=num(low_beta, up_beta)
        for j in np.linspace(low_beta, up_beta, numb):# \beta always equals to low_bound

            low_alpha=j*sigmaX-sigmaX*math.sqrt(np.clip(j**2-1+math.exp(-2*(R+Rc)),0,None))
            low_alpha=low_alpha/i
            up_alpha= j*sigmaX/i
            # print(f'up_alpha======{up_alpha}\n')
            # up_alpha=100

            #a=\sup
            # low_alpha=up_alpha
            numa=num(low_alpha,up_alpha)
            # numa=10000
            if low_alpha==0:
                low_alpha=0.00000001
            a_value=[]
            a_list=[]
            k_list=[]
            for k in np.linspace(low_alpha, up_alpha, numa): # a
                # k=up_alpha
                k_list.append(k)
                bb=sigmaX * math.exp(-(R + Rc))
                aa = math.sqrt(np.clip(
                    math.pow(sigmaX, 2) - k * np.clip((sigmaX ** 2 + i ** 2 - P), 0, None) + math.pow(k, 2) * math.pow(
                        i, 2), 0, None))
                # aa=math.pow(sigmaX, 2)-k*np.clip((sigmaX**2+i**2-P),0,None)+(k**2)*(i**2)
                # # aa=math.sqrt(np.clip(math.pow(sigmaX, 2) - 2 * k * j * sigmaX * i + math.pow(k, 2) * math.pow(i, 2),0,None))
                # cc=math.pow(np.clip((bb-  math.sqrt(aa)),0,None),2)
                # aa=math.pow(sigmaX, 2)-k*np.clip((sigmaX**2+i**2-P),0,None)+(k**2)*(i**2)
                # aa=math.sqrt(np.clip(math.pow(sigmaX, 2) - 2 * k * j * sigmaX * i + math.pow(k, 2) * math.pow(i, 2),0,None))
                cc=math.pow(np.clip((bb-  aa),0,None),2)
                if cc<=1e-10:
                    cc=0
                ga=np.clip(cc,0,None)/ math.pow(k, 2)
                a_list.append(ga)
                a_value.append(k)
            index_max=np.argmax(a_list)
            alist_value.append(a_value[index_max])
            # print(f'asup=={k}=====index={a_value[index_max]}\n')
            # optimal_a.append(a_list[index_max])
            b_list.append(max(a_list))
        ming=min(b_list)
        dg=math.pow(sigmaX, 2) + math.pow(i, 2) - 2 * sigmaX * math.sqrt(
                (1 - math.exp(-2* R))) * math.sqrt(np.clip(math.pow(i, 2) - ming,0,None))
        D_list.append(np.float64(dg))


    D_min=min(D_list)
    index_dmin = np.argmin(D_list)
    # print(f'index_dmin={index_dmin},D_min={D_min}')

    return D_min
    # return alist_value[index_dmin]

# def min_underDstar2forA_Sup(R, Rc, P, sigmaX):#$\underline{D}^{\prime}(R,R_c,P|W^2_2)$
#     # return a_value
#     alist_value=[]
#     low_conD = np.clip(sigmaX - math.sqrt(P), 0, None)
#     # up_conD = sigmaX
#     up_conD=10000
#     up_beta=1  #sigmax_H
#
#     D_list=[]
#     optimal_a=[]
#     numd= num(low_conD,up_conD)
#     for i in np.linspace(low_conD, up_conD, numd): #g
#         #(\sigma ^ 2_X+\sigma ^ 2_{\hat{X}}-P} / (2\sigma_X\sigma_{\hat{X}})
#         aa=math.sqrt(1 - math.exp(-2*(R + Rc)))
#         bb=(math.pow(sigmaX, 2) + math.pow(i, 2) - P) / (2 * sigmaX * i)
#         low_beta = max(aa,bb)
#         b_list=[]
#
#         numb=num(up_conD, up_conD)
#         for j in np.linspace(low_beta, up_beta, numb):# \beta always equals to low_bound
#
#             low_alpha=j*sigmaX-sigmaX*math.sqrt(np.clip(j**2-1+math.exp(-2*(R+Rc)),0,None))
#             low_alpha=low_alpha/i
#             up_alpha= j*sigmaX/i
#             # print(f'up_alpha======{up_alpha}\n')
#             # up_alpha=1000
#
#             #a=\sup
#             # low_alpha=up_alpha
#             numa=num(low_alpha,up_alpha)
#             if low_alpha==0:
#                 low_alpha=0.00000001
#             a_value=[]
#             a_list=[]
#             k_list=[]
#             for k in np.linspace(low_alpha, up_alpha, numa): # a
#                 # k=1
#                 k_list.append(k)
#                 bb=sigmaX * math.exp(-(R + Rc))
#                 # aa=math.pow(sigmaX, 2)-k*np.clip((sigmaX**2+i**2-P),0,None)+(k**2)*(i**2)
#                 # # aa=math.sqrt(np.clip(math.pow(sigmaX, 2) - 2 * k * j * sigmaX * i + math.pow(k, 2) * math.pow(i, 2),0,None))
#                 # cc=math.pow(np.clip((bb-  math.sqrt(aa)),0,None),2)
#
#                 aa=math.pow(sigmaX, 2)-k*np.clip((sigmaX**2+i**2-P),0,None)+(k**2)*(i**2)
#                 # aa=math.sqrt(np.clip(math.pow(sigmaX, 2) - 2 * k * j * sigmaX * i + math.pow(k, 2) * math.pow(i, 2),0,None))
#                 cc=math.pow(np.clip((bb-  math.sqrt(aa)),0,None),2)
#                 if cc<=1e-10:
#                     cc=0
#                 ga=np.clip(cc,0,None)/ math.pow(k, 2)
#                 a_list.append(ga)
#                 a_value.append(k)
#             index_max=np.argmax(a_list)
#             alist_value.append(a_value[index_max])
#             print(f'aaaa=={k}=====index={a_value[index_max]}\n')
#             # optimal_a.append(a_list[index_max])
#             b_list.append(max(a_list))
#         ming=min(b_list)
#         dg=math.pow(sigmaX, 2) + math.pow(i, 2) - 2 * sigmaX * math.sqrt(
#                 (1 - math.exp(-2* R))) * math.sqrt(np.clip(math.pow(i, 2) - ming,0,None))
#         D_list.append(dg)
#
#
#     D_min=min(D_list)
#     index_dmin = np.argmin(D_list)
#
#     return D_min
#     # return alist_value[index_dmin]

# def min_underDstar2(R, Rc, P, sigmaX, sigmaX_H):
#     from scipy.optimize import minimize
#
#
#     # args = (R, Rc, P, sigmaX)
#     low_conD = np.clip(sigmaX - math.sqrt(P), 0, None)
#     up_conD = sigmaX
#
#     sigmaX_H=low_conD#always ===low_bound
#     # low_beta=max(math.sqrt(1-math.exp(-2(R+Rc))),math.pow(sigmaX,2)+math.pow(sigmaX_H,2)-P)/2*sigmaX*sigmaX_H))
#     aa = math.sqrt(1 - math.exp(-2 * (R + Rc)))
#     bb = (math.pow(sigmaX, 2) + math.pow(sigmaX_H, 2) - P) / 2 * sigmaX * sigmaX_H
#     low_beta = max(aa, bb)
#     up_beta=1
#
#     beta=up_beta#always ===low_bound
#     low_alpha = beta* sigmaX - sigmaX * math.sqrt(np.clip(low_beta ** 2 - 1 + math.exp(-2 * (R + Rc)), 0, None))
#     low_alpha = low_alpha / sigmaX_H
#     up_alpha = beta * sigmaX / sigmaX_H
#
#     argsa = (low_alpha, up_alpha)
#     cons_a = consD(argsa)
#
#     x0 = 1
#     minDstar = lambda alpha: (math.pow(sigmaX, 2) + math.pow(sigmaX_H, 2) - 2 * sigmaX *  math.sqrt(
#         (1 - math.exp(-2 * R))) * math.sqrt(math.pow(sigmaX_H, 2) - np.clip(sigmaX* math.exp(-(R + Rc)) - math.sqrt(math.pow(sigmaX,2) -2*alpha*beta*sigmaX*sigmaX_H+ math.pow(alpha*sigmaX, 2)), 0,
#                    None) / math.pow(alpha, 2)))
#     res = minimize(minDstar, x0, method='SLSQP',
#                    options={'xatol': 1e-8, 'disp': True}, constraints=cons_a)
#     print(f'R==star2=={R}')
#     print(f"low and up {low_conD},{up_conD}")
#     print(f'res.===={res.x[0]}')
#     print(f'res.fun===={res.fun}')
#
#     return res.fun
#
def D2(R, Rc, P, sigmaX):
    threshold=1/2*math.log(sigmaX**2/P)
    if R<=threshold:
        D=2*(sigmaX**2)*math.exp(-2*R )+P-2*sigmaX*math.exp(-R)*math.sqrt(P)
    else:
        D = sigmaX ** 2 * math.exp(-2 * R)
    return D

# ***************************************
def calculate_alpha(sigma_x, sigma_x_hat, P, R, Rc):
    # Calculate the range for beta
    beta_low = np.clip((sigma_x ** 2 + sigma_x_hat ** 2 - P) / (2 * sigma_x * sigma_x_hat), 0, 1)
    beta_high = 1

    # Define a fine grid for beta within the range
    betas = np.linspace(beta_low, beta_high, num=100)
    term1=np.exp(-(R + Rc)) - np.sqrt(np.clip(1 - betas ** 2, 0, None))
    # Calculate alpha for each beta and find the minimum alpha
    alphas = np.clip(np.power(term1,2), 0, None) ** 2 / betas ** 2
    alpha_min = np.min(alphas)
    return alpha_min


def objective(sigma_x_hat, sigma_x, R, Rc, P):
    # Calculate alpha_min for the given sigma_x_hat
    alpha_min = calculate_alpha(sigma_x, sigma_x_hat, P, R, Rc)

    # Compute the objective D* for the given alpha_min alphas = {ndarray: (100,)} [0.97056849 0.97071532 0.97086293 0.97101132 0.9711605  0.9713105, 0.97146132 0.97161298 0.97176549 0.97191886 0.97207311 0.97222825, 0.9723843  0.97254128 0.9726992  0.97285807 0.97301792 0.97317876, 0.97334061 0.97350349 0.97366742 0.97383242 0.97399851 ...View as Arrayand sigma_x_hat
    D_star = sigma_x ** 2 + sigma_x_hat ** 2 - 2 * sigma_x * sigma_x_hat * np.sqrt(
        (1 - np.exp(-2 * R)) * (1 - alpha_min))
    return D_star


def minimize_D_star(R, Rc, P, sigma_x):
    # Define the range for sigma_x_hat
    sigma_x_hat_low = np.clip(sigma_x - np.sqrt(P), 0, None)
    sigma_x_hat_high = sigma_x

    # Define a fine grid for sigma_x_hat within the range
    sigma_x_hats = np.linspace(sigma_x_hat_low, sigma_x_hat_high, num=100)

    # Calculate the objective D* for each sigma_x_hat and find the minimum D*
    D_stars = [objective(sigma_x_hat, sigma_x, R, Rc, P) for sigma_x_hat in sigma_x_hats]
    D_star_min = np.min(D_stars)
    return D_star_min

def calculate_g(sigma_x, sigma_x_hat, P, R, Rc):
    # Calculate the range for beta
    beta_low = np.clip((sigma_x ** 2 + sigma_x_hat ** 2 - P) / (2 * sigma_x * sigma_x_hat), 0, 1)
    a=np.clip(np.sqrt(1 - np.exp(-2 * (R + Rc))), 0, None)
    beta_low=np.max([beta_low,a])
    beta_high = 1

    # Define a fine grid for beta within the range
    betas = np.linspace(beta_low, beta_high, num=100)
    def g(sigma_x_hat, sigma_x,  beta):
        alpha_low = beta * sigma_x - sigma_x_hat * np.sqrt(np.clip(beta ** 2 - 1 + np.exp(-2 * (R + Rc)),0,None))
        alpha_low = np.clip(alpha_low, 0, sigma_x)
        alpha_high = beta * sigma_x / sigma_x_hat

        def objective_g(alpha):
            term1 = sigma_x * np.exp(-(R + Rc))
            term2 = (sigma_x ** 2 - 2 * alpha * beta * sigma_x * sigma_x_hat + (sigma_x_hat * alpha) ** 2)
            term2 = np.clip(term2, 0, None)
            return -np.clip(np.sqrt(term1 - term2), 0, None)

        # Minimize the objective_alpha function
        res = minimize_scalar(objective_g, bounds=(alpha_low, alpha_high), method='bounded')
        return -res.fun

  
    gs= [g(sigma_x, sigma_x_hat, beta) for beta in betas]
    g_min = np.min(gs)
    return g_min
def objective2(sigma_x_hat, sigma_x, R, g_min):

    # Compute the objective D* for the given alpha_min and sigma_x_hat
    D_star2 = sigma_x ** 2 + sigma_x_hat ** 2 - 2 * sigma_x * np.sqrt(
        (1 - np.exp(-2 * R)) * (1 - g_min))

    return D_star2

def minimize_D2_star(R, Rc, P, sigma_x):
    # Define the range for sigma_x_hat
    sigma_x_hat_low = np.clip(sigma_x - np.sqrt(P), 0, None)
    sigma_x_hat_high = sigma_x

    # Define a fine grid for sigma_x_hat within the range
    sigma_x_hats = np.linspace(sigma_x_hat_low, sigma_x_hat_high, num=100)
    # Calculate the objective D* for each sigma_x_hat and find the minimum D*
    gmins = [calculate_g(sigma_x, sigma_x_hat, P,R, Rc) for sigma_x_hat in sigma_x_hats]
    D2_stars=[]
    for i in range(len(gmins)):
        D2_star = objective2(sigma_x_hats[i], sigma_x, R, gmins[i])
        D2_stars.append(D2_star)

    # D2_stars = [objective2(sigma_x_hat, sigma_x, R, gmin) for sigma_x_hat in sigma_x_hats for gmin in gmins]
    D_star2_min = np.min(D2_stars)
    return D_star2_min
def plotT4DstarR():
    # \underline{D}^* vs R ,R_c=1,P=0.1,\sigma^2_X=1
    # and overlineD_hat
    P = 0.1
    sigmaXH = sigmaP(P)
    # sigmaXH = 0.7023100624191236   #P=0.1
    Rc = 0.1
    sigmaX = 1
    # R_c = 0, R_c = 1, R_c =\infty
    Rc0_list = []  # d0
    Rc1_list = []  # d*
    Rc2_list = []  # d**
    Rc3_list = []  # d**
    R_list = []
    # start =0
    # stop =3
    # num = 100
    start =0.0000001
    stop =2.0
    num = 1000
    # overlineD(R,Rc,P,sigmaX,sigmaXH)  rc=1 P=1 ,sigmaX=1,sigmaXH=
    for r in np.linspace(start, stop, num):
        R_list.append(r)  #
        print(f'r======{r}')

        Rc0_list.append(D2(r, Rc, P, sigmaX))
        # minimize_D_star
        Rc1_list.append(overlineD_hat(r, Rc, P, sigmaX, sigmaXH))
        # Rc1_list.append(minimize_D_star(r, Rc, P, sigmaX))
        # Rc2_list.append(min_underDstar2forA_Sup(r, Rc, P, sigmaX))
        Rc2_list.append(min_underDstar2forA_Sup_hat(r, Rc, P, sigmaX)) #rewrite new equation 1 for loop
        # Rc2_list.append(underlineD_hat(r, Rc, P, sigmaX, sigmaXH))
        # Rc2_list.append(min_underDstar2for(r, Rc, P, sigmaX))
        # Rc2_list.append(min_underDstar2for(r, Rc, P, sigmaX))
        Rc3_list.append(underlineD_hat(r, Rc, P, sigmaX, sigmaXH))# more accurate than for loop
        # Rc3_list.append(min_underDstar2forA(r, Rc, P, sigmaX))
    # label0 = r'$\underline{D}^{\prime}$'
    label0 = r'$D$'
    label1 = r'$\overline{D}(R,R_c,P|W^2_2)$'#

    label2=r'$\underline{D}^{\prime}(R,R_c,P|W^2_2)$'
    label3 = r'$\underline{D}(R,R_c,P|W^2_2)$'
    line1 = plt.plot(R_list, Rc1_list, 'b', linestyle='--', label=label1, linewidth=2.0)
    # line2 = plt.plot(R_list, Rc2_list, 'sienna',linestyle='-',  label=label2, linewidth=2.0)
    line2 = plt.plot(R_list, Rc2_list, 'sienna', label=label2, linewidth=2.0)
    line3 = plt.plot(R_list, Rc3_list, 'r', linestyle=':', label=label3, linewidth=2.0)
    # line2 = plt.plot(R_list, Rc1_list, 'r', label=label1, linewidth=2.0)

    plt.ylabel(r'$D$')
    plt.xlabel(r'$R$')

    plt.legend(loc='upper right')
    # Adjust the subplot layout
    # plt.gcf().subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.3)
    plt.savefig('Fig4_new_P0.1Rc0.1.pdf')
    plt.show()
    print("sigmaXH====", sigmaXH)
def plotT4DstarR_diff():

    # \underline{D}^* vs R ,R_c=0.1,P=0.1,\sigma^2_X=1
    # and overlineD_hat
    P = 0.1
    sigmaXH = sigmaP(P)
    # sigmaXH = 0.7023100624191236   #P=0.1
    Rc = 0.1
    sigmaX = 1
    # R_c = 0, R_c = 1, R_c =\infty
    Rc0_list = []  # d0
    Rc1_list = []  # d*
    Rc2_list = []  # d**
    Rc3_list = []  # d**
    R_list = []
    start =1
    stop =2.0
    num = 10000
    P_thresholds=[]
    # overlineD(R,Rc,P,sigmaX,sigmaXH)  rc=1 P=1 ,sigmaX=1,sigmaXH=
    # import numpy as np
    # from scipy.optimize import minimize
    #
    # def equation(r):
    #     return 2 - np.exp(-2 * r) - 2 * np.sqrt(1 - np.exp(-2 * r)) * np.sqrt(1 - np.exp(-2 * (r + 0.1)))
    #
    # # Objective function to minimize
    # def objective(r):
    #     return abs(equation(r))
    #
    # # Initial guess for r
    # r_initial = 1.0
    #
    # # Minimize the objective function
    # result = minimize(objective, r_initial, method='SLSQP')
    #
    # print(f'The value of r that makes the equation approximately zero is: {result.x[0]}')
    #


    for r in np.linspace(start, stop, num):
        R_list.append(r)  #
        print(f'r======{r}')

        # Rc0_list.append(D2(r, Rc, P, sigmaX))
        # minimize_D_star
        # Rc1_list.append(overlineD_hat(r, Rc, P, sigmaX, sigmaXH))
        # Rc1_list.append(minimize_D_star(r, Rc, P, sigmaX))
        # Rc2_list.append(min_underDstar2forA_Sup(r, Rc, P, sigmaX))
        Rc2_list.append(min_underDstar2forA_Sup_hat(r, Rc, P, sigmaX)) #best choose
        # Rc2_list.append(underlineD_hat(r, Rc, P, sigmaX, sigmaXH))
        # Rc2_list.append(min_underDstar2for(r, Rc, P, sigmaX))
        # Rc2_list.append(min_underDstar2for(r, Rc, P, sigmaX))
        Rc3_list.append(underlineD_hat(r, Rc, P, sigmaX, sigmaXH))#best choose
        # Rc3_list.append(min_underDstar2forA(r, Rc, P, sigmaX))
        bb=2 - np.exp(-2 * r) - 2 * np.sqrt(1 - np.exp(-2 * r)) * np.sqrt(1 - np.exp(-2 * (r + Rc)))
        P_thresholds.append(sigmaX**2 * bb-P)
    # label0 = r'$\underline{D}^{\prime}$'
    label0 = r'$D_diff$'
    label1 = r'$\overline{D}(R,R_c,P|W^2_2)$'#
    diff=[(Rc2_list[i]-Rc3_list[i]) for i in range(len(Rc3_list))]
    # diff = [0 if abs(d) < 1e-4 else d for d in diff]
    # non_zero_indices = [index for index, value in enumerate(diff) if value > 0]
    # print(f'R_===={R_list[non_zero_indices[-1]]}')
    P_index=[index for index, value in enumerate(P_thresholds) if value >=0]
    print(f'P_thresholds-P>0  ={P_thresholds}')
    print(f'R={R_list[P_index[-1]]}')     #P=0.1,Rc =0.1, R: 1.051981
# the bound of r======1.0521052579157917
#
# R=1.0519052379237923
# 1.053
    # print(f'RRR={R_list[P_index[-1]]}')
    # # [2, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76]
    # label2=r'$\underline{D}^{\prime}(R,R_c,P|W^2_2)$'
    # label3 = r'$\underline{D}(R,R_c,P|W^2_2)$'
    # label4 = r'$(\underline{D}-\underline{D}^{\prime})* 1e5$'
    # # Extract data for the zoomed region
    # R_zoom = [R_list[i] for i in np.arange(0,non_zero_indices[-1]+1)]
    # P_zoom= [R_list[i] for i in P_index]
    # print(f'P={P},Rc={Rc},R======={R_zoom}')
    #
    # # Rc2_zoom = [Rc2_list[i] / 1e10 for i in non_zero_indices]
    # # Rc3_zoom = [Rc3_list[i] / 1e10 for i in non_zero_indices]
    # # Rc_diff=[(Rc3_list[i]-Rc2_list[i]) * 1e5 for i in non_zero_indices]
    # P_diff=[diff[i] for i in P_index]
    # d_diff=[diff[i] for i in np.arange(0,non_zero_indices[-1]+1)]
    # # # line1 = ax.plot(R_list, Rc1_list, 'b', linestyle='--', label=label1, linewidth=2.0)
    # # line2 = ax.plot(R_list, Rc2_list, 'sienna',linestyle='-',  label=label2, linewidth=2.0)
    # # line3 = ax.plot(R_list, Rc3_list, 'r', linestyle=':', label=label3, linewidth=2.0)


    # Plotting the results
    # fig, ax = plt.subplots(figsize=(10, 6))
    # ax.plot(R_list, Rc0_list, label=label0)
    # ax.plot(R_list, Rc1_list, 'b', linestyle='--', label=label1, linewidth=2.0)
    # ax.plot(R_list, Rc2_list, 'sienna',linestyle='-',  label=label2, linewidth=2.0)
    # ax.plot(R_list, Rc3_list, 'r', linestyle=':', label=label3, linewidth=2.0)
    # ax.plot(R_list, diff,  'r',linestyle='-',label=r'$\underline{D}^{\prime}-\underline{D}$')
    # ax.plot(R_list, diff, 'r', linestyle='-',label=r'$\underline{D}^{\prime}-\underline{D}$')
    # label2=r'$\underline{D}^{\prime}(R,R_c,P|W^2_2)$'
    # label3 = r'$\underline{D}(R,R_c,P|W^2_2)$'
    plt.plot(R_list, diff, 'r', linestyle='-', label=r'$\underline{D}^{\prime}(R,R_c,P|W^2_2)-\underline{D}(R,R_c,P|W^2_2)$')
    # for r, P_threshold,diff in zip(R_list, P_thresholds,diff):
    #     if (P - P_threshold) < 0 and P> 0:
    #         print("greater than 0")
    #         # plt.plot(r, diff, marker='*', color='b', markersize=5)
    #     else:
    #         print(f'the bound of r======{r}')
    #         # plt.plot(r, diff, marker='*', color='g', markersize=5)

    # ax.plot(R_list, P_diff, 'sienna', linestyle=':', label='P_diff')
    # ax.plot(R_zoom, d_diff, 'g', linestyle='--', label='d_diff')
    # ax.plot(R_list, P_thresholds, 'sienna', linestyle=':', label='P_theshold')

    # ax.plot(R_list, Rc2_list, label=label2)
    # ax.plot(R_list, Rc3_list, label=label3)
    plt.legend()
    plt.xlabel(r'$R$')
    plt.ylabel(r'$\underline{D}^{\prime}-\underline{D}$')
    # ax.set_title('Plot of different D functions vs. R')

    # Inset for zoomed region
    # ax_inset = inset_axes(ax, width="30%", height="30%", loc="center")
    # ax_inset.plot(R_zoom, Rc_diff, 'green',linestyle='-',label=label4)
    # # ax_inset.plot(R_zoom, Rc3_zoom, 'r', linestyle=':')
    # ax_inset.set_xlim([min(R_zoom), max(R_zoom)])
    # ax_inset.set_ylim([min(Rc_diff), max(Rc_diff)])
    # ax_inset.set_xlabel('R')
    # ax_inset.set_ylabel(label4)
    # ax_inset.set_title('Plot of different D functions vs. R')
    # ax_inset.set_ylim([min(min(Rc2_zoom), min(Rc3_zoom)), max(max(Rc2_zoom), max(Rc3_zoom))])
    # ax_inset.legend(loc='upper right')

    # Mark the zoomed region on the main plot
    # mark_inset(ax, ax_inset, loc1=4, loc2=4, fc="none", ec="0.5")

    # plt.show()
    #
    # # 创建主图
    # fig, ax = plt.subplots()
    # # ax.plot(x, y1, label='sin(x)')
    # # ax.plot(x, y2, label='sin(x + 0.1)')
    # # ax.legend()
    # # line1 = ax.plot(R_list, Rc1_list, 'b', linestyle='--', label=label1, linewidth=2.0)
    # line2 = ax.plot(R_list, Rc2_list, 'sienna',linestyle='-',  label=label2, linewidth=2.0)
    # line3 = ax.plot(R_list, Rc3_list, 'r', linestyle=':', label=label3, linewidth=2.0)
    # # line2 = plt.plot(R_list, Rc1_list, 'r', label=label1, linewidth=2.0)
    # ax.set_ylabel(r'$D$')
    # ax.set_xlabel(r'$R$')
    # ax.legend(loc='upper right')
    # # 添加插入的放大器
    # axins = inset_axes(ax, width="30%", height="30%", loc='center')
    # # 在放大区域中绘制相同的数据
    # axins.plot(R_list, Rc2_list,'sienna',linestyle='-')
    # axins.plot(R_list, Rc3_list,'r', linestyle=':')
    # # 设置放大区域的范围
    # x1, x2, y1, y2 = R_list[56], R_list[76], Rc2_list[56], Rc2_list[76] # 放大的x和y范围
    # # x1, x2, y1, y2 = 0.0, 0.06, 1.5, 2.1  # 放大的x和y范围
    # axins.set_xlim(x1, x2)
    # axins.set_ylim(y1, y2)
    # # 设置放大区域的刻度
    # axins.set_xticklabels('')
    # axins.set_yticklabels('')
    # # 添加圆圈表示的放大区域
    # circle = Ellipse(xy=((x1 + x2) / 2, (y1 + y2) / 2), width=(x2 - x1)*2, height=(y2 - y1),
    #                  edgecolor='red', fc='none', lw=1.5, ls='--')
    # # circle = Ellipse(xy=(0.1, 1.5), width=(x2 - x1)/2, height=(y2 - y1)/2,
    # #                  edgecolor='red', fc='none', lw=1.5, ls='--')
    # ax.add_patch(circle)
    # # 添加放大区域的指示线
    # con1 = ConnectionPatch(xyA=((x1 + x2) / 2, (y1 + y2) / 2), coordsA=ax.transData,
    #                        xyB=(0.5, 0.5), coordsB=axins.transAxes,
    #                        color='red', lw=1.5, ls='--')
    # con2 = ConnectionPatch(xyA=((x1 + x2) / 2, (y1 + y2) / 2), coordsA=ax.transData,
    #                        xyB=(1, 1), coordsB=axins.transAxes,
    #                        color='red', lw=1.5, ls='--')
    # ax.add_artist(con1)
    # ax.add_artist(con2)
    # Adjust the subplot layout
    # plt.gcf().subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.3)

    plt.savefig('theroem6RD_fullsize_P_0.1_Rc0.1.pdf')
    plt.show()
    print("sigmaXH====", sigmaXH)
def plotT4DstarP_diff():
    from matplotlib.patches import Ellipse, ConnectionPatch
    from mpl_toolkits.axes_grid1.inset_locator import (inset_axes,
                                                       mark_inset)
    # \underline{D}^* vs R ,R_c=1,P=0.1,\sigma^2_X=1
    # and overlineD_hat
    R=0.1
    # sigmaXH = sigmaP(P)
    # sigmaXH = 0.7023100624191236   #P=0.1
    Rc = 0.1
    sigmaX = 1
    # R_c = 0, R_c = 1, R_c =\infty
    Rc0_list = []  # d0
    Rc1_list = []  # d*
    Rc2_list = []  # d**
    Rc3_list = []  # d**
    R_list = []
    start =0.00000001
    stop =1.0#
    num = 10000
    aa=2 - np.exp(-2 * R) - 2 * np.sqrt(1 - np.exp(-2 * R)) * np.sqrt(1 - np.exp(-2 * (R + Rc)))

    P_threshold=sigmaX**2 *aa

    P_thresholds=[]
    # stop=P_thresholds
    # overlineD(R,Rc,P,sigmaX,sigmaXH)  rc=1 P=1 ,sigmaX=1,sigmaXH=
    for r in np.linspace(start, stop, num): #r represent P
        R_list.append(r)  #
        sigmaXH = sigmaP(r)
        print(f'P======{r}')


        # Rc0_list.append(D2(R, Rc, r, sigmaX))
        # minimize_D_star
        # Rc1_list.append(overlineD_hat(r, Rc, P, sigmaX, sigmaXH))
        # Rc1_list.append(minimize_D_star(r, Rc, P, sigmaX))
        # Rc2_list.append(min_underDstar2forA_Sup(r, Rc, P, sigmaX))
        Rc2_list.append(min_underDstar2forA_Sup_hat(R, Rc, r, sigmaX))
        # Rc2_list.append(underlineD_hat(r, Rc, P, sigmaX, sigmaXH))
        # Rc2_list.append(min_underDstar2for(r, Rc, P, sigmaX))
        # Rc2_list.append(min_underDstar2for(r, Rc, P, sigmaX))
        Rc3_list.append(underlineD_hat(R, Rc, r, sigmaX, sigmaXH))#best choose

        # Rc3_list.append(min_underDstar2forA(r, Rc, R, sigmaX))
        # bb=2 - np.exp(-2 * r) - 2 * np.sqrt(1 - np.exp(-2 * r)) * np.sqrt(1 - np.exp(-2 * (r + Rc)))
        # P_thresholds.append(sigmaX**2 * bb-P)
        P_thresholds.append(P_threshold)

    diff=[(Rc2_list[i]-Rc3_list[i]) for i in range(len(Rc3_list))]
    non_zero_indices = [index for index, value in enumerate(diff) if value >= 0]
    print(non_zero_indices)
    print(f'P_thresholds===={P_threshold}')

    # Plotting the results
    # fig, ax = plt.subplots(figsize=(10, 6))
    plt.plot(R_list, diff, 'r', linestyle='-',
            label=r'$\underline{D}^{\prime}(R,R_c,P|W^2_2)-\underline{D}(R,R_c,P|W^2_2)$')
    # ax.plot(R_list, diff,  'r',linestyle='-',label=r'$\underline{D}^{\prime}-\underline{D}$')
    # for p, P_threshold in zip(R_list, P_thresholds):
    #     if (p - P_threshold) < 0 and p > 0:
    #         ax.plot(R_list, diff, marker='*', color='b', markersize=5)
    
    # for p, dif in zip(R_list, diff):
    #     if dif> 0:
    #         ax.plot(R_list, diff, marker='*', color='b', markersize=5)

    plt.legend()
    plt.xlabel(r'$P$')
    plt.ylabel(r'$\underline{D}^{\prime}-\underline{D}$')


    plt.savefig('theroem6_fullsize_DP_R0.1_P0.1.png')
    plt.show()


def plotT6underlineDstarR():
    # \underline{D}^* vs R ,R_c=1,P=0.1,\sigma^2_X=1
    # and overlineD_hat
    P = 0.1
    sigmaXH = sigmaP(P)
    # sigmaXH = 0.7023100624191236   #P=0.1
    Rc = 0.1
    sigmaX = 1
    # R_c = 0, R_c = 1, R_c =\infty
    Rc0_list = []  # d0
    Rc1_list = []  # d*
    Rc2_list = []  # d**
    Rc3_list = []  # d**
    R_list = []
    Rc4_list = []
    Rc5_list = []

    start =0
    stop =2
    num = 100
    # overlineD(R,Rc,P,sigmaX,sigmaXH)  rc=1 P=1 ,sigmaX=1,sigmaXH=
    for r in np.linspace(start, stop, num):
        R_list.append(r)  #
        print(f'r=====***********====={r}\n')

        Rc0_list.append(D2(r, Rc, P, sigmaX))
        # minimize_D_star
        Rc1_list.append(overlineD_hat(r, Rc, P, sigmaX, sigmaXH))

        Rc2_list.append(min_underDstar2forA_Sup(r, Rc, P, sigmaX))
        # Rc2_list.append(underlineD_hat_T6(r, Rc, P, sigmaX, sigmaXH, 10))

        Rc3_list.append(underlineD_hat_T6(r, Rc, P, sigmaX, sigmaXH,1))
        # Rc3_list.append(min_underDstar2forA(r, Rc, P, sigmaX))
    # label0 = r'$\underline{D}^{\prime}$'
    # label0 = r'$D$'
    label1 = r'$\overline{D}(R,R_c,P|W^2_2)$'#

    label2=r'$\underline{D}^{\prime}(R,R_c,P|W^2_2)$'
    label3 = r'$\underline{D}(R,R_c,P|W^2_2)$'
    # label4 = r'$\underline{D}(R,R_c,P|W^2_2)forA$'
    # label5 = r'$\underline{D}(R,R_c,P|W^2_2)forA$'
    line1 = plt.plot(R_list, Rc1_list, 'b', linestyle='--', label=label1, linewidth=2.0)
    line2 = plt.plot(R_list, Rc2_list, 'sienna',linestyle='-',  label=label2, linewidth=2.0)
    line3 = plt.plot(R_list,Rc3_list, 'r', linestyle=':', label=label3, linewidth=2.0)
    # line4 = plt.plot(R_list, Rc4_list, 'g', label=label4, linewidth=2.0)
    # line5 = plt.plot(R_list, Rc5_list, color=(0,0.4,0.3), label=label4, linewidth=2.0)

    # plt.yscale('log')

    plt.ylabel(r'$D$')
    plt.xlabel(r'$R$')

    plt.legend(loc='upper right')
    plt.savefig('theroem6RD_prime_fig3linesp_01.pdf')
    plt.show()
    print("sigmaXH====", sigmaXH)
def plotT6underlineDstarR_a():
    # \underline{D}^* vs R ,R_c=1,P=0.1,\sigma^2_X=1
    # and overlineD_hat
    P = 0.01
    sigmaXH = sigmaP(P)
    # sigmaXH = 0.7023100624191236   #P=0.1
    Rc = 1.0
    sigmaX = 1
    # R_c = 0, R_c = 1, R_c =\infty
    Rc0_list = []  # d0
    Rc1_list = []  # d*
    Rc2_list = []  # d**
    Rc3_list = []  # d**
    R_list = []
    Rc4_list = []
    Rc5_list = []

    start =0
    stop =2
    num = 100
    # overlineD(R,Rc,P,sigmaX,sigmaXH)  rc=1 P=1 ,sigmaX=1,sigmaXH=
    for r in np.linspace(start, stop, num):
        R_list.append(r)  #
        print(f'r======{r}')

        Rc0_list.append(D2(r, Rc, P, sigmaX))
        # minimize_D_star
        Rc1_list.append(overlineD_hat(r, Rc, P, sigmaX, sigmaXH))
        # Rc1_list.append(minimize_D_star(r, Rc, P, sigmaX))
        Rc2_list.append(min_underDstar2forA_Sup(r, Rc, P, sigmaX))
        # Rc2_list.append(underlineD_hat_T6_minimax(r, Rc, P, sigmaX, sigmaXH))
        # Rc2_list.append(min_underDstar2for(r, Rc, P, sigmaX))
        # Rc2_list.append(min_underDstar2for(r, Rc, P, sigmaX))
        Rc3_list.append(underlineD_hat_T6(r, Rc, P, sigmaX, sigmaXH,1))
        # Rc4_list.append(min_underDstar2forA(r, Rc, P, sigmaX))
    # label0 = r'$\underline{D}^{\prime}$'
    label0 = r'$D$'
    label1 = r'$\overline{D}(R,R_c,P|W^2_2)$'#
    label2=r'optimal_a'

    # label2=r'$\underline{D}^{\prime}(R,R_c,P|W^2_2)$'
    label3 = r'$\underline{D}(R,R_c,P|W^2_2)$'
    label4 = r'$\underline{D}(R,R_c,P|W^2_2)forA$'
    label5 = r'$\underline{D}(R,R_c,P|W^2_2)forA$'
    # line1 = plt.plot(R_list, Rc1_list, 'b', linestyle='--', label=label1, linewidth=2.0)
    line2 = plt.plot(R_list, Rc2_list, 'sienna',linestyle='-',  label=label2, linewidth=2.0)
    # line3 = plt.plot(R_list,Rc3_list, 'r', linestyle=':', label=label3, linewidth=2.0)
    # line4 = plt.plot(R_list, Rc4_list, 'g', label=label4, linewidth=2.0)
    # line5 = plt.plot(R_list, Rc5_list, color=(0,0.4,0.3), label=label4, linewidth=2.0)

    # plt.yscale('log')

    plt.ylabel(r'$A$')
    plt.xlabel(r'$R$')

    plt.legend(loc='upper right')
    plt.savefig('theroem6RD_prime_fig3lines_plot_a1.pdf')
    plt.show()
    print("sigmaXH====", sigmaXH)
def consx(args):
    min, max = args
    cons = ({'type': 'ineq', 'fun': lambda x: -x +max}, \
            {'type': 'ineq', 'fun': lambda x: x - min})
    return cons
def minx(p):
    xp_l = []
    for i in np.linspace(0.00000001,1,10000):
        x=i
        threshold=-math.log(x)+0.5*(x**2)-0.5

        if threshold<=p:
            XP=math.sqrt(2*(1-math.exp(-p)))-1+x
            xp_l.append(XP)

    XPmin=min(xp_l)
    print(f'XPmin={XPmin}')
    return XPmin
def plotxp():
    directory = "./results"
    plt.rcParams['text.usetex'] = True
    P = 1
    sigmaXH = sigmaP(P)
    sigmaXH = 0.2290613399456898
    Rc = 1
    sigmaX = 1
    # R_c = 0, R_c = 1, R_c =\infty
    Rc0_list = []  # d0
    Rc1_list = []  # d1
    Rci_list = []  # di
    R_list = []
    start = 0
    stop = 10
    num = 1000
    # overlineD(R,Rc,P,sigmaX,sigmaXH)  rc=1 P=1 ,sigmaX=1,sigmaXH=
    for p in np.linspace(start, stop, num):
        R_list.append(p)  #
        print(f'p={p}')

        Rc0_list.append(minx(p))

    label0 = r'$x$'
    line1 = plt.plot(R_list, Rc0_list, 'b', label=label0, linewidth=2.0)
    plt.ylabel(r'$F$')
    plt.xlabel(r'$p$')

    plt.legend(loc='upper right')
    plt.savefig('theroem4RD_underline_star.pdf')
    plt.show()
    # print("sigmaXH====", sigmaXH)

def plotT5DR11():
    # underlineD and baselineD
    Rc0_list = []  # d0
    Rc1_list = []  # d1
    Rci_list = []  # di
    R_list = []
    start = 0.0
    stop = 5
    # stopR=math.log(2)
    num = 100
    for a in np.linspace(start, stop, num):
        R, D = T4DR(a)


        R_list.append(R)  # P list
        Rc0_list.append(D)
        # Rcr1=R_cr1(d)s
        Rc1_list.append(baseline(R))
        # Rci_list.append(C2_DRRcP2(math.inf,sigmaXH))
    # label0=r'$P_{\mathrm{c}}=0$'
    # label1 = r'$P_{\mathrm{c}}=1$'
    # label2=r'$P_{\mathrm{c}}=2$'
    label0 = r'$\overline{D}_s$'
    label1 = r'$\overline{D}$'
    # label2=r'$R=\infty$'
    # line3 = plt.plot(R_list,Rci_list, 'sienna', label=label2, linewidth=2.0)
    line2 = plt.plot(R_list, Rc1_list, 'r', label=label1, linewidth=2.0)
    line1 = plt.plot(R_list, Rc0_list, 'b', label=label0, linewidth=2.0)

    plt.ylabel(r'$D$')
    plt.xlabel(r'$R$')
    plt.legend(loc='upper right')
    plt.savefig('T4DR1-log2.pdf')
    plt.show()
def plotT5DR12():
    # underlineD and baselineD
    Rc0_list = []  # d0
    Rc1_list = []  # d1
    Rci_list = []  # di
    R_list = []
    start = 0.0
    stop = 5
    # stopR=math.log(2)
    num = 100
    for a in np.linspace(start, stop, num):
        R, D = T4DR(a)


        R_list.append(R)  # P list
        Rc0_list.append(D)
        # Rcr1=R_cr1(d)s
        Rc1_list.append(baseline(R))
        # Rci_list.append(C2_DRRcP2(math.inf,sigmaXH))
    # label0=r'$P_{\mathrm{c}}=0$'
    # label1 = r'$P_{\mathrm{c}}=1$'
    # label2=r'$P_{\mathrm{c}}=2$'
    label0 = r'$\overline{D}_s$'
    label1 = r'$\overline{D}$'
    # label2=r'$R=\infty$'
    # line3 = plt.plot(R_list,Rci_list, 'sienna', label=label2, linewidth=2.0)
    line2 = plt.plot(R_list, Rc1_list, 'r', label=label1, linewidth=2.0)
    line1 = plt.plot(R_list, Rc0_list, 'b', label=label0, linewidth=2.0)

    plt.ylabel(r'$D$')
    plt.xlabel(r'$R$')
    plt.legend(loc='upper right')
    plt.savefig('T4DR1-log2.pdf')
    plt.show()

def DM(R,P,sigmaX):
    threshold= math.pow(sigmaX,2)*math.pow(1-math.sqrt(1-math.exp(-2*R)),2)
    if P>=threshold:
        D=math.pow(sigmaX,2)*math.exp(-2*R)
    else:
        D=math.pow(sigmaX,2)+math.pow(sigmaX-math.sqrt(P),2)-2*sigmaX*(sigmaX-math.sqrt(P))*math.sqrt(1-math.exp(-2*R))
    return D
def DJ(R,P,sigmaX):
    threshold= math.pow(sigmaX,2)*math.exp(-2*R)
    if P>=threshold:
        D=math.pow(sigmaX,2)*math.exp(-2*R)
    else:
        D=math.pow(sigmaX,2)*math.exp(-2*R)+math.pow(sigmaX*math.exp(-2*R)-math.sqrt(P),2)
    return D
def plotDMDJ():
    # underlineD_hat and overlineD_hat
    R= 1

    # sigmaXH = 0.2290613399456898
    Rc = 1
    sigmaX = 1
    # R_c = 0, R_c = 1, R_c =\infty
    Rc0_list = []  # d0
    Rc1_list = []  # d1
    Rci_list = []  # di
    R_list = []
    start = 0
    stop = 2
    num = 100
    sigmaX=1
    P=0.01
    # overlineD(R,Rc,P,sigmaX,sigmaXH)  rc=1 P=1 ,sigmaX=1,sigmaXH=
    for r in np.linspace(start, stop, num):

        # sigmaXH = sigmaP(P)
        R_list.append(r)  #

        Rc0_list.append(DM(r,  P, sigmaX))

        Rc1_list.append(DJ(r,  P, sigmaX))
    label1 = r'$D_J$'
    label0 = r'$D_M$'

    # label2=r'$R=\infty$'
    # line3 = plt.plot(R_list,Rci_list, 'sienna', label=label2, linewidth=2.0)
    line2 = plt.plot(R_list, Rc1_list, 'r', label=label1, linewidth=2.0)
    line1 = plt.plot(R_list, Rc0_list, 'b', label=label0, linewidth=2.0)


    plt.ylabel(r'$D$')
    plt.xlabel(r'$R$')

    plt.legend(loc='upper right')
    plt.savefig('plotDMDJ_slideP15.pdf')
    plt.show()
    print("sigmaXH====", sigmaXH)

############################ paper reviver suggestion plot ###############
def positive_part(x):
    return np.maximum(x, 0)

# Functions based on the provided equations
def Dn1(R, P,sigmaX):
    return sigmaX**2 * 2**(-2 * R) + positive_part(sigmaX * 2**(-R) - np.sqrt(P))**2

def Dn2(R, P,sigmaX):
    return sigmaX**2 * 2**(-2 * R) + positive_part(sigmaX - sigmaX * np.sqrt(1 - 2**(-2 * R)) - np.sqrt(P))**2

def Dn3(R, Rc,sigmaX):
    return 2 * sigmaX**2 - 2 * sigmaX**2 * np.sqrt((1 - 2**(-2 * R)) * (1 - 2**(-2 * (R + Rc))))
def Dn3_Rc_infinity(R,sigmaX):
    return 2 * sigmaX**2 - 2 * sigmaX**2 * np.sqrt(1 - 2**(-2 * R))
def plotDn3():
    # rc=0,rc=1,rc=\infinity
    Rc0_list = []  # d0
    Rc1_list = []  # d1
    Rci_list = []  # di
    R_list = []
    start = 0.0
    stop = 5
    sigmaX = 1
    # stopR=math.log(2)
    num = 5000
    for R in np.linspace(start, stop, num):
        # R, D = T4DR(a)


        R_list.append(R)  # P list
        Rc0_list.append(Dn3(R, 0,sigmaX))
        # Rcr1=R_cr1(d)s
        Rc1_list.append(Dn3(R, 1,sigmaX))
        Rci_list.append(Dn3_Rc_infinity(R,sigmaX))
    # label0=r'$P_{\mathrm{c}}=0$'
    # label1 = r'$P_{\mathrm{c}}=1$'
    # label2=r'$P_{\mathrm{c}}=2$'
    label0 = r'$R_c=0$'
    label1 = r'$R_c=1$'
    label2=r'$R_c=\infty$'
    line0 = plt.plot(R_list, Rc0_list, 'b', label=label0, linewidth=2.0)

    line1 = plt.plot(R_list, Rc1_list, 'sienna', label=label1, linewidth=2.0)
    line2 = plt.plot(R_list, Rci_list, 'r', label=label2, linewidth=2.0)


    plt.ylabel(r'$D$')
    plt.xlabel(r'$R$')
    plt.legend(loc='upper right')
    plt.savefig('RD_Dn3.pdf')
    plt.show()


def plotDn12_DP():
    # Corollary 2
    # RC=math.inf, interate P set R=0,R=0.1,R=math.inf

    Rc0_list = []  # d0
    Rc1_list = []  # d1
    Rci_list = []  # di
    R_list = []
    sigmaX=1
    start = 0.0
    stop = 0.25
    num = 10000

    for P in np.linspace(start, stop, num):

        R_list.append(P)  # P list
        Rc0_list.append(Dn1(1, P,sigmaX))

        Rc1_list.append(Dn2(1, P,sigmaX))

    label0 = r'$R_c=0$'
    label1 = r'$R_c=\infty$'
    # label2 = r'$R_c=\infty$'
    # label2 = r'$R=0.5$'
    # label2=r'$R=\infty$'
    # line3 = plt.plot(R_list, Rci_list, 'sienna', label=label2, linewidth=2.0)
    line1 = plt.plot(R_list, Rc0_list, 'b', label=label0, linewidth=2.0)
    line2 = plt.plot(R_list, Rc1_list, 'r', label=label1, linewidth=2.0)


    plt.ylabel(r'$D$')
    plt.xlabel(r'$P$')
    plt.legend(loc='upper right')
    plt.savefig('plotDn12_DP.pdf')
    plt.show()



def calculate_zeta(P, Rc, sigma2_x):
    # Calculate zeta values
    zeta1 = 4 * np.exp(-2 * Rc) - 1
    zeta2 = 4 * np.exp(-2 * Rc) + (2 * P) / sigma2_x
    zeta3 = (4 * sigma2_x**2 - P) * P / (sigma2_x**4)

    return zeta1, zeta2, zeta3

# # Given values
# P = 0.1       # Set your value for P
# Rc = 0.1      # Set your value for Rc
# sigma2_x = 1  # Set your value for sigma^2_x
#
# # Calculate zeta values
# zeta1, zeta2, zeta3 = calculate_zeta(P, Rc, sigma2_x)
#
# # Print results
# print(f"zeta1: {zeta1}")
# print(f"zeta2: {zeta2}")
# print(f"zeta3: {zeta3}")
#equation 35
def calculate_R(P, Rc, sigma2_x):
    # Constants defined in your equation
    # C1 = np.sqrt(sigma2_x ** 2 - 4 * sigma2_x * Rc) / (2 * Rc)
    # C2 = sigma2_x / Rc
    #
    # # Calculate C3 based on the equation
    # C3 = 2 * np.sqrt(sigma2_x ** 2 - 4 * sigma2_x * Rc)
    # Calculate zeta values
    zeta1, zeta2, zeta3 = calculate_zeta(P, Rc, sigma2_x)
    print(f"zeta1: {zeta1}")
    print(f"zeta2: {zeta2}")
    print(f"zeta3: {zeta3}")
    if P > sigma2_x:
        R = 0
    elif Rc < np.log(2) and P < sigma2_x:
        R = -0.5 * np.log(zeta3 / zeta2)
    else:
        a=zeta2-math.sqrt(zeta2**2-4*zeta1*zeta2)
        a=a/(2*zeta1)
        R = -0.5 * np.log(a)
        print(f'R==={R}')

    return R


# Given values
P = 0.1  # Set your value for P
Rc = 0.1  # Set your value for Rc
sigma2_x = 1  # Set your value for sigma^2_x

# Calculate R
R = calculate_R(P, Rc, sigma2_x)
print(f"The calculated value of R is: {R}")
if __name__ == "__main__":
    # calculate_R(0.1, 0.1, 1)
    directory = "./results"
    plt.rcParams['text.usetex'] = True
    # plotT4DR()
    # from scipy.optimize import fsolve
    # def equation123(R):
    #     return compute_expression(1, 0.22906133994565517, R, 1, 1)
    # # Initial guess for R
    # initial_guess = 0.05
    # # Use fsolve to find the root
    # solution = fsolve(equation123, initial_guess)
    # print(solution[0])
    # # Print the solution
    # print("equation123,The solution for R is:", solution[0])

    # def equation44(R):
    #     return varsigma(R, 1, 1)
    # # Initial guess for R
    # initial_guess = 0.0
    # # Use fsolve to find the root
    # solution44 = fsolve(equation44, initial_guess)
    # print(solution[0])
    #
    # # Print the solution
    # print("equation44,The solution for R is:", solution44[0])

    # plotTheorem3() #fig 5
    # plotC2moreover()
    # plotTheorem3()#fig5RD
    # plotTheorem3DP() #fig5PD fig6
    # plotoverlap()
    # plotC3C1()##fig 8
    # plotC3C2()#fig 9
    plotC3C3()
    # plotT4case2DR()#fig9DR
    # plotT4case2DP()#fig9DP
    # plotC2moreover()
    # plotC2moreoverDP()
    # plotTheorem2()
    # plotT4DstarR()

    # plotT6underlineDstarR_a()
    # plotDMDJ()
    # plotxp()
    # plotT6underlineDstarR()
    # plotPart2T8DR()
    # plotTheoremT5part2_11()
    # plotTheoremT5part2_12()
    # plotDn3()
    # plotDn12_DP()
    # plot_E123()
    # plot_E123_2()
    # plotT4DstarR()
    # plotT4DstarP_diff()
    # plotT4DstarR_diff()
    print('Plot done!!!!!!!!!!!!!!!!!')


    #   \mu_X=0, \sigma^2_X=1, R_c=1, P=1, 画\overline{D}和R的关系，以及\underline{D}和R关系。这两条曲线应该部分重合，如果不明显的话，把P改成0.5或0.1试试
    # P = 1
    # sigmaXH = sigmaP(P)
    # sigmaXH = 0.2290613399456898
    # Rc = 1
    # sigmaX = 1
    # # R_c = 0, R_c = 1, R_c =\infty
    # Rc0_list = []  # d0
    # Rc1_list = []  # d1
    # Rci_list = []  # di
    # R_list = []
    # start = 0
    # stop = 10
    # num = 1000
    # # overlineD(R,Rc,P,sigmaX,sigmaXH)  rc=1 P=1 ,sigmaX=1,sigmaXH=
    # for p in np.linspace(start, stop, num):
    #     R_list.append(p)  #
    #     print(f'p={p}')
    #
    #     Rc0_list.append(minx(p))
    #
    # label0 = r'$x$'
    # line1 = plt.plot(R_list, Rc0_list, 'b', label=label0, linewidth=2.0)
    # plt.ylabel(r'$F$')
    # plt.xlabel(r'$p$')
    #
    # plt.legend(loc='upper right')
    # plt.savefig('theroem4RD_underline_star.pdf')
    # plt.show()
    # print("sigmaXH====", sigmaXH)2







