# Source: https://github.com/ruozhichen/rgb2Lab-rgb2hsl
# -*- coding: utf-8 -*-
import math
import numpy as np


def rgb2lab(inputColor):
    RGB = [0, 0, 0]
    for i in range(0, len(inputColor)):
        RGB[i] = inputColor[i] / 255.0

    X = RGB[0] * 0.4124 + RGB[1] * 0.3576 + RGB[2] * 0.1805
    Y = RGB[0] * 0.2126 + RGB[1] * 0.7152 + RGB[2] * 0.0722
    Z = RGB[0] * 0.0193 + RGB[1] * 0.1192 + RGB[2] * 0.9505
    XYZ = [X, Y, Z]
    XYZ[0] /= 95.045 / 100
    XYZ[1] /= 100.0 / 100
    XYZ[2] /= 108.875 / 100

    L = 0
    for i in range(0, 3):
        v = XYZ[i]
        if v > 0.008856:
            v = pow(v, 1.0 / 3)
            if i == 1:
                L = 116.0 * v - 16.0
        else:
            v *= 7.787
            v += 16.0 / 116
            if i == 1:
                L = 903.3 * XYZ[i]
        XYZ[i] = v

    a = 500.0 * (XYZ[0] - XYZ[1])
    b = 200.0 * (XYZ[1] - XYZ[2])
    Lab = [int(L), int(a), int(b)]
    return Lab


# colors size: n*3, when n is very large, it improves speed using matrix calculation
def rgb2lab_matrix(colors):
    n = len(colors)
    colors = np.array(colors)
    colors = colors.astype("float")
    RGBs = colors / 255.0
    Xs = RGBs[:, 0] * 0.4124 + RGBs[:, 1] * 0.3576 + RGBs[:, 2] * 0.1805
    Ys = RGBs[:, 0] * 0.2126 + RGBs[:, 1] * 0.7152 + RGBs[:, 2] * 0.0722
    Zs = RGBs[:, 0] * 0.0193 + RGBs[:, 1] * 0.1192 + RGBs[:, 2] * 0.9505
    XYZs = np.vstack((Xs, Ys, Zs)).transpose()

    XYZs[:, 0] = XYZs[:, 0] / (95.045 / 100.0)
    XYZs[:, 1] = XYZs[:, 1] / (100.0 / 100.0)
    XYZs[:, 2] = XYZs[:, 2] / (108.875 / 100.0)
    L = np.zeros((n, 3), dtype="float")
    for i in range(0, 3):
        v = XYZs[:, i]
        vv = np.where(v > 0.008856, v ** (1.0 / 3), v * 7.787 + 16.0 / 116)
        L[:, i] = np.where(v > 0.008856, 116.0 * vv - 16.0, v * 903.3)
        XYZs[:, i] = vv

    As = 500.0 * (XYZs[:, 0] - XYZs[:, 1])
    Bs = 200.0 * (XYZs[:, 1] - XYZs[:, 2])
    Ls = L[:, 1]
    LABs = np.vstack((Ls, As, Bs)).transpose()
    LABs = LABs.astype("int")
    return LABs


def lab2rgb(inputColor):
    L = inputColor[0]
    a = inputColor[1]
    b = inputColor[2]
    # d=6.0/29
    T1 = 0.008856
    T2 = 0.206893
    d = T2
    fy = math.pow((L + 16) / 116.0, 3)
    fx = fy + a / 500.0
    fz = fy - b / 200.0
    # Y = fy > d ? fy * fy * fy : (fy - 16.0 / 116) * 3 * d * d
    fy = (fy) if (fy > T1) else (L / 903.3)
    Y = fy
    fy = (
        (math.pow(fy, 1.0 / 3)) if (fy > T1) else (7.787 * fy + 16.0 / 116)
    )  # calculate XYZ[1], XYZ[0]=a/500.0+XYZ[1]

    # compute original XYZ[0]
    fx = fy + a / 500.0
    X = (
        (math.pow(fx, 3.0)) if (fx > T2) else ((fx - 16.0 / 116) / 7.787)
    )  # v^3>T1, so v>T1^(1/3)=

    # compute original XYZ[2]
    fz = fy - b / 200.0
    Z = (math.pow(fz, 3.0)) if (fz > T2) else ((fz - 16.0 / 116) / 7.787)

    X *= 0.95045
    Z *= 1.08875
    R = 3.240479 * X + (-1.537150) * Y + (-0.498535) * Z
    G = (-0.969256) * X + 1.875992 * Y + 0.041556 * Z
    B = 0.055648 * X + (-0.204043) * Y + 1.057311 * Z
    # R = max(min(R,1),0)
    # G = max(min(G,1),0)
    # B = max(min(B,1),0)
    RGB = [R, G, B]
    for i in range(0, 3):
        RGB[i] = min(int(round(RGB[i] * 255)), 255)
        RGB[i] = max(RGB[i], 0)
    return RGB


# colors size: n*3, when n is very large, it improves speed using matrix calculation
def lab2rgb_matrix(colors):
    n = len(colors)
    colors = np.array(colors)
    Ls = colors[:, 0]
    As = colors[:, 1]
    Bs = colors[:, 2]
    T1 = 0.008856
    T2 = 0.206893
    d = T2
    fys = ((Ls + 16) / 116.0) ** 3.0
    fxs = fys + As / 500.0
    fzs = fys - Bs / 200.0
    Xs = np.zeros((n), dtype="float")
    Ys = np.zeros((n), dtype="float")
    Zs = np.zeros((n), dtype="float")

    fys = np.where(fys > T1, fys, Ls / 903.3)
    Ys = fys
    fys = np.where(fys > T1, fys ** (1.0 / 3), fys * 7.787 + 16.0 / 116)

    fxs = fys + As / 500.0
    Xs = np.where(fxs > T2, fxs**3.0, (fxs - 16.0 / 116) / 7.787)

    fzs = fys - Bs / 200.0
    Zs = np.where(fzs > T2, fzs**3.0, (fzs - 16.0 / 116) / 7.787)

    Xs *= 0.95045
    Zs *= 1.08875
    Rs = 3.240479 * Xs + (-1.537150) * Ys + (-0.498535) * Zs
    Gs = (-0.969256) * Xs + 1.875992 * Ys + 0.041556 * Zs
    Bs = 0.055648 * Xs + (-0.204043) * Ys + 1.057311 * Zs
    RGBs = np.vstack((Rs, Gs, Bs)).transpose()
    RGBs = np.maximum(RGBs * 255, 0.0)
    RGBs = np.minimum(RGBs, 255.0)
    RGBs = RGBs.astype("int")
    return RGBs


def CIEDE2000(Lab_1, Lab_2):
    """
    Calculates CIEDE2000 color distance between two CIE L*a*b* colors
    Source: https://github.com/lovro-i/CIEDE2000
    """

    C_25_7 = 6103515625  # 25**7

    L1, a1, b1 = Lab_1[0], Lab_1[1], Lab_1[2]
    L2, a2, b2 = Lab_2[0], Lab_2[1], Lab_2[2]
    C1 = np.sqrt(a1**2 + b1**2)
    C2 = np.sqrt(a2**2 + b2**2)
    C_ave = (C1 + C2) / 2
    G = 0.5 * (1 - np.sqrt(C_ave**7 / (C_ave**7 + C_25_7)))

    L1_, L2_ = L1, L2
    a1_, a2_ = (1 + G) * a1, (1 + G) * a2
    b1_, b2_ = b1, b2

    C1_ = np.sqrt(a1_**2 + b1_**2)
    C2_ = np.sqrt(a2_**2 + b2_**2)

    if np.all(b1_ == 0) and np.all(a1_ == 0):
        h1_ = 0
    elif np.all(a1_ >= 0):
        h1_ = np.arctan2(b1_, a1_)
    else:
        h1_ = np.arctan2(b1_, a1_) + 2 * np.pi

    if np.all(b2_ == 0) and np.all(a2_ == 0):
        h2_ = 0
    elif np.all(a2_ >= 0):
        h2_ = np.arctan2(b2_, a2_)
    else:
        h2_ = np.arctan2(b2_, a2_) + 2 * np.pi

    dL_ = L2_ - L1_
    dC_ = C2_ - C1_
    dh_ = h2_ - h1_
    if np.all(C1_ * C2_ == 0):
        dh_ = 0
    elif np.all(dh_ > np.pi):
        dh_ -= 2 * np.pi
    elif np.all(dh_ < -np.pi):
        dh_ += 2 * np.pi
    dH_ = 2 * np.sqrt(C1_ * C2_) * np.sin(dh_ / 2)

    L_ave = (L1_ + L2_) / 2
    C_ave = (C1_ + C2_) / 2

    _dh = np.abs(h1_ - h2_)
    _sh = h1_ + h2_
    C1C2 = C1_ * C2_

    if np.all(_dh <= np.pi) and np.all(C1C2 != 0):
        h_ave = (h1_ + h2_) / 2
    elif np.all(_dh > np.pi) and np.all(_sh < 2) * np.pi and np.all(C1C2 != 0):
        h_ave = (h1_ + h2_) / 2 + np.pi
    elif np.all(_dh > np.pi) and np.all(_sh >= 2) * np.pi and np.all(C1C2 != 0):
        h_ave = (h1_ + h2_) / 2 - np.pi
    else:
        h_ave = h1_ + h2_

    T = (
        1
        - 0.17 * np.cos(h_ave - np.pi / 6)
        + 0.24 * np.cos(2 * h_ave)
        + 0.32 * np.cos(3 * h_ave + np.pi / 30)
        - 0.2 * np.cos(4 * h_ave - 63 * np.pi / 180)
    )

    h_ave_deg = h_ave * 180 / np.pi
    if np.all(h_ave_deg < 0):
        h_ave_deg += 360
    elif np.all(h_ave_deg > 360):
        h_ave_deg -= 360
    dTheta = 30 * np.exp(-(((h_ave_deg - 275) / 25) ** 2))

    R_C = 2 * np.sqrt(C_ave**7 / (C_ave**7 + C_25_7))
    S_C = 1 + 0.045 * C_ave
    S_H = 1 + 0.015 * C_ave * T

    Lm50s = (L_ave - 50) ** 2
    S_L = 1 + 0.015 * Lm50s / np.sqrt(20 + Lm50s)
    R_T = -np.sin(dTheta * np.pi / 90) * R_C

    k_L, k_C, k_H = 1, 1, 1

    f_L = dL_ / k_L / S_L
    f_C = dC_ / k_C / S_C
    f_H = dH_ / k_H / S_H

    dE_00 = np.sqrt(f_L**2 + f_C**2 + f_H**2 + R_T * f_C * f_H)
    return dE_00
