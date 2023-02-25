import scipy.ndimage as ndimage
import re
import os
import cv2
import numpy as np

def GuidedfilterRGB(I, p, omega=32, eps=0.01):
    w_size = (omega, omega)
    I = I / 255
    I_r, I_g, I_b = I[:, :, 0], I[:, :, 1], I[:, :, 2]

    mean_I_r = cv2.blur(I_r, w_size)
    mean_I_g = cv2.blur(I_g, w_size)
    mean_I_b = cv2.blur(I_b, w_size)

    mean_p = cv2.blur(p, w_size)

    mean_Ip_r = cv2.blur(I_r * p, w_size)
    mean_Ip_g = cv2.blur(I_g * p, w_size)
    mean_Ip_b = cv2.blur(I_b * p, w_size)

    cov_Ip_r = mean_Ip_r - mean_I_r * mean_p
    cov_Ip_g = mean_Ip_g - mean_I_g * mean_p
    cov_Ip_b = mean_Ip_b - mean_I_b * mean_p
    cov_Ip = np.stack([cov_Ip_r, cov_Ip_g, cov_Ip_b], axis=-1)

    var_I_rr = cv2.blur(I_r * I_r, w_size) - mean_I_r * mean_I_r
    var_I_rg = cv2.blur(I_r * I_g, w_size) - mean_I_r * mean_I_g
    var_I_rb = cv2.blur(I_r * I_b, w_size) - mean_I_r * mean_I_b
    var_I_gb = cv2.blur(I_g * I_b, w_size) - mean_I_g * mean_I_b
    var_I_gg = cv2.blur(I_g * I_g, w_size) - mean_I_g * mean_I_g
    var_I_bb = cv2.blur(I_b * I_b, w_size) - mean_I_b * mean_I_b

    a = np.zeros(I.shape)
    for x, y in np.ndindex(I.shape[:2]):
        Sigma = np.array([
            [var_I_rr[x, y], var_I_rg[x, y], var_I_rb[x, y]],
            [var_I_rg[x, y], var_I_gg[x, y], var_I_gb[x, y]],
            [var_I_rb[x, y], var_I_gb[x, y], var_I_bb[x, y]]
        ])
        c = cov_Ip[x, y, :]

        a[x, y, :] = np.linalg.inv(Sigma + eps * np.eye(3)).dot(c)

    mean_a = np.stack([cv2.blur(a[:, :, 0], w_size), cv2.blur(a[:, :, 1], w_size), cv2.blur(a[:, :, 2], w_size)],
                      axis=-1)
    mean_I = np.stack([mean_I_r, mean_I_g, mean_I_b], axis=-1)

    b = mean_p - np.sum(a * mean_I, axis=2)
    mean_b = cv2.blur(b, w_size)
    q = np.sum(mean_a * I, axis=2) + mean_b
    return q

def GuidedfilterGRAY(im, p, r, eps):
    mean_I = cv2.boxFilter(im, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    mean_Ip = cv2.boxFilter(im * p, cv2.CV_64F, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(im * im, cv2.CV_64F, (r, r))
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))

    q = mean_a * im + mean_b
    return q

def DSP(img, sz):
    img = img.astype(np.int32)
    sum_filter = ndimage.generic_filter(img, Func_sum, footprint=np.ones((sz, sz, 3)), mode='nearest')
    sum_filter = sum_filter / np.max(sum_filter)
    return sum_filter

def Func_sum(a):
    return np.sum(a)

def TransmissionEstimate(im, A, sz):
    im_min0 = im.min(axis=2)
    A_min0 = A.max(axis=2)
    im_min = ndimage.minimum_filter(im_min0, footprint=np.ones((sz, sz)), mode='nearest')
    A_min = ndimage.minimum_filter(A_min0, footprint=np.ones((sz, sz)), mode='nearest')
    t = 1 - ((im_min) / (A_min))
    return t

def TransmissionRefine(im, et):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    gray = np.float64(gray) / 255
    r = 32
    eps = 0.05
    t = GuidedfilterGRAY(gray, et, r, eps)
    return t


def Recover(im, t, A, tx=0.1):
    res = np.empty(im.shape, im.dtype)
    t = cv2.max(t, tx)
    for ind in range(0, 3):
        res[:, :, ind] = (im[:, :, ind] - A[:, :, ind]) / t + A[:, :, ind]
    return res


def atof(text):
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    float regex comes from https://stackoverflow.com/a/12643073/190597
    '''
    return [atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text)]


count = 0
folder_with = "Input Folder"
frame = []
for subdirw, dirsw, filesw in os.walk(folder_with):
    filesw.sort(key=natural_keys)
count = 1

for file in filesw:
    img0 = os.path.join(subdirw, file)
    src = cv2.imread(img0)
    A = DSP(src, 15)

    r = 32
    eps = 0.05
    newa1 = GuidedfilterRGB(src, A[:, :, 0], r, eps)
    newa2 = GuidedfilterRGB(src, A[:, :, 1], r, eps)
    newa3 = GuidedfilterRGB(src, A[:, :, 2], r, eps)
    newA = cv2.merge((newa1, newa2, newa3))

    I = src.astype('float64') / 255

    te = TransmissionEstimate(I, newA, 15)
    t = TransmissionRefine(src, te)
    J = Recover(I, t, newA, 0.1)
    print(count)
    cv2.imwrite('Output Folder/img (' + str(count) + ').png',J*255)
    count += 1

