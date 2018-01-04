# coding=utf-8
import preprocess
import matplotlib.image as mpimg
from scipy import misc
import numpy as np
import z_e_step
import m_step
import pickle
import os


def inverse_texture_synthesis(image, window_size, sp, xc, cp, zm, zn):
    #  sp, xc, cp 具体和 paper 中的描述一致, zm、zn 是输出纹理的宽和高
    xm, xn, c = image.shape
    #  zp shape (xm, xn, 2)  表示原始图像中的每一个 window 到输出图像中的 window 的最佳映射
    #  xq shape (zm, zn, 2)  表示输出图像中的每一个 window 到原始图像中的 window 的最佳映射
    zp = np.zeros((xm, xn, 2)).astype(int)
    xq = np.zeros((zm, zn, 2)).astype(int)
    half_w = (window_size - 1) / 2
    zp[:, :, 0] = np.random.randint(low=half_w, high=zm - half_w, size=(xm, xn)).astype(int)
    zp[:, :, 1] = np.random.randint(low=half_w, high=zn - half_w, size=(xm, xn)).astype(int)
    xq[:, :, 0] = np.random.randint(low=half_w, high=xm - half_w, size=(zm, zn)).astype(int)
    xq[:, :, 1] = np.random.randint(low=half_w, high=xn - half_w, size=(zm, zn)).astype(int)

    Z = image[0:zm, 0:zn, :].copy()  # Z 的初始化

    iter_num = 10
    for it in range(iter_num):
        print("Iteration %d" % (it + 1))
        z_e_step.z_e_step(image, zm, zn, window_size, xq, zp, sp, Z)
        m_step.inverse_m_step(image, Z, window_size, xc, zp, cp)
        m_step.forward_m_step(image, Z, window_size, xq, sp)

    return Z


if __name__ == '__main__':
    picture_path = '/home/sk/Desktop/TextureSynthesis-master/Image/Texture-01.png'
    Xc_save_path = '/home/sk/Desktop/its/res/xc'
    img = mpimg.imread(picture_path)
    window_size = 7
    zm = 10
    zn = 10
    sp_size = 5
    # xc, cp = preprocess.get_xc_and_cp(img, window_size, zm, True, Xc_save_path)
    # sp = preprocess.k_coherence_similarity_set(img, window_size, sp_size)
    # xc_file = open('xc', 'w')
    # cp_file = open('cp', 'w')
    # sp_file = open('sp', 'w')
    #
    # pickle.dump(xc, xc_file)
    # pickle.dump(cp, cp_file)
    # pickle.dump(sp, sp_file)

    xc = pickle.load(open('xc'))
    cp = pickle.load(open('cp'))
    sp = pickle.load(open('sp'))
    Z = inverse_texture_synthesis(img, window_size, sp, xc, cp, zm, zn)
    misc.imsave(os.path.join('/home/sk/Desktop/its/res/output', 'res.png'), Z)
