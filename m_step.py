# coding=utf-8
import numpy as np
import scipy.spatial
import z_e_step


def forward_m_step(image, Z, window_size, xq, sp):
    # 完成对 xq 的更新: 对于输出图像中的所有的 window, 找到在输入图像中与之最为匹配的 window
    # 与 e-step 类似的,候选的元素都在 k coherence set 里面
    # 在候选的元素里面找到一个最接近的元素即可

    mz, nz, c = Z.shape
    value_num_one_window = window_size * window_size * c
    half_w = (window_size - 1) / 2
    khs = z_e_step.get_k_coherence_set(mz, nz, window_size, xq, sp)
    for i_for_z in range(half_w, mz - half_w):
        for j_for_z in range(half_w, nz - half_w):
            crt_window = Z[i_for_z - half_w: i_for_z + half_w + 1, j_for_z - half_w: j_for_z + half_w + 1].reshape(value_num_one_window)
            candidates = khs[i_for_z][j_for_z]
            candidate_values = [
                image[pos[0] - half_w: pos[0] + half_w + 1, pos[1] - half_w: pos[1] + half_w + 1].reshape(
                    value_num_one_window) for pos in candidates]

            energy = 10e10
            for candidate_index, one_candidate_value in enumerate(candidate_values):
                # 计算 candidate 和当前的 window 的差异,差异较小的话则进行更新
                e = np.sum((crt_window - one_candidate_value) ** 2)
                if e < energy:
                    energy = e
                    xq[i_for_z, j_for_z] = candidates[candidate_index]


    pass


def inverse_m_step(image, Z, window_size, xc, zp, cp):
    # 完成对 zp 的更新:对于输入图像中的所有的 window,找到在输出图像中的与之最匹配的 window
    # 对于那些在 Xc 中的 window,直接在 Z 中进行最近邻匹配
    # 对于那些不在 Xc 中的 window, cp 已经定义了这些 window 在 Xc 中的最佳的映射

    # 对 Z 建立 kd tree进行最近邻搜索
    half_w = (window_size - 1) / 2
    mz, nz, c = Z.shape
    mx, nx, cx = image.shape
    value_num_in_one_window = window_size * window_size * c
    all_windows_in_z = np.array([Z[i: i + window_size, j: j + window_size].reshape(value_num_in_one_window)
                                 for i in range(mz - window_size + 1) for j in range(nz - window_size + 1)])
    z_window_index_coor = np.array([[i + half_w, j + half_w]
                                    for i in range(mz - window_size + 1) for j in range(nz - window_size + 1)])

    all_z_window_tree = scipy.spatial.KDTree(all_windows_in_z)
    xc_size = xc.shape[0]
    zp_value_for_xc = np.zeros((xc_size, 2))
    for xc_index in range(xc_size):
        i_xc = xc[xc_index, 0]
        j_xc = xc[xc_index, 1]
        crt_xc_window = image[i_xc - half_w: i_xc + half_w + 1, j_xc - half_w: j_xc + half_w + 1] \
            .reshape(value_num_in_one_window)
        res_d, res_i = all_z_window_tree.query(crt_xc_window)
        nearest_coor = z_window_index_coor[res_i]
        zp_value_for_xc[xc_index] = nearest_coor
        zp[i_xc, j_xc] = nearest_coor

    # cp 内记录的是在 xc 中的 index
    for i_for_x in range(half_w, mx - half_w):
        for j_for_x in range(half_w, nx - half_w):
            zp[i_for_x, j_for_x] = zp_value_for_xc[cp[i_for_x, j_for_x, 0]]
