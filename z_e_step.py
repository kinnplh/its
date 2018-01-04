# coding=utf-8
import numpy as np


def get_k_coherence_set(mz, nz, window_size, xq, sp):
    half_w = (window_size - 1) / 2
    offset = np.zeros((window_size, window_size, 2))
    offset[:, :, 0] = np.diag(range(- half_w, half_w + 1)).dot(np.ones((window_size, window_size)))
    offset[:, :, 1] = np.ones((window_size, window_size)).dot(np.diag(range(- half_w, half_w + 1)))

    kcs = []
    for i in range(mz):
        last_list = list()
        kcs.append(last_list)
        for j in range(nz):
            last_list.append([])

    for i in range(half_w, mz - half_w):
        for j in range(half_w, nz - half_w):
            related_windows_ori = xq[i - half_w: i + half_w + 1, j - half_w: j + half_w + 1, :]
            # 以周围的像素为中心的所有的 region 对应的 region 都在上面了
            related_windows = related_windows_ori - offset
            # 上面找到的是  在原始图像中,和像素 i,j 相似的 region
            # 然后再在之前构建的 sp 中进行查找就可以了
            crt_set = sp[related_windows[:, :, 0].reshape(window_size * window_size).astype(int), related_windows[:, :,
                                                                                                  1].reshape(
                window_size * window_size).astype(int)].astype(int)
            extended_list = [item for sublist in crt_set.tolist() for item in sublist]
            kcs[i][j] = list(set([tuple(z) for z in extended_list]))
            if np.min(kcs[i][j]) < 3:
                print 'something error'
    return kcs


def z_e_step(image, mz, nz, window_size, xq, zp, sp, Z):
    # xq 输出图像中的每个 window,对应其在输入图像中的每个 window
    # sp 对于原始图像中的每一个 window, 找到若干个在原始图像中与之最相近的 window
    # 这个函数对应的仅仅是一次迭代
    half_w = (window_size - 1) / 2
    mx, nx, c = image.shape
    kcs = get_k_coherence_set(mz, nz, window_size, xq, sp)  # 这一轮迭代的每个在 Z 中的像素的所有的可能的取值(以在原图的坐标表示
    alpha = 0.01
    sample_rate_forward = 1
    sample_rate_inverse = 1
    weight_forward = float(sample_rate_forward ** 2) * alpha / float(mz * nz)
    weight_inverse = float(sample_rate_inverse ** 2) / float(mx * nx)

    # 对于每一个在输出图像中的像素,找到其所有的相似的像素在原图中
    # 具体操作:  在输出文件中,每一个像素有 window_size * window_size 个 window包含这个像素,那么在原始图像中,也同样的有这么多个 window
    # 具体操作:  那么根据 offset,就可以找到 window_size * window_size 个像素与之对应,那么就可以取出同样多的 window,用于后续第二部分计算
    # 第二部分的计算,就是对于一个在 Z 中的window,找到在 X 中所有对应的 window,要求距离之和最小
    pixel_in_z_to_all_in_x_forward = []
    for i in range(mz):
        last_list = list()
        pixel_in_z_to_all_in_x_forward.append(last_list)
        for j in range(nz):
            last_list.append([])

    for i_for_z in range(half_w, mz - half_w, sample_rate_forward):
        for j_for_z in range(half_w, nz - half_w, sample_rate_forward):
            # i_for_z, j_for_z 就是当前考虑的在 z 中的window
            crt_map_in_x = xq[i_for_z, j_for_z]
            # 对于原始 window 中的每个像素,我们都找到了新的映射
            for i_offset in range(-half_w, half_w + 1):
                for j_offset in range(-half_w, half_w + 1):
                    # 对于在输出图像中的像素 (i_for_z + i_offset, j_for_z + j_offset), 在输入图像中找到了一个对应
                    pixel_in_z_to_all_in_x_forward[i_for_z + i_offset][j_for_z + j_offset].append(
                        tuple(crt_map_in_x + [i_offset, j_offset]))

    #  从反方向寻找一个对应关系
    pixel_in_z_to_all_in_x_inverse = []
    for i in range(mz):
        last_list = list()
        pixel_in_z_to_all_in_x_inverse.append(last_list)
        for j in range(nz):
            last_list.append([])

    for i_for_x in range(half_w, mx - half_w, sample_rate_inverse):
        for j_for_x in range(half_w, nx - half_w, sample_rate_inverse):
            # i_for_x, j_for_x 就是当前考虑的 在 x 中的 window
            crt_map_in_z = zp[i_for_x, j_for_x]
            # 对于 z 中的 window 中的每一个像素,找到了在 x 中的一个对应
            for i_offset in range(- half_w, half_w + 1):
                for j_offset in range(- half_w, half_w + 1):
                    pixel_in_z_to_all_in_x_inverse[crt_map_in_z[0] + i_offset][crt_map_in_z[1] + j_offset]\
                        .append((i_for_x + i_offset, j_for_x + j_offset))

    for i_for_z in range(half_w, mz - half_w):
        for j_for_z in range(half_w, nz - half_w):
            print i_for_z, j_for_z
            # 对于 Z 中的每一个项目,尝试进行优化
            # 优化方案:尝试在 kcs 中寻找能够使得 energy 不断下降的点,并用其值来不断更新 Z 中的值
            energy = 10e10

            forward_compare_pos = pixel_in_z_to_all_in_x_forward[i_for_z][j_for_z]
            inverse_compare_pos = pixel_in_z_to_all_in_x_inverse[i_for_z][j_for_z]
            candidates = kcs[i_for_z][j_for_z]

            forward_compare_value = [tuple(image[pos[0], pos[1]]) for pos in forward_compare_pos]
            inverse_compare_value = [tuple(image[pos[0], pos[1]]) for pos in inverse_compare_pos]
            candidates_value = [tuple(image[pos[0], pos[1]]) for pos in candidates]

            for candidate_index, one_candidate_value in enumerate(candidates_value):
                sub_energy_forward = 0
                for one_forward_value in forward_compare_value:
                    value_diff = list(map(lambda x: x[0] - x[1], zip(one_candidate_value, one_forward_value)))
                    value_diff_pow2 = map(lambda x: x * x, value_diff)
                    value_diff_pow2_sum = reduce(lambda x, y: x + y, value_diff_pow2)
                    sub_energy_forward += value_diff_pow2_sum

                sub_energy_forward *= weight_forward

                sub_energy_inverse = 0
                for one_inverse_value in inverse_compare_value:
                    value_diff = list(map(lambda x: x[0] - x[1], zip(one_candidate_value, one_inverse_value)))
                    value_diff_pow2 = map(lambda x: x * x, value_diff)
                    value_diff_pow2_sum = reduce(lambda x, y: x + y, value_diff_pow2)
                    sub_energy_inverse += value_diff_pow2_sum

                sub_energy_inverse *= weight_inverse
                total_energy = sub_energy_forward + sub_energy_inverse

                if total_energy < energy:
                    energy = total_energy
                    # 对 Z 以及 Xq 进行更新
                    Z[i_for_z, j_for_z] = one_candidate_value
                    # 这个 window 的最佳映射当然应该是数据来源对应的那个了
                    xq[i_for_z, j_for_z] = candidates[candidate_index]

