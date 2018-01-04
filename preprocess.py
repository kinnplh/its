# coding=utf-8
from sklearn.cluster import KMeans
from scipy import misc
import os.path
import numpy as np
import heapq
import scipy.spatial


def get_xc_and_cp(image, window_size, z_size, is_temp_res_save, save_path):
    cp_size_per_window = 1
    m, n, c = image.shape  # 长,宽和通道数量
    size_per_window = window_size * window_size * c
    half_w = (window_size - 1) / 2
    all_window_info = np.array(
        [image[x: x + window_size, y: y + window_size].reshape(size_per_window) for x in range(m - window_size + 1) for
         y in range(n - window_size + 1)])  # 每行实际上对应于一个窗口中的所有的值
    index_image_cor_map = np.array([[x + half_w, y + half_w] for x in range(m - window_size + 1) for y in
                                    range(n - window_size + 1)])  # 用来计算上面的矩阵中第 n 行在原始图像中的实际的坐标
    class_size = z_size * z_size
    kmeans_res = KMeans(n_clusters=class_size).fit(all_window_info)  # 聚类的数量和输出的结果(Z)的大小相似
    #  Xc 这个集合是聚类的中心,但是不能够是上面计算出来的中心,因为上面计算出来的中心不一定在图片中存在。对于每一类,找到距离其类重心最近的那个样本
    Xc = np.zeros((class_size, 2))  # 表示 Xc 元素在原始的图像中的坐标(以中心坐标来表示这个window坐标)
    all_class_centers = np.zeros((class_size, size_per_window))
    for i in range(class_size):
        crt_center = kmeans_res.cluster_centers_[i]
        all_window_dis = np.sum((all_window_info - crt_center) ** 2, 1)
        min_index = np.argmin(all_window_dis)

        if kmeans_res.labels_[min_index] != i:
            print ("错误。离类中心最近的一个元素竟然不属于这个类")

        all_class_centers[i, :] = all_window_info[min_index, :]
        index_ori = index_image_cor_map[min_index, :]
        Xc[i, :] = index_ori
        if is_temp_res_save:
            misc.imsave(os.path.join(save_path, '%d.png' % i),
                        image[index_ori[0] - half_w: index_ori[0] + half_w + 1,
                        index_ori[1] - half_w: index_ori[1] + half_w + 1])
    cp = np.zeros((m, n, cp_size_per_window))
    # 对于每一个 window,在 Xc 中找到 cp_window_size 个最相似的 window。变量 cp 中记录的是在 Xc 中的索引
    count_strange = 1
    for i in range((m - window_size + 1) * (n - window_size + 1)):
        crt_window = all_window_info[i, :]
        crt_window_to_all_centers = np.sum((all_class_centers - crt_window) ** 2, 1)
        assert len(crt_window_to_all_centers) == class_size
        res = heapq.nsmallest(cp_size_per_window, xrange(len(crt_window_to_all_centers)),
                              crt_window_to_all_centers.take)

        if res[0] != kmeans_res.labels_[i]:
            print ('奇怪。不属于离自己最近的那个类中心所对应的类 %d/%d' % (count_strange, i + 1))
            count_strange += 1
        index_ori = index_image_cor_map[i, :]
        cp[index_ori[0], index_ori[1]] = res
    return Xc, cp


def k_coherence_similarity_set(image, window_size, sp_size):
    #  对于每一个window,返回前 sp_size 个与之最为接近的 window
    #  在原文中实际上是要找 对于每一个像素 p,找到那些邻居中存在与 p 接近的像素的像素
    #  这里的 window 理论上来说应该是以中心点作为坐标的,但是这里还是以左上角作为坐标,在后续使用的时候需要谨慎
    #  使用建立 kd tree 的方法来查找最近的那些 window
    #  返回的结果: 每个像素对应于一个 sp_size, 2 的矩阵
    m, n, c = image.shape  # 长,宽和通道数量
    half_w = (window_size - 1) / 2
    size_per_window = window_size * window_size * c
    all_window_info = np.array(
        [image[x: x + window_size, y: y + window_size].reshape(size_per_window) for x in range(m - window_size + 1) for
         y in range(n - window_size + 1)])  # 每行实际上对应于一个窗口中的所有的值
    index_image_cor_map = np.array([[x + half_w, y + half_w] for x in range(m - window_size + 1) for y in
                                    range(n - window_size + 1)])  # 用来计算上面的矩阵中第 n 行在原始图像中的实际的坐标

    all_window_tree = scipy.spatial.KDTree(all_window_info)  # 文档中要求最后一维的长度是数据的维数  也就是说每行是一个数据

    sp = np.zeros((m, n, sp_size, 2))
    for i in range((m - window_size + 1) * (n - window_size + 1)):
        print "%d/%d" % (i + 1, (m - window_size + 1) * (n - window_size + 1))
        crt_window = all_window_info[i, :]
        res_d, res_i = all_window_tree.query(crt_window, sp_size)
        assert res_i[0] == i
        index_ori = index_image_cor_map[i, :]
        sp[index_ori[0], index_ori[1]] = index_image_cor_map[res_i, :]
    return sp
