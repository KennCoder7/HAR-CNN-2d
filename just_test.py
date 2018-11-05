import numpy as np
# import time
import itertools as it

test = [1, 2, 3]
test0 = it.permutations(test, 3)
print("print test0: ", test0)
# print test0:  <itertools.permutations object at 0x000001B87F78FC50>
test1 = list(test0)
print("print test1: ", test1)
# print test1:  [(1, 2, 3), (1, 3, 2), (2, 1, 3), (2, 3, 1), (3, 1, 2), (3, 2, 1)]
test2 = np.array(test1)
print("print test2: ", test2.shape, test2)
# print test2:  (6, 3) [[1 2 3]
#  [1 3 2]
#  [2 1 3]
#  [2 3 1]
#  [3 1 2]
#  [3 2 1]]
test3 = list(test2)
print("print test3: ", test3)
# print test3:  [array([1, 2, 3]), array([1, 3, 2]), array([2, 1, 3]),
# array([2, 3, 1]), array([3, 1, 2]), array([3, 2, 1])]
# test = np.array([1, 2, 3], dtype=int)
# test1 = it.permutations(test, 3)
# # print(list(test1))  # [(1, 2, 3), (1, 3, 2), (2, 1, 3), (2, 3, 1), (3, 1, 2), (3, 2, 1)]
# # test2 = list(test1)
# # print(test2)    # [(1, 2, 3), (1, 3, 2), (2, 1, 3), (2, 3, 1), (3, 1, 2), (3, 2, 1)]
# print(np.array(list(it.permutations(test, 3))))


# def permu_array(arr):
#     re_array_method = np.array(list(it.permutations([1, 2, 3])))
#     out_array = arr
#     new_array = np.array([], dtype=float)
#     print(new_array.shape)
#     for n in range(re_array_method.shape[0]):
#         for i in range(len(arr)):
#             if n != 0:
#                 new_array = np.append(new_array, arr[re_array_method[n][i] - 1])
#     out_array = np.append(out_array, new_array)
#     return out_array  # [1. 2. 3. 1. 3. 2. 2. 1. 3. 2. 3. 1. 3. 1. 2. 3. 2. 1.]

# def permu_array(arr):
#     re_array_method1 = np.array([1, 3, 2], dtype=int)
#     re_array_method2 = np.array([1, 1, 2], dtype=int)
#     re_array_method3 = np.array([3, 3, 2], dtype=int)
#     # new_array = np.array([], dtype=float)
#     new_array1 = np.array([], dtype=float)
#     new_array2 = np.array([], dtype=float)
#     new_array3 = np.array([], dtype=float)
#     for i in range(len(arr)):
#         new_array1 = np.append(new_array1, arr[re_array_method1[i] - 1])
#         new_array2 = np.append(new_array2, arr[re_array_method2[i] - 1])
#         new_array3 = np.append(new_array3, arr[re_array_method3[i] - 1])
#     new_array = np.append(arr, new_array1)
#     new_array = np.append(new_array, new_array2)
#     new_array = np.append(new_array, new_array3)
#     return new_array  # 123 132 112 332


# print(permu_array([1, 2, 3]))

# re_array_method1 = np.array([1, 3, 2], dtype=int)
# re_array_method2 = np.array([1, 1, 2], dtype=int)
# re_array_method3 = np.array([3, 3, 2], dtype=int)
# test = np.array(([1, 3, 2], [1, 1, 2], [3, 3, 2], [1, 1, 1]), dtype=int)
# print(re_array_method1.shape)   # (3,)
# print(test.shape)   # (4, 3)
# print(test)
# # [[1 3 2]
# #  [1 1 2]
# #  [3 3 2]
# #  [1 1 1]]
# print(test[0].shape)    # (3,)
# for i in range(test.shape[0]):
#     print(test[i])
#     # [1 3 2]
#     # [1 1 2]
#     # [3 3 2]
#     # [1 1 1]
# initial_time = time.time()
# print(initial_time)
# print(time.localtime(initial_time))
# print(time.asctime(time.localtime(initial_time)))
#
# train_x = np.load('UCI_data/np_2d/np_train_x_2d.npy')
# train_y = np.load('UCI_data/np_2d/np_train_y_2d.npy')
# test_x = np.load('UCI_data/np_2d/np_test_x_2d.npy')
# test_y = np.load('UCI_data/np_2d/np_test_y_2d.npy')
#
# # d = test_x[0]
#
# # print(d.shape)
#
#
# def read_array(segmt, n):
#     d1 = np.array([], dtype=float)
#     for i in range(segmt.shape[0]):
#         d1 = np.append(d1, segmt[i][n])
#     return d1   # d1 [x-g, y-g, z-g, x-Tacc. y-Tacc, z-Tacc, x-Lacc, y-Lacc, z-Lacc]
#
#
# def array_fake_dft(arr):
#     re_array_method1 = np.array([1, 3, 5, 7, 9, 2, 4, 6, 8], dtype=int)
#     re_array_method2 = np.array([1, 4, 7, 1, 5, 8, 2, 5, 9], dtype=int)
#     re_array_method3 = np.array([3, 6, 9, 4, 8, 3, 7, 2, 6], dtype=int)
#     new_array = np.array([], dtype=float)
#     new_array1 = np.array([], dtype=float)
#     new_array2 = np.array([], dtype=float)
#     new_array3 = np.array([], dtype=float)
#     for i in range(len(arr)):
#         new_array1 = np.append(new_array1, arr[re_array_method1[i] - 1])
#         new_array2 = np.append(new_array2, arr[re_array_method2[i] - 1])
#         new_array3 = np.append(new_array3, arr[re_array_method3[i] - 1])
#     new_array = np.append(arr, new_array1)
#     new_array = np.append(new_array, new_array2)
#     new_array = np.append(new_array, new_array3)
#     return new_array
#
#
# def seg_dft(data):
#     new_arr1 = array_fake_dft(read_array(data, 0))
#     for j in range(data.shape[1]):
#         if j != 0:
#             new_arr1 = np.vstack((new_arr1, array_fake_dft(read_array(data, j))))
#     return new_arr1.transpose().reshape((new_arr1.shape[1], new_arr1.shape[0], 1))
#
#
# def dataset_dft(data, proc_name):
#     new_data = seg_dft(data[0]).reshape((1, 36, 128, 1))
#     n = 0
#     for i in range(data.shape[0]):
#         if i != 0:
#             new_data = np.vstack((new_data, seg_dft(data[i]).reshape((1, 36, 128, 1))))
#         if i - n > 0.05 * data.shape[0]:
#             n = i
#             print("### Process(", proc_name, ") --- DFT --- In progress --- [ ",
#                   100 * round(i / data.shape[0], 2), "% ] Finished ###")
#     return new_data

# test = d
# print("### test shape: ", test.shape)
# print("### read_array(d, 0): ", read_array(d, 0).shape)
# print("### array_fake_dft(read_array(d, 0)): ", array_fake_dft(read_array(d, 0)).shape)
#
# new_arr1 = array_fake_dft(read_array(test, 0))
# for j in range(test.shape[1]):
#     if j != 0:
#         new_arr1 = np.vstack((new_arr1, array_fake_dft(read_array(test, j))))
# print("### new_arr1 shape: ", new_arr1.shape)
# print("### new_arr1 reshape: ", new_arr1.transpose().reshape((new_arr1.shape[1], new_arr1.shape[0], 1)).shape)
# print(test[0].shape)
# print("######################################")
# print(new_arr1.transpose().reshape((new_arr1.shape[1], new_arr1.shape[0], 1))[0].shape)
# print(seg_dft(d).shape)

# 2018/11/3
# (9, 128, 1)
# ### test shape:  (9, 128, 1)
# ### read_array(d, 0):  (9,)
# ### array_fake_dft(read_array(d, 0)):  (36,)
# ### new_arr1 shape:  (128, 36)
# ### new_arr1 reshape:  (36, 128, 1)
# (128, 1)
# ######################################
# (128, 1)
# (36, 128, 1)

# test1 = test_x[0]
# test2 = test_x[1]
# print("debug: ", test1.shape, test2.shape)  # debug:  (9, 128, 1) (9, 128, 1)
# print(np.vstack((test1, test2)).shape)  # (18, 128, 1)
# test_arr = np.array(test1.shape)
# print(test_arr) # [  9 128   1]

# print(test1.reshape((1, 9, 128, 1)).shape)  # (1, 9, 128, 1)
# print(test_x.shape)  # (2947, 9, 128, 1)
# print(np.vstack((test1.reshape((1, 9, 128, 1)), test2.reshape((1, 9, 128, 1)))).shape)  # (2, 9, 128, 1)

# data = test_x[0:3]
# # print(data.shape)   # (3, 9, 128, 1)
# # print(data.shape[0])    # 3
# new_data = seg_dft(data[0]).reshape((1, 36, 128, 1))
# for i in range(data.shape[0]):
#     if i != 0:
#         new_data = np.vstack((new_data, seg_dft(data[i]).reshape((1, 36, 128, 1))))
# print(new_data.shape)   # (3, 36, 128, 1)

# data = test_x
# print(data.shape)   # (3, 9, 128, 1)
# print(data.shape[0])    # 3
# new_data = seg_dft(data[0]).reshape((1, 36, 128, 1))
# n = 0
# for i in range(data.shape[0]):
#     if i != 0:
#         new_data = np.vstack((new_data, seg_dft(data[i]).reshape((1, 36, 128, 1))))
#     if i - n > 0.1 * data.shape[0]:
#         n = i
#         print("### Process --- DFT --- In progress --- [ ",
#               100 * round(i / data.shape[0], 2), "% ] Finished ###")
# print(new_data.shape)   # (2947, 36, 128, 1)

# data = test_x[0]
# print(data.shape)
# print(data.reshape(9, 68, 1))     # ValueError: cannot reshape array of size 1152 into shape (9,68,1)

# print(eval(test_x))  # error

# train_x = np.load('UCI_data/np_2d/np_train_x_algorithm1.npy')
# test_x = np.load('UCI_data/np_2d/np_test_x_algorithm1.npy')

# print(train_x.shape)


# train_2d = train_x[0]
# print(train_2d.shape)   # (36, 128, 1)
# print(np.fft.fftn(train_2d).shape)  # (36, 128, 1)
# print(train_2d[0].shape)
# print(np.fft.fftn(train_2d)[0].shape)
# print(train_2d[0])
# print(np.fft.fftn(train_2d)[0])