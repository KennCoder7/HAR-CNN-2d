import numpy as np


train_x = np.load('UCI_data/np_2d/np_train_x_2d.npy')
test_x = np.load('UCI_data/np_2d/np_test_x_2d.npy')


def read_array_old(segmt, n):
    d1 = np.array([], dtype=float)
    for i in range(segmt.shape[0]):
        d1 = np.append(d1, segmt[i][n])
    return d1   # d1 [x-g, y-g, z-g, x-Tacc. y-Tacc, z-Tacc, x-Lacc, y-Lacc, z-Lacc]


def read_array(segmt, n):
    return segmt[:, n].reshape(segmt.shape[0])


def array_algorithm1(arr):
    re_array_method1 = np.array([1, 3, 5, 7, 9, 2, 4, 6, 8], dtype=int)
    re_array_method2 = np.array([1, 4, 7, 1, 5, 8, 2, 5, 9], dtype=int)
    re_array_method3 = np.array([3, 6, 9, 4, 8, 3, 7, 2, 6], dtype=int)
    # new_array = np.array([], dtype=float)
    new_array1 = np.array([], dtype=float)
    new_array2 = np.array([], dtype=float)
    new_array3 = np.array([], dtype=float)
    for i in range(len(arr)):
        new_array1 = np.append(new_array1, arr[re_array_method1[i] - 1])
        new_array2 = np.append(new_array2, arr[re_array_method2[i] - 1])
        new_array3 = np.append(new_array3, arr[re_array_method3[i] - 1])
    new_array = np.append(arr, new_array1)
    new_array = np.append(new_array, new_array2)
    new_array = np.append(new_array, new_array3)
    return new_array    # 123456789 135792468 147158259 369483726


def seg_algorithm1(data):
    new_arr1 = array_algorithm1(read_array(data, 0))
    for j in range(data.shape[1]):
        if j != 0:
            new_arr1 = np.vstack((new_arr1, array_algorithm1(read_array(data, j))))
    return new_arr1.transpose().reshape((new_arr1.shape[1], new_arr1.shape[0], 1))  # (36, 128, 1)


def dataset_algorithm1(data, proc_name):
    new_data = seg_algorithm1(data[0]).reshape((1, 36, 128, 1))
    n = 0
    for i in range(data.shape[0]):
        if i != 0:
            new_data = np.vstack((new_data, seg_algorithm1(data[i]).reshape((1, 36, 128, 1))))
        if i - n > 0.05 * data.shape[0]:
            n = i
            print("### Process --- (", proc_name, "_ data ) Algorithm1 --- In progress --- [ ",
                  100 * round(i / data.shape[0], 2), "% ] Finished ###")
    return new_data


# train_x_algorithm1 = dataset_algorithm1(train_x, "train")
# np.save("./UCI_data/np_2d/np_train_x_algorithm1.npy", train_x_algorithm1)
# print(train_x_algorithm1.shape)    # (7352, 36, 128, 1)
# test_x_algorithm1 = dataset_algorithm1(test_x, "test")
# np.save("./UCI_data/np_2d/np_test_x_algorithm1.npy", test_x_algorithm1)
# print(test_x_algorithm1.shape)     # (2947, 36, 128, 1)


# test = test_x
# test_d = test[0]
# # print(test_d.shape)     # (9, 128, 1)
# # print(test[0:2].shape)    # (2, 9, 128, 1)
# # print(test_d[0:2].shape)    # (2, 128, 1)
# # print(test_d[:, 0].reshape(9).shape)    # (9,)
# print(read_array(test_d, 0))
# print(read_array_old(test_d, 0))
