import numpy as np

train_x = np.load('ADL_data/np_2d_new/np_data_2d_v1.npy')


def read_array(segmt, n):
    return segmt[:, n].reshape(segmt.shape[0])


# print(read_array(train_x[0], 0).shape)  # (3,)


def permu_array_old(arr):
    re_array_method1 = np.array([1, 3, 2], dtype=int)
    re_array_method2 = np.array([1, 1, 2], dtype=int)
    re_array_method3 = np.array([3, 3, 2], dtype=int)
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
    return new_array  # 123 132 112 332


def permu_array(arr):
    re_array_method = ([1, 3, 2], [1, 1, 2], [3, 3, 2])
    out_array = arr
    new_array = np.array([], dtype=float)
    print(new_array.shape)
    for n in range(3):
        for i in range(len(arr)):
            new_array = np.append(new_array, arr[re_array_method[n][i] - 1])
    out_array = np.append(out_array, new_array)
    return out_array  # 123 132 112 332


def permu_seg(data):
    new_arr1 = permu_array(read_array(data, 0))
    for j in range(data.shape[1]):
        if j != 0:
            new_arr1 = np.vstack((new_arr1, permu_array(read_array(data, j))))
    return new_arr1.transpose().reshape((new_arr1.shape[1], new_arr1.shape[0], 1))  # (36, 128, 1)


# print(permu_seg(train_x[0]).shape)  # (12, 68, 1)


def permu_dataset(data, proc_name):
    new_data = permu_seg(data[0]).reshape((1, permu_seg(data[0]).shape[0], permu_seg(data[0]).shape[1], 1))
    n = 0
    for i in range(data.shape[0]):
        if i != 0:
            new_data = np.vstack((new_data, permu_seg(data[i]).
                                  reshape((1, permu_seg(data[0]).shape[0], permu_seg(data[0]).shape[1], 1))))
        if i - n > 0.05 * data.shape[0]:
            n = i
            print("### Process --- (", proc_name, "_ data ) Permutation --- In progress --- [ ",
                  int(100 * round(i / data.shape[0], 2)), "% ] Finished ###")
    return new_data


train_x_algorithm1 = permu_dataset(train_x, "train")
np.save("ADL_data/np_2d_new/np_data_2d_permu.npy", train_x_algorithm1)
print(train_x_algorithm1.shape)  # (7055, 12, 68, 1)
