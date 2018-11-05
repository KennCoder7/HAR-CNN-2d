import numpy as np
import itertools as it

train_x = np.load('ADL_data/np_2d_new/np_data_2d_v1.npy')


def read_array(segmt, n):
    return segmt[:, n].reshape(segmt.shape[0])


# print(read_array(train_x[0], 0).shape)  # (3,)


def permu_array(arr):
    re_array_method = np.array(list(it.permutations([1, 2, 3])))
    out_array = arr
    new_array = np.array([], dtype=float)
    for n in range(re_array_method.shape[0]):
        for i in range(len(arr)):
            if n != 0:
                new_array = np.append(new_array, arr[re_array_method[n][i] - 1])
    out_array = np.append(out_array, new_array)
    return out_array  # [1. 2. 3. 1. 3. 2. 2. 1. 3. 2. 3. 1. 3. 1. 2. 3. 2. 1.]


def permu_seg(data):
    new_arr1 = permu_array(read_array(data, 0))
    for j in range(data.shape[1]):
        if j != 0:
            new_arr1 = np.vstack((new_arr1, permu_array(read_array(data, j))))
    return new_arr1.transpose().reshape((new_arr1.shape[1], new_arr1.shape[0], 1))


print(permu_seg(train_x[0]).shape)  # (18, 68, 1)


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
np.save("ADL_data/np_2d_new/np_data_2d_permu_v1.npy", train_x_algorithm1)
print(train_x_algorithm1.shape)  # (7055, 18, 68, 1)

# test = read_array(train_x[0], 0)
# print(test)
# test1 = permu_array(test)
# print(test1)
# [-0.12264881  1.26280523 -0.66659658]
# [-0.12264881  1.26280523 -0.66659658 -0.12264881 -0.66659658  1.26280523
#   1.26280523 -0.12264881 -0.66659658  1.26280523 -0.66659658 -0.12264881
#  -0.66659658 -0.12264881  1.26280523 -0.66659658  1.26280523 -0.12264881]