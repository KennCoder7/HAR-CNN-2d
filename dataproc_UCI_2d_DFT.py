import numpy as np

train_x = np.load('UCI_data/np_2d/np_train_x_algorithm1.npy')
test_x = np.load('UCI_data/np_2d/np_test_x_algorithm1.npy')


def dataset_dft(data, proc_name):
    new_data = np.fft.fftn(data[0]).reshape((1, data.shape[1], data.shape[2], data.shape[3]))
    n = 0
    for i in range(data.shape[0]):
        if i != 0:
            new_data = np.vstack((new_data, np.fft.fftn(data[i]).reshape((1, data.shape[1], data.shape[2], data.shape[3]))))
        if i - n > 0.05 * data.shape[0]:
            n = i
            print("### Process --- (", proc_name, "_ data ) DFT --- In progress --- [ ",
                  int(100 * round(i / data.shape[0], 2)), "% ] Finished ###")
    return new_data


test_x_dft = dataset_dft(test_x, "test")
np.save("./UCI_data/np_2d/np_test_x_dft.npy", test_x_dft)
print(test_x_dft.shape)     # (2947, 36, 128, 1)
train_x_dft = dataset_dft(train_x, "train")
np.save("./UCI_data/np_2d/np_train_x_dft.npy", train_x_dft)
print(train_x_dft.shape)    # (7352, 36, 128, 1)


# test_x = test_x[0:10]
# print(dataset_dft(test_x, "test").shape)    # (10, 36, 128, 1)
# test_a = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int)
# print(test_a[0:5])  # [0 1 2 3 4]
# print(test_x[0:10].shape)      # (10, 36, 128, 1)
# print(test_x[0].shape)      # (36, 128, 1)
# print(test_x[[0]].shape)    # (1, 36, 128, 1)

# train_2d = train_x[0]
# print(train_2d.shape)   # (36, 128, 1)
# print(np.fft.fftn(train_2d).shape)  # (36, 128, 1)
# print(train_2d[0].shape)
# print(np.fft.fftn(train_2d)[0].shape)
# print(train_2d[0])
# print(np.fft.fftn(train_2d)[0])
