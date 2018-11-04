import numpy as np

train_x = np.load('UCI_data/np_2d/np_train_x_algorithm1.npy')
test_x = np.load('UCI_data/np_2d/np_test_x_algorithm1.npy')


def dataset_dft(data, proc_name):
    new_data = np.fft.fft2(data[0].reshape((data.shape[1], data.shape[2]))).\
        reshape((1, data.shape[1], data.shape[2], data.shape[3]))
    n = 0
    for i in range(data.shape[0]):
        if i != 0:
            new_data = np.vstack((new_data, np.fft.fft2(data[i].reshape((data.shape[1], data.shape[2]))).
                                  reshape((1, data.shape[1], data.shape[2], data.shape[3]))))
        if i - n > 0.05 * data.shape[0]:
            n = i
            print("### Process --- (", proc_name, "_ data ) DFT --- In progress --- [ ",
                  int(100 * round(i / data.shape[0], 2)), "% ] Finished ###")
    return new_data


test_x_dft = dataset_dft(test_x, "test")
np.save("./UCI_data/np_2d/np_test_x_dft_v1.npy", test_x_dft)
print(test_x_dft.shape)     # (2947, 36, 128, 1)
train_x_dft = dataset_dft(train_x, "train")
np.save("./UCI_data/np_2d/np_train_x_dft_v1.npy", train_x_dft)
print(train_x_dft.shape)    # (7352, 36, 128, 1)

