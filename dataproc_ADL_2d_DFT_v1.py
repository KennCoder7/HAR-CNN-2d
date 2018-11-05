import numpy as np

train_x = np.load("ADL_data/np_2d_new/np_data_2d_permu_v1.npy")


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


train_x_dft = dataset_dft(train_x, "train")
np.save("./ADL_data/np_2d_new/np_data_2d_dft_v1.npy", train_x_dft)
print(train_x_dft.shape)    # (7055, 18, 68, 1)
