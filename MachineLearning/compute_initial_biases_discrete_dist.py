import dataset
import numpy as np


def get_biases(data_path):
    train_data = dataset.get_dataset(data_path, data_version=2)
    val_counts = np.zeros((542, 6))
    k = 0
    for elem in train_data:
        print(k)
        (gen_matrix, _), output = elem
        output = np.clip(output, 0, 5)
        batch_size = output.shape[0]
        print(batch_size)
        for j in range(batch_size):
            for i in range(542):
                val_counts[i][int(output[j][i])] = val_counts[i][int(output[j][i])] + 1
        k += 1

    print(val_counts)

    total_counts_per_site = np.sum(val_counts, axis=-1)

    print(total_counts_per_site)

    freq = val_counts / total_counts_per_site[:, None]
    print(freq)
    log_freq = np.log(freq)
    log_freq[log_freq < -100] = -100
    print(log_freq)
    np.save("MachineLearning/ConvProbPredictor/initial_biases.npy", log_freq)
    np.save(
        "MachineLearning/ConvProbPredictor/expected_loss.npy", -np.sum(log_freq * freq)
    )
    print(np.max(val_counts, axis=0))
    print(np.min(val_counts))


get_biases("../storm_data/v2/")
