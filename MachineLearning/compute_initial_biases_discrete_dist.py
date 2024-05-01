import dataset
import numpy as np


def get_biases(data_path):
    train_data = dataset.get_dataset(data_path, data_version=2)
    val_counts = np.zeros((530, 8))
    summed_densities = np.zeros((530, 8))
    k = 0
    total_samples = 0
    
    for elem in train_data:
        print(k)
        _, output = elem
        output = np.clip(output, 0, 7)
        batch_size = output.shape[0]
        for j in range(batch_size):
            summed_densities += output
            total_samples += 1

        k += 1
    
    avg_density = summed_densities / total_samples
    print(avg_density)
    
    #otal_counts_per_site = np.sum(val_counts, axis = -1)
    
    #print(total_counts_per_site)
    
    #freq = val_counts / total_counts_per_site[:, None]
    #print(freq)
    log_avg_density = np.log(avg_density)
    log_avg_density = log_avg_density[log_avg_density < -100] = -100
    print(log_freq)
    np.save('MachineLearning/ConvProbPredictor/initial_biases_new.npy', log_avg_density)
    np.save('MachineLearning/ConvProbPredictor/expected_loss_new.npy', - np.sum( log_avg_density * avg_density ))
    #print(np.max(val_counts, axis=0))
    #print(np.min(val_counts))
    
        
get_biases('/nesi/project/uoa03669/ewin313/TropicalCycloneAI/Data/run2')