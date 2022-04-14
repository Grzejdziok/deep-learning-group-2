import numpy as np
import json
if __name__ == '__main__':
    filename_1 = 'data_random_lenet300100_mnist_random_1_run.json'
    iterations_1 = 1
    filename_2 = 'data_random_lenet300100_mnist_random_2_runs.json'
    iterations_2 = 2
    filename_out = 'data_random.json'

    f = open(filename_1)
    data1 = json.load(f)
    PM_LIST_1 = data1['PM_LIST']
    VALIDATION_ITERATIONS_1= np.array(data1['VALIDATION_ITERATIONS'])
    PRUNE_RATE_1= data1['PRUNE_RATE']
    test_accuracies_1= np.array(data1['test_accuracies'])
    test_losses_1= np.array(data1['test_losses'])
    val_accuracies_1= np.array(data1['val_accuracies'])
    val_losses_1= np.array(data1['val_losses'])

    f = open(filename_2)
    data2 = json.load(f)
    PM_LIST_2 = data2['PM_LIST']
    VALIDATION_ITERATIONS_2= np.array(data2['VALIDATION_ITERATIONS'])
    PRUNE_RATE_2= data2['PRUNE_RATE']
    test_accuracies_2= np.array(data2['test_accuracies'])
    test_losses_2= np.array(data2['test_losses'])
    val_accuracies_2= np.array(data2['val_accuracies'])
    val_losses_2= np.array(data2['val_losses'])



    if np.all(PM_LIST_1==PM_LIST_2) and np.all(VALIDATION_ITERATIONS_1 == VALIDATION_ITERATIONS_2) and np.all(PRUNE_RATE_1 == PRUNE_RATE_2):
        test_accuracies_1 = test_accuracies_1[:,:iterations_1]
        test_losses_1 = test_losses_1[:,:iterations_1]
        val_accuracies_1 = val_accuracies_1[:,:iterations_1]
        val_losses_1 = val_losses_1[:,:iterations_1]
        test_accuracies_2 = test_accuracies_2[:,:iterations_2]
        test_losses_2 = test_losses_2[:,:iterations_2]
        val_accuracies_2 = val_accuracies_2[:,:iterations_2]
        val_losses_2 = val_losses_2[:,:iterations_2]
        test_accuracies = np.append(test_accuracies_1, test_accuracies_2, axis=1)
        test_losses = np.append(test_losses_1, test_losses_2, axis=1)
        val_accuracies = np.append(val_accuracies_1, val_accuracies_2, axis=1)
        val_losses = np.append(val_losses_1, val_losses_2, axis=1)

        export_dict = {"VALIDATION_ITERATIONS": VALIDATION_ITERATIONS_1.tolist(),
                    "PM_LIST": PM_LIST_1,
                    "PRUNE_RATE": PRUNE_RATE_1,
                    "test_accuracies": test_accuracies.tolist(),
                    "test_losses": test_losses.tolist(),
                    "val_accuracies": val_accuracies.tolist(),
                    "val_losses": val_losses.tolist(),
                    }
        with open(filename_out, 'w') as file:
            json.dump(export_dict, file)
    else:
        print("The files contain different data!")