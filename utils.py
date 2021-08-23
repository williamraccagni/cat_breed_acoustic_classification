import matplotlib.pyplot as plt

def plot_labels_balancing( path : str, title : str, filename : str, y : list, labels_map : list, labels_struct : list):

    quantity = []
    for l in labels_struct:
        quantity.append(y.count(l))

    plt.bar(labels_map, quantity)

    plt.xlabel('Labels')
    plt.ylabel('Quantity')
    plt.title(title + ': Balance of Labels')

    plt.savefig(path + '/' + filename + '_labels_balancing.png')
    plt.clf()  # clear plot for next method
