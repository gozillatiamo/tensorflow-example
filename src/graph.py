import matplotlib.pyplot as plt


# plot graph accuracy
def plot_graphs(history, string, enable_val=False):
    plt.plot(history.history[string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    if enable_val:
        plt.plot(history.history["val_" + string])
        plt.legend([string, "val_" + string])
    plt.show()
