import os
import matplotlib.pyplot as plt

def draw_train_val_graph(train_loss, val_acc, home_dir):
    if not os.path.exists(os.path.join(home_dir, 'save')):
        os.mkdir(os.path.join(home_dir, 'save'))
    plt.plot([i + 1 for i in range(len(train_loss))], train_loss, label='train_loss')
    plt.plot([i + 1 for i in range(len(val_acc))], val_acc, label='val_acc')
    plt.legend()
    plt.show()
    plt.savefig(os.path.join(home_dir, 'save', 'training.png'))