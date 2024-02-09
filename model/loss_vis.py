import matplotlib.pyplot as plt
import numpy as np

def drow(train_losses, valid_losses, train_accs, valid_accs, name):
    plt.figure(figsize=(15, 7))

    # 繪製 Training loss 和 Validation loss
    plt.subplot(121)
    plt.plot(range(len(train_losses)), train_losses, label='Training Loss')
    plt.plot(range(len(valid_losses)), valid_losses, label='Validation Loss')
    plt.legend(loc='upper left')
    plt.title('Loss')

    # 繪製 Training accuracy 和 Validation accuracy
    plt.subplot(122)
    plt.plot(range(len(train_accs)), train_accs, label='Training Accuracy')
    plt.plot(range(len(valid_accs)), valid_accs, label='Validation Accuracy')
    plt.yticks(np.arange(0.8, 1, 0.05))
    plt.legend(loc='upper left')
    plt.title('Accuracy')

    plt.savefig("eval_fig/%s.png"%name)