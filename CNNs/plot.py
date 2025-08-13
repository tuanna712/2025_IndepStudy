import matplotlib.pyplot as plt
def metrics_plot(num_epochs, train_losses, test_losses, train_accs, test_accs, train_errs, test_errs):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 3, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(range(1, num_epochs + 1), train_accs, label='Train Accuracy')
    plt.plot(range(1, num_epochs + 1), test_accs, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Test Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(range(1, num_epochs + 1), train_errs, label='Train Error')
    plt.plot(range(1, num_epochs + 1), test_errs, label='Test Error')
    plt.xlabel('Epoch')
    plt.ylabel('Error (%)')
    plt.title('Training and Test Error')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()