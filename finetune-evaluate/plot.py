import matplotlib.pyplot as plt

def plot_learning_curves(results):
    """
    Plots the learning curves for training loss.
    
    Args:
        results (dict): Dictionary with keys:
            - 'epochs': list of epoch numbers
            - 'train_loss': list of training loss values
    """
    epochs = results['epochs']
    train_loss = results['train_loss']
    
    # Plot 1: Training and Eval Loss
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, marker='o', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()



results = {
    'epochs': [0.4, 0.81, 1.16, 1.57, 1.97, 2.32, 2.73],
    'train_loss': [1.8102, 1.6783, 1.59, 1.6031, 1.5876, 1.5161, 1.4266]
}

plot_learning_curves(results)