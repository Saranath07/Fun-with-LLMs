import json
import matplotlib.pyplot as plt
import os

def plot_and_save_metrics_from_json(metrics_file, output_dir="Application/trainLLAMA/metrics_plots"):
    """
    Reads a JSON file containing metrics, plots each metric, and saves the plots as images.
    
    :param metrics_file: Path to the JSON file containing metrics data.
    :param output_dir: Directory where the plot images will be saved.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load metrics from JSON file
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)

    # Plot and save training losses
    if "train_losses" in metrics:
        plt.figure(figsize=(10, 6))
        plt.plot(metrics['train_losses'], label='Training Loss', color='blue')
        plt.title('Training Loss Over Steps', fontsize=16)
        plt.xlabel('Steps', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.grid(alpha=0.5)
        plt.legend()
        file_path = os.path.join(output_dir, "training_loss.png")
        plt.savefig(file_path)
        plt.close()  # Close the plot to free memory

    # Plot and save evaluation losses
    if "eval_losses" in metrics:
        plt.figure(figsize=(10, 6))
        plt.plot(metrics['eval_losses'], label='Evaluation Loss', color='orange')
        plt.title('Evaluation Loss Over Epochs', fontsize=16)
        plt.xlabel('Epochs', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.grid(alpha=0.5)
        plt.legend()
        file_path = os.path.join(output_dir, "evaluation_loss.png")
        plt.savefig(file_path)
        plt.close()

    # Plot and save learning rates
    if "learning_rates" in metrics:
        plt.figure(figsize=(10, 6))
        plt.plot(metrics['learning_rates'], label='Learning Rate', color='green')
        plt.title('Learning Rate Schedule', fontsize=16)
        plt.xlabel('Steps', fontsize=14)
        plt.ylabel('Learning Rate', fontsize=14)
        plt.grid(alpha=0.5)
        plt.legend()
        file_path = os.path.join(output_dir, "learning_rate.png")
        plt.savefig(file_path)
        plt.close()

    # Plot and save perplexities
    if "perplexities" in metrics:
        plt.figure(figsize=(10, 6))
        plt.plot(metrics['perplexities'], label='Perplexity', color='red')
        plt.title('Perplexity Over Steps', fontsize=16)
        plt.xlabel('Steps', fontsize=14)
        plt.ylabel('Perplexity', fontsize=14)
        plt.grid(alpha=0.5)
        plt.legend()
        file_path = os.path.join(output_dir, "perplexity.png")
        plt.savefig(file_path)
        plt.close()

    # Plot and save GPU memory usage
    if "gpu_memory" in metrics:
        plt.figure(figsize=(10, 6))
        plt.plot(metrics['gpu_memory'], label='GPU Memory Usage (MB)', color='purple')
        plt.title('GPU Memory Usage Over Steps', fontsize=16)
        plt.xlabel('Steps', fontsize=14)
        plt.ylabel('Memory (MB)', fontsize=14)
        plt.grid(alpha=0.5)
        plt.legend()
        file_path = os.path.join(output_dir, "gpu_memory_usage.png")
        plt.savefig(file_path)
        plt.close()

    print(f"Plots saved to directory: {output_dir}")



# Example Usage
metrics_file_path = "Application/trainLLAMA/training_logs/training_metrics_20241117_155558.json"  # Replace with the actual path
plot_and_save_metrics_from_json(metrics_file_path)
