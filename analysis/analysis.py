"""
The task of this script is as follows:
- Based on the database variable value, which could be cifar10 or mnist. It will open mnist_logs.txt or cifar10_logs.txt.
- Parse the log file to extract epoch data and create three plots:
  1. Time vs Epoch
  2. Error Rate vs Epoch: Will contain validation error rate vs epoch
  3. Loss vs Epoch: Will contain training loss and validation loss vs epoch with proper legends
"""

import matplotlib.pyplot as plt
import re
import os
import numpy as np

def parse_log_file(database):
    """
    Parse the log file based on the database type (mnist or cifar10).
    Returns lists of extracted data for plotting.
    """
    filename = f"{database}_logs.txt"
    filepath = os.path.join(os.path.dirname(__file__), filename)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Log file {filename} not found in analysis directory")
    
    epochs = []
    times = []
    training_losses = []
    validation_losses = []
    validation_error_rates = []
    
    with open(filepath, 'r') as file:
        for line in file:
            line = line.strip()
            
            # Parse epoch information
            epoch_match = re.match(r'Epoch (\d+) of \d+ took ([\d.]+)s', line)
            if epoch_match:
                epoch = int(epoch_match.group(1))
                time = float(epoch_match.group(2))
                epochs.append(epoch)
                times.append(time)
                continue
            
            # Parse training loss (case insensitive for "training" vs "Training")
            if re.search(r'training loss:', line, re.IGNORECASE):
                loss_match = re.search(r'training loss:\s*([\d.]+(?:e[+-]?\d+)?)', line, re.IGNORECASE)
                if loss_match:
                    training_losses.append(float(loss_match.group(1)))
                continue
            
            # Parse validation loss (case insensitive)
            if re.search(r'validation loss:', line, re.IGNORECASE):
                loss_match = re.search(r'validation loss:\s*([\d.]+(?:e[+-]?\d+)?)', line, re.IGNORECASE)
                if loss_match:
                    validation_losses.append(float(loss_match.group(1)))
                continue
            
            # Parse validation error rate (case insensitive)
            if re.search(r'validation error rate:', line, re.IGNORECASE):
                error_match = re.search(r'validation error rate:\s*([\d.]+(?:e[+-]?\d+)?)%', line, re.IGNORECASE)
                if error_match:
                    validation_error_rates.append(float(error_match.group(1)))
                continue
    
    return epochs, times, training_losses, validation_losses, validation_error_rates

def create_plots(database):
    """
    Create and save the three required plots for the given database.
    """
    try:
        epochs, times, training_losses, validation_losses, validation_error_rates = parse_log_file(database)
        
        # Ensure all lists have the same length (number of epochs)
        min_length = min(len(epochs), len(times), len(training_losses), 
                        len(validation_losses), len(validation_error_rates))
        
        epochs = epochs[:min_length]
        times = times[:min_length]
        training_losses = training_losses[:min_length]
        validation_losses = validation_losses[:min_length]
        validation_error_rates = validation_error_rates[:min_length]
        
        print(f"Parsed {min_length} epochs for {database.upper()}")
        
        # Set up the plotting style
        plt.style.use('default')
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['font.size'] = 12
        
        # Plot 1: Time vs Epoch
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, times, 'b-', linewidth=2, marker='o', markersize=4)
        plt.xlabel('Epoch')
        plt.ylabel('Time (seconds)')
        plt.title(f'{database.upper()} - Training Time per Epoch')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{database}_time_vs_epoch.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {database}_time_vs_epoch.png")
        
        # Plot 2: Error Rate vs Epoch
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, validation_error_rates, 'r-', linewidth=2, marker='s', markersize=4)
        plt.xlabel('Epoch')
        plt.ylabel('Validation Error Rate (%)')
        plt.title(f'{database.upper()} - Validation Error Rate vs Epoch')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{database}_error_rate_vs_epoch.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {database}_error_rate_vs_epoch.png")
        
        # Plot 3: Loss vs Epoch (Training and Validation)
        epochs = epochs[1:]
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, training_losses[1:], 'b-', linewidth=2, marker='o', markersize=4, label='Training Loss')
        plt.plot(epochs, validation_losses[1:], 'r-', linewidth=2, marker='s', markersize=4, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{database.upper()} - Training and Validation Loss vs Epoch')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{database}_loss_vs_epoch.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {database}_loss_vs_epoch.png")
        
        # Print summary statistics
        print(f"\n{database.upper()} Summary:")
        print(f"Total epochs: {len(epochs)}")
        print(f"Average time per epoch: {np.mean(times):.2f}s")
        print(f"Total Training Time: {np.sum(times):.3f}s")
        print(f"Final training loss: {training_losses[-1]:.6f}")
        print(f"Final validation loss: {validation_losses[-1]:.6f}")
        print(f"Final validation error rate: {validation_error_rates[-1]:.4f}%")
        print(f"Best validation error rate: {min(validation_error_rates):.4f}%")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Error processing {database} logs: {e}")

# Main execution
database = 'mnist'  # or could be 'cifar10'

if __name__ == "__main__":
    create_plots(database)
