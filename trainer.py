import os
import sys
import torch
import logging
import datetime
import numpy as np
import matplotlib.pyplot as plt
from utils.metrics import set_seed, compute_metrics
from torch.utils.data import DataLoader

# Make the repository importable regardless of where the script is launched from.
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

class DTATrainer:
    def __init__(
        self,
        model,
        train_args,
        train_dataset,
        val_dataset,
        forward_step,
        compute_metrics=compute_metrics,
        optimizer=None,
        get_dataloader_fn=None
    ):
        self.model = model
        self.args = train_args
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.compute_metrics = compute_metrics
        self.forward_step = forward_step
        self.get_dataloader_fn = get_dataloader_fn or self.get_dataloader
        
        # Set up logging
        self.setup_logging()
        
        # Log start time and training args
        current_time = datetime.datetime.now().strftime("%Y/%m/%d %H:%M")
        logging.info(f"Training started at: {current_time}")
        logging.info("Training Arguments:")
        for arg, value in vars(self.args).items():
            logging.info(f"  {arg}: {value}")
        logging.info("-" * 50)
        
        # Set random seeds
        set_seed(self.args.seed)
        logging.info(f"Random seed set to: {self.args.seed}")
        
        # Setup optimizer
        self.optimizer = optimizer if optimizer else torch.optim.Adam(
            model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=getattr(self.args, 'weight_decay', 0.0),
        )
        
        # Loss function
        self.loss_fn = torch.nn.MSELoss()
        
        # Set default log_steps if not provided
        self.args.log_steps = getattr(self.args, 'log_steps', 10)
        
        # Initialize lists to store metrics for plotting
        self.train_losses = []
        self.val_losses = []
        self.val_cis = []
        self.val_rm2s = []
        self.epochs = []

    def setup_logging(self):
        # Create experiment directory under <repo>/results
        exp_dir = os.path.join(_REPO_ROOT, 'results', self.args.model_name)
        os.makedirs(exp_dir, exist_ok=True)
        
        # Set up logging
        current_date = datetime.datetime.now().strftime("%m%d")
        log_file = os.path.join(exp_dir, f'{current_date}_{self.args.model_name}_{self.args.dataset}.log')
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        # Create output directory for checkpoints
        self.args.output_dir = os.path.join(exp_dir, f'checkpoints_{self.args.dataset}')
        os.makedirs(self.args.output_dir, exist_ok=True)

    def get_dataloader(self, dataset, batch_size, shuffle=False):
        """
        Convert a dataset into a DataLoader.
        Args:
            dataset: The dataset to convert.
            batch_size: The batch size for the DataLoader.
            shuffle: Whether to shuffle the dataset.
        Returns:
            DataLoader: A DataLoader for the given dataset.
        """
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            # collate_fn=dataset.collate_fn # todo
        )

    def train(self):
        best_model = None
        best_val_loss = float('inf')
        global_step = 0
        
        # Use the custom or default get_dataloader function for training
        train_dataloader = self.get_dataloader_fn(self.train_dataset, self.args.batch_size, shuffle=True)
        
        for epoch in range(self.args.num_epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for batch in train_dataloader:
                # Forward pass using the provided forward_step function
                scores, labels = self.forward_step(batch, self.model)
                loss = self.loss_fn(scores, labels)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                global_step += 1
            
            avg_train_loss = train_loss / len(train_dataloader)
            logging.info(f'Epoch {epoch + 1}/{self.args.num_epochs} - Training Loss: {avg_train_loss:.4f}')
            
            # Store training loss
            self.train_losses.append(avg_train_loss)
            
            # Validation only at specified log steps
            if (epoch + 1) % self.args.log_steps == 0:
                val_metrics = self.evaluate(self.val_dataset)
                val_loss = val_metrics['mse']
                
                # Store validation metrics for plotting
                self.epochs.append(epoch + 1)
                self.val_losses.append(val_loss)
                self.val_cis.append(val_metrics['ci'])
                self.val_rm2s.append(val_metrics['rm2'])
                
                # Log validation metrics
                logging.info(f'Validation metrics at epoch {epoch + 1}: {val_metrics}')
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model = self.model.state_dict()
                    torch.save(
                        best_model,
                        os.path.join(self.args.output_dir, 'best_model.pth')
                    )
                    logging.info(f'New best model saved with validation loss: {val_loss:.4f}')
        
        if hasattr(self.args, 'do_plot') and self.args.do_plot:
            self.plot_training_curves()
        
        return best_model

    def plot_training_curves(self):
        """
        Plot training curves with three subplots:
        1. Training and validation loss
        2. Validation CI
        3. Validation RM2
        """
        plt.style.use('seaborn')
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot 1: Training and Validation Loss
        train_epochs = np.arange(1, len(self.train_losses) + 1)
        ax1.plot(train_epochs, self.train_losses, 'b-', label='Training Loss')
        ax1.plot(self.epochs, self.val_losses, 'r-', label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot 2: Validation CI
        ax2.plot(self.epochs, self.val_cis, 'g-')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('CI Score')
        ax2.set_title('Validation CI')
        ax2.grid(True)
        
        # Plot 3: Validation RM2
        ax3.plot(self.epochs, self.val_rm2s, 'm-')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('RM2 Score')
        ax3.set_title('Validation RM2')
        ax3.grid(True)
        
        # Adjust layout and save
        plt.tight_layout()
        
        # Save plot
        plot_dir = os.path.join(self.args.output_dir, 'plots')
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, f'{self.args.model_name}_{self.args.dataset}_training_curves.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Training curves plot saved to {plot_path}")

    @torch.no_grad()
    def evaluate(self, dataset):
        self.model.eval()
        y_true = []
        y_pred = []
        total_loss = 0.0
        
        # Use the custom or default get_dataloader function for validation
        val_dataloader = self.get_dataloader_fn(dataset, self.args.batch_size)
        
        for batch in val_dataloader:
            scores, labels = self.forward_step(batch, self.model)
            loss = self.loss_fn(scores, labels)
            total_loss += loss.item()
            y_pred += scores.detach().cpu().tolist()
            y_true += labels.detach().cpu().tolist()
        
        # Calculate all metrics
        metrics = self.compute_metrics(y_true, y_pred)
        metrics['loss'] = total_loss / len(val_dataloader)
        
        return metrics

    def save_model(self, output_dir=None):
        """Save the model, optimizer, and training args"""
        if output_dir is None:
            output_dir = self.args.output_dir
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model
        torch.save(self.model.state_dict(), 
                  os.path.join(output_dir, 'final_model.pth'))
        
        logging.info(f"Model saved to {output_dir}")

    def predict(self, test_dataset, load_best_model=True):
        if load_best_model:
            best_model_path = os.path.join(self.args.output_dir, 'best_model.pth')
            if os.path.exists(best_model_path):
                self.model.load_state_dict(torch.load(best_model_path))
                logging.info("Loaded best model for prediction")
            else:
                logging.warning("Best model not found, using current model state")

        self.model.eval()
        predictions = []
        y_true = []
        total_loss = 0.0
        
        # Use the custom or default get_dataloader function for testing
        test_dataloader = self.get_dataloader_fn(test_dataset, self.args.batch_size)
        
        with torch.no_grad():
            for batch in test_dataloader:
                scores, labels = self.forward_step(batch, self.model)
                loss = self.loss_fn(scores, labels)
                
                total_loss += loss.item()
                predictions.extend(scores.cpu().numpy())
                y_true.extend(labels.cpu().numpy())

        # Calculate metrics
        metrics = self.compute_metrics(y_true, predictions)
        metrics['loss'] = total_loss / len(test_dataloader)
        
        # Log test results
        logging.info("Test Results:")
        for metric_name, metric_value in metrics.items():
            logging.info(f"{metric_name}: {metric_value:.4f}")
        
        return predictions, metrics

    def save_predictions(self, predictions, output_file='predictions.txt'):
        """
        Save predictions to a file
        Args:
            predictions: Model predictions
            output_file: File path to save predictions
        """
        output_path = os.path.join(self.args.output_dir, output_file)
        with open(output_path, 'w') as f:
            for pred in predictions:
                f.write(f"{pred}\n")
        logging.info(f"Predictions saved to {output_path}")
