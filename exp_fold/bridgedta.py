"""
BridgeDTA 5-fold cross-validation script.
Usage: python bridgedta.py <fold>  (fold: 0-4)
"""
import os
import random
import sys
import torch
import numpy as np

import os, sys as _sys
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)
from paths import setup_imports, get_repo_root, apply_training_cli_overrides
setup_imports()

try:
    # Optional checkpoint-resume trainer variant (not shipped with this repo).
    from trainer_resume import DTATrainer  # change 2 resume version
except ModuleNotFoundError:
    from trainer import DTATrainer
from models.Bridge_DTA.bridge_dta_full import BridgeDTA
from datahelper.bridge_dataset import (
    BridgeDTADataset, CHARPROTLEN,
    bridge_forward_step, get_bridge_dataloader,
)

# Global device setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Get fold number
fold = sys.argv[1] if len(sys.argv) > 1 else "0"

class TrainingArguments:
    def __init__(self):
        self.model_name = f"Bridge_DTA_FULL_{fold}"
        self.learning_rate = 1e-4
        self.weight_decay = 1e-5
        self.num_epochs = 500
        self.batch_size = 128
        self.seed = 42
        self.log_steps = 20
        self.do_plot = True
        self.dataset = "kiba"
        self.cold_start = fold
        self.ci_weight = 1.0
        self.resume_checkpoint = ""
        # KPGT feature mode: "graph" (2304-d) or "node" (768-d)
        self.kpgt_feature_level = "node"
        # Whether to prepend [fp_vn, md_vn] as CLS+REG tokens at the beginning of K/V sequence in node mode
        self.kpgt_use_global_tokens = True
        # Allow command-line overrides (e.g. --dataset davis --num-epochs 1)
        apply_training_cli_overrides(
            self,
            keys=("dataset", "num_epochs", "batch_size", "seed", "log_steps", "cold_start", "resume_checkpoint"),
        )


def _load_kpgt_features(kpgt_path, level, use_global_tokens=True):
    """Load KPGT features, return dict[smiles -> np.ndarray] and LPM dimension.

    In node mode with use_global_tokens=True, prepend [fp_vn, md_vn] as
    CLS+REG global tokens at the beginning of each molecule's atom sequence.
    """
    if level == "graph":
        with np.load(kpgt_path) as npz:
            kpgt_features = {key: npz[key] for key in npz.files}
        first = next(iter(kpgt_features.values()))
        return kpgt_features, int(first.shape[-1])
    if level == "node":
        with np.load(kpgt_path, allow_pickle=True) as npz:
            smiles_arr = npz['smiles']
            atom_feats = npz['atom_feats']
            offsets = npz['offsets']
            fp_vn = npz['fp_vn'] if use_global_tokens else None
            md_vn = npz['md_vn'] if use_global_tokens else None
        kpgt_features = {}
        for i in range(len(smiles_arr)):
            atoms = atom_feats[offsets[i]:offsets[i + 1]]
            if use_global_tokens:
                global_tokens = np.stack([fp_vn[i], md_vn[i]], axis=0)
                kpgt_features[str(smiles_arr[i])] = np.concatenate(
                    [global_tokens, atoms], axis=0
                )
            else:
                kpgt_features[str(smiles_arr[i])] = atoms
        return kpgt_features, int(atom_feats.shape[-1])
    raise ValueError(f"Unknown kpgt_feature_level: {level}")


def main():
    args = TrainingArguments()
    # Use global device variable
    base_dir = str(get_repo_root())

    # Load KPGT and ESM features
    kpgt_filename = (
        f"{args.dataset.lower()}_node.npz"
        if args.kpgt_feature_level == "node"
        else f"{args.dataset.lower()}.npz"
    )
    kpgt_path = os.path.join(base_dir, "models/Bridge_DTA/feature_kpgt", kpgt_filename)
    esm_path = os.path.join(base_dir, "models/Bridge_DTA/feature_esm", f"{args.dataset.lower()}.npz")

    print(f"Loading KPGT features ({args.kpgt_feature_level}-level) from {kpgt_path} ...")
    kpgt_features, kpgt_dim = _load_kpgt_features(
        kpgt_path,
        args.kpgt_feature_level,
        use_global_tokens=args.kpgt_use_global_tokens,
    )
    print(f"  -> {len(kpgt_features)} molecules, kpgt_dim={kpgt_dim}, "
          f"global_tokens={args.kpgt_use_global_tokens}")
    print("Loading ESM features...")
    esm_features = np.load(esm_path, allow_pickle=True)['dict'][()]


    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set dataset-related parameters
    max_fasta_len = 1000
    
    # Load datasets
    train_set = BridgeDTADataset(
        root=f"{base_dir}/data",
        dataset=args.dataset.lower(),
        mode='train',
        cold_start=args.cold_start,
        kpgt_features=kpgt_features,
        esm_features=esm_features,
        max_fasta_len=max_fasta_len,
        device=device
    )
    
    test_set = BridgeDTADataset(
        root=f"{base_dir}/data",
        dataset=args.dataset.lower(),
        mode='test',
        cold_start=args.cold_start,
        kpgt_features=kpgt_features,
        esm_features=esm_features,
        max_fasta_len=max_fasta_len,
        device=device
    )

    # Initialize model using the simple version
    model = BridgeDTA(
        fasta_char_set_len=CHARPROTLEN+1,
        num_feat_xd=78,
        num_feat_out=128,
        dr=0.2,
        kpgt_dim=kpgt_dim,
    ).to(device)
    print(f"Model initialized, using device: {device}")

    # Custom trainer with CI loss handling
    class CIAwareTrainer(DTATrainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.ci_weight = self.args.ci_weight
            self.start_epoch = 0
            self.resume_global_step = 0
            self.resume_best_val_loss = float('inf')
            self.resume_source = None

        def load_training_checkpoint(self, checkpoint_path):
            checkpoint_path = os.path.abspath(os.path.expanduser(checkpoint_path))
            checkpoint = torch.load(
                checkpoint_path,
                map_location=device,
                weights_only=True,
            )
            required_keys = {
                'model_state_dict', 'optimizer_state_dict', 'epoch',
                'best_val_loss', 'global_step', 'torch_rng_state',
                'numpy_rng_state', 'python_rng_state',
                'training_history', 'metadata'
            }
            missing_keys = required_keys.difference(checkpoint)
            if missing_keys:
                raise ValueError(
                    f"Resume checkpoint is missing training state: {sorted(missing_keys)}"
                )

            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.start_epoch = int(checkpoint['epoch'])
            self.resume_global_step = int(checkpoint['global_step'])
            self.resume_best_val_loss = float(checkpoint['best_val_loss'])
            self.resume_source = checkpoint_path
            torch.set_rng_state(checkpoint['torch_rng_state'])
            random.setstate(checkpoint['python_rng_state'])
            numpy_rng_state = checkpoint['numpy_rng_state']
            np.random.set_state((
                numpy_rng_state['bit_generator'],
                numpy_rng_state['state'].cpu().numpy(),
                numpy_rng_state['position'],
                numpy_rng_state['has_gauss'],
                numpy_rng_state['cached_gaussian'],
            ))
            if torch.cuda.is_available() and checkpoint.get('cuda_rng_state_all') is not None:
                torch.cuda.set_rng_state_all(checkpoint['cuda_rng_state_all'])
            training_history = checkpoint['training_history']
            self.train_losses = training_history['train_losses']
            self.epochs = training_history['epochs']
            self.val_losses = training_history['val_losses']
            self.val_cis = training_history['val_cis']
            self.val_rm2s = training_history['val_rm2s']

            logging.info(
                "Resuming training from %s at epoch %d with metadata %s",
                checkpoint_path,
                self.start_epoch,
                checkpoint['metadata'],
            )
            
        def train(self):
            best_model = None
            best_val_loss = self.resume_best_val_loss
            global_step = self.resume_global_step
            
            # Use the custom or default get_dataloader function for training
            train_dataloader = self.get_dataloader_fn(self.train_dataset, self.args.batch_size, shuffle=True)
            
            for epoch in range(self.start_epoch, self.args.num_epochs):
                # Training
                self.model.train()
                train_loss = 0.0
                train_ci_loss = 0.0
                
                for batch in train_dataloader:
                    # Forward pass using the provided forward_step function
                    scores, labels, ci_loss = self.forward_step(batch, self.model)
                    mse_loss = self.loss_fn(scores, labels)
                    
                    # Combined loss with weighting
                    loss = mse_loss + self.ci_weight * ci_loss
                    
                    # Backward pass
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                    train_loss += mse_loss.item()
                    train_ci_loss += ci_loss.item()
                    global_step += 1
                
                avg_train_loss = train_loss / len(train_dataloader)
                avg_train_ci_loss = train_ci_loss / len(train_dataloader)
                logging.info(f'Epoch {epoch + 1}/{self.args.num_epochs} - Training MSE Loss: {avg_train_loss:.4f}, CI Loss: {avg_train_ci_loss:.4f}')
                
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
                        numpy_rng_state = np.random.get_state()
                        checkpoint_path = os.path.abspath(
                            os.path.join(self.args.output_dir, 'best_model.pth')
                        )
                        torch.save(
                            {
                                'model_state_dict': best_model,
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                'epoch': epoch + 1,
                                'best_val_loss': best_val_loss,
                                'global_step': global_step,
                                'torch_rng_state': torch.get_rng_state(),
                                'cuda_rng_state_all': (
                                    torch.cuda.get_rng_state_all()
                                    if torch.cuda.is_available()
                                    else None
                                ),
                                'python_rng_state': random.getstate(),
                                'numpy_rng_state': {
                                    'bit_generator': numpy_rng_state[0],
                                    'state': torch.from_numpy(numpy_rng_state[1].copy()),
                                    'position': numpy_rng_state[2],
                                    'has_gauss': numpy_rng_state[3],
                                    'cached_gaussian': numpy_rng_state[4],
                                },
                                'training_history': {
                                    'train_losses': self.train_losses,
                                    'epochs': self.epochs,
                                    'val_losses': self.val_losses,
                                    'val_cis': self.val_cis,
                                    'val_rm2s': self.val_rm2s,
                                },
                                'metadata': {
                                    'dataset': self.args.dataset,
                                    'fold': self.args.cold_start,
                                    'seed': self.args.seed,
                                    'checkpoint_path': checkpoint_path,
                                    'resumed_from': self.resume_source,
                                },
                            },
                            checkpoint_path,
                        )
                        logging.info(f'New best model saved with validation loss: {val_loss:.4f}')
            
            if hasattr(self.args, 'do_plot') and self.args.do_plot:
                self.plot_training_curves()
            
            return best_model
            
        @torch.no_grad()
        def evaluate(self, dataset):
            self.model.eval()
            y_true = []
            y_pred = []
            total_loss = 0.0
            
            # Use the custom or default get_dataloader function for validation
            val_dataloader = self.get_dataloader_fn(dataset, self.args.batch_size)
            
            for batch in val_dataloader:
                scores, labels, _ = self.forward_step(batch, self.model)
                loss = self.loss_fn(scores, labels)
                total_loss += loss.item()
                y_pred += scores.detach().cpu().tolist()
                y_true += labels.detach().cpu().tolist()
            
            # Calculate CI directly from model for comparison
            batch_ci = self.model.compute_ci(torch.tensor(y_pred), torch.tensor(y_true))
            
            # Calculate all metrics
            metrics = self.compute_metrics(y_true, y_pred)
            metrics['loss'] = total_loss / len(val_dataloader)
            metrics['batch_ci'] = batch_ci
            
            return metrics
            
        def predict(self, test_dataset, load_best_model=True):
            if load_best_model:
                best_model_path = os.path.join(self.args.output_dir, 'best_model.pth')
                if os.path.exists(best_model_path):
                    checkpoint = torch.load(
                        best_model_path,
                        map_location=device,
                        weights_only=True,
                    )
                    model_state = checkpoint.get('model_state_dict', checkpoint)
                    self.model.load_state_dict(model_state)
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
                    scores, labels, _ = self.forward_step(batch, self.model)
                    loss = self.loss_fn(scores, labels)
                    
                    total_loss += loss.item()
                    predictions.extend(scores.cpu().numpy())
                    y_true.extend(labels.cpu().numpy())

            # Calculate metrics
            metrics = self.compute_metrics(y_true, predictions)
            metrics['loss'] = total_loss / len(test_dataloader)
            
            # Calculate batch CI directly from model
            batch_ci = self.model.compute_ci(torch.tensor(predictions), torch.tensor(y_true))
            metrics['batch_ci'] = batch_ci
            
            # Log test results
            logging.info("Test Results:")
            for metric_name, metric_value in metrics.items():
                logging.info(f"{metric_name}: {metric_value:.4f}")
            
            return predictions, metrics

    # Initialize CI-aware trainer
    import logging
    trainer = CIAwareTrainer(
        model=model,
        train_args=args,
        train_dataset=train_set,
        val_dataset=test_set,
        forward_step=bridge_forward_step,
        get_dataloader_fn=get_bridge_dataloader
    )

    if args.resume_checkpoint:
        trainer.load_training_checkpoint(args.resume_checkpoint)

    # Train model
    print('Starting training...')
    best_model = trainer.train()

    # Test model
    print('Testing model...')
    predictions, test_metrics = trainer.predict(test_set)
    print(f"Test metrics: {test_metrics}")

if __name__ == "__main__":
    main()
