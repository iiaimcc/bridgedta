import os
import sys
import torch
import numpy as np

sys.path.append("/home/zh/workspace/BridgeDTA")

from trainer import DTATrainer
from models.Bridge_DTA.bridge_dta_full import BridgeDTA
from datahelper.bridge_dataset import (
    BridgeDTADataset, CHARPROTLEN,
    bridge_forward_step, get_bridge_dataloader,
)

# Global device setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TrainingArguments:
    def __init__(self):
        self.model_name = "Bridge_DTA_FULL"
        self.learning_rate = 1e-4
        self.weight_decay = 1e-5
        self.num_epochs = 500
        self.batch_size = 128
        self.seed = 42
        self.log_steps = 20
        self.do_plot = True
        self.dataset = "kiba"
        self.cold_start = ""
        self.ci_weight = 1.0    # tmp 0.2/1.0
        # KPGT feature mode: "graph" (2304-d) or "node" (768-d token-level)
        self.kpgt_feature_level = "node"
        # Whether to prepend [fp_vn, md_vn] as CLS+REG tokens at the beginning of K/V sequence in node mode
        self.kpgt_use_global_tokens = True


def _load_kpgt_features(kpgt_path, level, use_global_tokens=True):
    """Load KPGT features, return dict[smiles -> np.ndarray] and LPM dimension.

    - level == "graph": file contains dict[smiles -> [kpgt_dim]], kpgt_dim=2304
    - level == "node":  file contains (smiles, atom_feats, offsets, fp_vn, md_vn);
                        reconstructs dict[smiles -> [n_atoms (+2), d_g_feats]].
                        When use_global_tokens=True, prepends [fp_vn, md_vn]
                        as CLS+REG global tokens at the beginning of the sequence.
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
    base_dir = "/home/zh/workspace/BridgeDTA"

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
    with np.load(esm_path, allow_pickle=True) as _esm_npz:
        esm_features = _esm_npz['dict'][()]


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
                        torch.save(
                            best_model,
                            os.path.join(self.args.output_dir, 'best_model.pth')
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
                    self.model.load_state_dict(torch.load(best_model_path, map_location=device))
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

    # Train model
    print('Starting training...')
    best_model = trainer.train()

    # Test model
    print('Testing model...')
    predictions, test_metrics = trainer.predict(test_set)
    print(f"Test metrics: {test_metrics}")

if __name__ == "__main__":
    main()
