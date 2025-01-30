import os
import copy
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd
import random

from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score

from dataset import MetaMolRT_Dataset, MolRT_Dataset
from utils import csv2pkl_wfilter



class TSTL_PreTrainer:
    def __init__(
        self,
        model,
        config,
        data_config, 
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.data_config = data_config
        
        # Initialize TSTLPre parameters
        self.inner_lr = config['train']['inner_lr']
        self.meta_lr = config['train']['meta_lr']
        self.num_inner_steps = config['train']['num_inner_steps']
        self.criterion = nn.HuberLoss(delta=1.0)
        
        # Meta-optimizer setup
        self.meta_optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.meta_lr,
            weight_decay=config['train']['meta_weight_decay']
        )
        self.meta_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.meta_optimizer,
            mode='min',
            factor=config['train']['meta_lr_factor'],
            patience=config['train']['meta_lr_patience'],
            min_lr=1e-7
        )

    def _init_datasets(self, train_task_paths, pretrain_task_path, valid_task_paths):
        # Initialize datasets
        self.meta_train_inner_dataset = MetaMolRT_Dataset(
            data_paths=train_task_paths,
            data_config=self.data_config,
            device=self.device,
            k_shot=self.config['train']['k_shot']
        )
        
        self.meta_train_outer_dataset = MetaMolRT_Dataset(
            data_paths=[pretrain_task_path],
            data_config=self.data_config,
            device=self.device,
            k_shot=None
        )
        
        self.meta_train_outer_loader = DataLoader(
            self.meta_train_outer_dataset,
            batch_size=self.config['train']['outer_batch_size'],
            shuffle=True,
            drop_last=True
        )
        
        self.meta_valid_inner_dataset = MetaMolRT_Dataset(
            data_paths=valid_task_paths,
            data_config=self.data_config,
            scalers=self.meta_train_inner_dataset.scalers,
            device=self.device,
            k_shot=self.config['train']['k_shot']
        )

    def _warmup_train(self):
        warmup_optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['train']['warmup_lr'],
            weight_decay=self.config['train']['warmup_weight_decay']
        )
        
        for epoch in range(self.config['train'].get('warmup_epochs', 0)):
            warmup_losses = []
            # for batch in tqdm(self.meta_train_outer_loader, desc=f"Warmup Epoch {epoch+1}"):
            for batch in self.meta_train_outer_loader: 
                _, x_data, x_mask, y_data = batch
                x_data = x_data.to(self.device).permute(0, 2, 1)
                x_mask = x_mask.to(self.device, dtype=torch.bool)
                y_data = y_data.to(self.device)
                
                batch_size, _, num_points = x_data.size()
                idx_base = torch.arange(0, batch_size, device=self.device).view(-1, 1, 1) * num_points
                
                warmup_optimizer.zero_grad()
                pred = self.model(x_data, None, idx_base, x_mask)
                loss = self.criterion(pred.squeeze(), y_data.squeeze())
                loss.backward()
                warmup_optimizer.step()
                
                warmup_losses.append(loss.item())
            
            print(f"Warmup epoch {epoch} - Loss: {np.mean(warmup_losses):.4f}")

    def inner_loop(self, support_data, query_data=None):
        _, x_support, x_mask, y_support = support_data
        x_support = x_support.to(self.device).permute(0, 2, 1)
        x_mask = x_mask.to(self.device, dtype=torch.bool)
        y_support = y_support.to(self.device)
        
        batch_size, _, num_points = x_support.size()
        idx_base = torch.arange(0, batch_size, device=self.device).view(-1, 1, 1) * num_points
        
        local_model = copy.deepcopy(self.model)
        local_optimizer = optim.AdamW(local_model.parameters(), lr=self.inner_lr)
        
        support_losses = []
        for _ in range(self.num_inner_steps):
            support_pred = local_model(x_support, None, idx_base, x_mask)
            support_loss = self.criterion(support_pred.squeeze(), y_support.squeeze())
            
            local_optimizer.zero_grad()
            support_loss.backward()
            local_optimizer.step()
            
            support_losses.append(support_loss.item())
        
        if query_data is not None:
            _, x_query, x_mask, y_query = query_data
            x_query = x_query.to(self.device).permute(0, 2, 1)
            x_mask = x_mask.to(self.device, dtype=torch.bool)
            y_query = y_query.to(self.device)
            
            batch_size, _, num_points = x_query.size()
            idx_base = torch.arange(0, batch_size, device=self.device).view(-1, 1, 1) * num_points
            query_pred = local_model(x_query, None, idx_base, x_mask)
            query_loss = self.criterion(query_pred.squeeze(), y_query.squeeze())
            
            return query_loss, local_model.state_dict()
            
        return np.mean(support_losses)

    def meta_train_step(self, support_tasks, query_batch):
        meta_loss = 0.0
        self.meta_optimizer.zero_grad()
        
        for support_data in support_tasks:
            query_loss, _ = self.inner_loop(support_data, query_batch)
            meta_loss += query_loss
        
        meta_loss = meta_loss / len(support_tasks)
        meta_loss.backward()
        self.meta_optimizer.step()
        self.meta_scheduler.step(meta_loss.item())
        
        return meta_loss.item()

    def fit(self, train_task_paths, pretrain_task_path, valid_task_paths, checkpoint_path): 
        # Initialize datasets
        self._init_datasets(train_task_paths, pretrain_task_path, valid_task_paths)
        
        # Warmup phase
        if self.config['train'].get('warmup_epochs', 0) > 0:
            print("Starting warmup training...")
            self._warmup_train()
        
        # Meta-training phase
        print("Starting TSTLPre training...")
        best_valid_loss = float('inf')
        early_stop_patience = 0
        
        # Create checkpoint directory if it doesn't exist
	    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

        for epoch in range(1, self.config['train']['epochs'] + 1):
            # Training
            meta_losses = []
            # for batch in tqdm(self.meta_train_outer_loader, desc=f"Epoch {epoch}"):
            for batch in self.meta_train_outer_loader: 
                meta_loss = self.meta_train_step(
                    self.meta_train_inner_dataset.k_samples(epoch), 
                    batch
                )
                meta_losses.append(meta_loss)
            
            avg_meta_loss = np.mean(meta_losses)
            
            # Validation
            valid_losses = []
            for support_data in self.meta_valid_inner_dataset.k_samples(epoch):
                valid_losses.append(self.inner_loop(support_data))
            avg_valid_loss = np.mean(valid_losses)
            
            # Early stopping and checkpointing
            if avg_valid_loss < best_valid_loss:
                best_valid_loss = avg_valid_loss
                self.save_checkpoint(checkpoint_path, epoch, best_valid_loss)
                early_stop_patience = 0
            else:
                early_stop_patience += 1
                if early_stop_patience >= self.config['train']['early_stop_patience']:
                    print("Early stopping triggered!")
                    print(f"Best validation loss: {best_valid_loss:.4f}")
                    break
            
            print(f"Epoch {epoch} - Meta Loss: {avg_meta_loss:.4f}, Valid Loss: {avg_valid_loss:.4f}, Early stopping patience: {early_stop_patience}/{self.config['train']['early_stop_patience']}")
    
    def save_checkpoint(self, path, epoch, best_val_loss): 
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'meta_optimizer_state_dict': self.meta_optimizer.state_dict(),
            'meta_scheduler_state_dict': self.meta_scheduler.state_dict(),
            'best_val_mae': best_val_loss
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.meta_optimizer.load_state_dict(checkpoint['meta_optimizer_state_dict'])
        self.meta_scheduler.load_state_dict(checkpoint['meta_scheduler_state_dict'])
        return checkpoint['epoch'], checkpoint['best_val_mae']



class TSTL_FtTrainer:
    def __init__(self, model, config, device, seed=42):
        self.model = model
        self.config = config
        self.device = device
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def load_data(self, data_path, data_config):
        if data_path.endswith('.csv'):
            pkl_dict = csv2pkl_wfilter(data_path, data_config['encoding'])
            pkl_path = data_path.replace('.csv', '.pkl')
            with open(pkl_path, 'wb') as f:
                pickle.dump(pkl_dict, f)
        elif not data_path.endswith('.pkl'):
            raise ValueError('Unsupported data format:', data_path)
        else:
            with open(data_path, 'rb') as f:
                pkl_dict = pickle.load(f)

        all_data = []
        unique_smiles = set()
        for idx in range(len(pkl_dict)):
            smiles = pkl_dict[idx]['smiles']
            if smiles not in unique_smiles:
                all_data.append(pkl_dict[idx])
                unique_smiles.add(smiles)
        return all_data

    def eval_step(self, model, loader, scaler=None, return_predictions=False):
        model.eval()
        y_true, y_pred = [], []
        
        # for batch in tqdm(loader, desc='Eval'):
        for batch in loader: 
            _, x, mask, y = batch
            batch_size, num_points, _ = x.size()
            x = x.to(device=self.device, dtype=torch.float).permute(0, 2, 1)
            mask = mask.to(device=self.device, dtype=torch.bool)
            y = y.to(device=self.device, dtype=torch.float)
            idx_base = torch.arange(0, batch_size, device=self.device).view(-1, 1, 1) * num_points
            
            with torch.no_grad():
                pred = model(x, None, idx_base, mask).squeeze()
            
            if scaler:
                pred = torch.tensor(scaler.inverse_transform(
                    pred.detach().cpu().numpy().reshape(-1, 1))).squeeze().to(self.device)
                y = torch.tensor(scaler.inverse_transform(
                    y.detach().cpu().numpy().reshape(-1, 1))).squeeze().to(self.device)
            
            y_true.append(y.cpu().numpy())
            y_pred.append(pred.cpu().numpy())

        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        return (mae, r2, y_true, y_pred) if return_predictions else (mae, r2)

    def train_step(self, model, loader, optimizer, scaler=None):
        model.train()
        y_true, y_pred = [], []
        total_loss = 0
        
        # for batch in tqdm(loader, desc='Train'):
        for batch in loader: 
            _, x, mask, y = batch
            batch_size, num_points, _ = x.size()
            x = x.to(device=self.device, dtype=torch.float).permute(0, 2, 1)
            mask = mask.to(device=self.device, dtype=torch.bool)
            y = y.to(device=self.device, dtype=torch.float)
            idx_base = torch.arange(0, batch_size, device=self.device).view(-1, 1, 1) * num_points
            
            optimizer.zero_grad()
            pred = model(x, None, idx_base, mask).squeeze()
            loss = nn.HuberLoss(delta=1.0)(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            if scaler:
                pred = torch.tensor(scaler.inverse_transform(
                    pred.detach().cpu().numpy().reshape(-1, 1))).squeeze().to(self.device)
                y = torch.tensor(scaler.inverse_transform(
                    y.detach().cpu().numpy().reshape(-1, 1))).squeeze().to(self.device)
            
            y_true.append(y.cpu().numpy())
            y_pred.append(pred.cpu().numpy())

        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        return mean_absolute_error(y_true, y_pred), r2_score(y_true, y_pred), total_loss/len(loader)

    @staticmethod
    def average_model_weights(state_dicts): 
        if not state_dicts:
            raise ValueError("Empty state dictionaries list")
        
        avg_state_dict = {}
        keys = state_dicts[0].keys()
        
        if not all(set(d.keys()) == set(keys) for d in state_dicts[1:]):
            raise ValueError("Inconsistent state dictionary keys")
        
        for key in keys:
            avg_state_dict[key] = sum(d[key] for d in state_dicts) / len(state_dicts)
        
        return avg_state_dict

    def train_fold(self, model, train_loader, valid_loader, fold_idx, scaler, checkpoint_path): 
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config['train']['lr'],
            weight_decay=self.config['train']['weight_decay']
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=self.config['train']['lr_factor'],
            patience=self.config['train']['lr_patience'], min_lr=1e-7
        )

        best_valid_mae = float('inf')
        best_valid_r2 = float('-inf')
        early_stop_patience = 0
        early_stop_step = self.config['train']['early_stop_step']

        for epoch in range(1, self.config['train']['epochs'] + 1):
            train_mae, train_r2, train_loss = self.train_step(model, train_loader, optimizer, scaler)
            valid_mae, valid_r2, y_true, y_pred = self.eval_step(model, valid_loader, scaler, return_predictions=True)
            
            if valid_mae < best_valid_mae:
                best_valid_mae = valid_mae
                best_valid_r2 = valid_r2
                early_stop_patience = 0

                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_mae': best_valid_mae,
                    'fold': fold_idx
                }
                torch.save(checkpoint, checkpoint_path.replace('.pt', f'_fold{fold_idx}.pt'))
            else:
                early_stop_patience += 1
                if early_stop_patience >= early_stop_step:
                    print('Early stopping triggered')
                    break

            print(f"\nFold {fold_idx} Epoch {epoch}:")
            print(f"Train: MAE={train_mae:.4f}, R2={train_r2:.4f}, Loss={train_loss:.4f}")
            print(f"Valid: MAE={valid_mae:.4f}, R2={valid_r2:.4f}")
            print(f"Best valid MAE: {best_valid_mae:.4f}, R2: {best_valid_r2:.4f}")
            print(f"Early stopping patience: {early_stop_patience}/{early_stop_step}")

            scheduler.step(valid_mae)

        return best_valid_mae, best_valid_r2

    def create_model_soup(self, model, train_loader, valid_loader, fold_idx, scaler, pretrain_paths, checkpoint_path):
        checkpoints = [torch.load(path.replace('.pt', f'_fold{fold_idx}.pt'), map_location=self.device) 
                      for path in pretrain_paths]
        state_dicts = [checkpoint['model_state_dict'] for checkpoint in checkpoints]
        
        # Evaluate models
        metrics = []
        for idx, state_dict in enumerate(state_dicts):
            model.load_state_dict(state_dict)
            train_mae, train_r2 = self.eval_step(model, train_loader, scaler)
            harmonic_mean = 2 * (1/train_mae) * train_r2 / ((1/train_mae) + train_r2)
            metrics.append(harmonic_mean - (0.8 if idx < 2 else 0))
        
        # Sort and create soup
        soups = []
        best_harm = 0
        best_metrics = {'mae': float('inf'), 'r2': float('-inf')}
        
        for state_dict, metric in sorted(zip(state_dicts, metrics), key=lambda x: x[1], reverse=True):
            model.load_state_dict(state_dict)
            train_mae, train_r2 = self.eval_step(model, train_loader, scaler)
            
            if (train_mae > 0 and train_r2 > 0 and 
                2 * (1/train_mae) * train_r2 / ((1/train_mae) + train_r2) > best_harm):
                soups.append(state_dict)
                valid_mae, valid_r2 = self.eval_step(model, valid_loader, scaler)
                best_metrics = {'mae': valid_mae, 'r2': valid_r2}

        # Save soup model
        if soups: 
            torch.save(
                {'model_state_dict': self.average_model_weights(soups)},
                checkpoint_path.replace('.pt', f'_fold{fold_idx}.pt')
            )
        else:
            raise ValueError("No models selected for model soup")

        return best_metrics['mae'], best_metrics['r2']

    def fit(self, data_path, data_config, pretrained_paths=[], checkpoint_paths=[], 
           ensemble_path='', result_path='', k_folds=5): 
        """Run transfer learning followed by model soup creation."""
        all_data = self.load_data(data_path, data_config)
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=self.seed)
        transfer_metrics = {i: {'mae': [], 'r2': []} for i in range(len(pretrained_paths))}
        ensemble_metrics = {'mae': [], 'r2': []}

        # Create save directories if needed
        all_paths = checkpoint_paths + ([ensemble_path] if ensemble_path else [])
        for path in all_paths:
            if path:
                os.makedirs(os.path.dirname(path), exist_ok=True)

        # Perform k-fold cross validation
        for fold_idx, (train_idx, valid_idx) in enumerate(kf.split(all_data)):
            print(f'\nProcessing Fold {fold_idx + 1}/{k_folds}')
            
            # Prepare data for this fold
            train_data = [copy.deepcopy(all_data[i]) for i in train_idx]
            valid_data = [copy.deepcopy(all_data[i]) for i in valid_idx]
            
            # Create datasets
            train_set = MolRT_Dataset(train_data, mode='data', 
                                    add_noise=self.config['train']['add_noise'])
            scaler = train_set.return_scaler()
            valid_set = MolRT_Dataset(valid_data, mode='data', 
                                    add_noise=0, scaler=scaler)
            
            # Create dataloaders
            train_loader = DataLoader(
                train_set, 
                batch_size=self.config['train']['batch_size'],
                shuffle=True, 
                num_workers=self.config['train']['num_workers']
            )
            valid_loader = DataLoader(
                valid_set, 
                batch_size=self.config['train']['batch_size'],
                shuffle=False, 
                num_workers=self.config['train']['num_workers']
            )

            # Step 1: Transfer Learning Phase
            for idx, (resume_path, checkpoint_path) in enumerate(zip(pretrained_paths, checkpoint_paths)):
                print(f"\nTraining with pretrained model {idx + 1}/{len(pretrained_paths)}: {resume_path}")
                
                # Initialize model and load pretrained weights
                model = copy.deepcopy(self.model).to(self.device)
                if resume_path:
                    state_dict = torch.load(resume_path, map_location=self.device)['model_state_dict']
                    model.load_state_dict(
                        {k: v for k, v in state_dict.items() if not k.startswith("decoder")}, 
                        strict=False
                    )
                
                # Train model
                mae, r2 = self.train_fold(
                    model, train_loader, valid_loader,
                    fold_idx, scaler, checkpoint_path
                )
                
                # Store metrics
                transfer_metrics[idx]['mae'].append(mae)
                transfer_metrics[idx]['r2'].append(r2)
                print(f"Model {idx + 1} Fold {fold_idx + 1} - MAE: {mae:.4f}, R2: {r2:.4f}")

            # Step 2: Ensemble Phase (Model Soup)
            if ensemble_path:
                print(f"\nCreating model soup for fold {fold_idx + 1}")
                mae, r2 = self.create_model_soup(
                    self.model.to(self.device), 
                    train_loader, valid_loader,
                    fold_idx, scaler, checkpoint_paths, ensemble_path
                )
                ensemble_metrics['mae'].append(mae)
                ensemble_metrics['r2'].append(r2)
                print(f"Ensemble Fold {fold_idx + 1} - MAE: {mae:.4f}, R2: {r2:.4f}")

        # Save results
        if result_path:
            # Prepare transfer learning results
            transfer_data = []
            for model_idx in range(len(pretrained_paths)):
                metrics = transfer_metrics[model_idx]
                for fold_idx in range(k_folds):
                    transfer_data.append({
                        'Model': model_idx + 1,
                        'Fold': fold_idx + 1,
                        'MAE': metrics['mae'][fold_idx],
                        'R2': metrics['r2'][fold_idx]
                    })
                # Add summary statistics
                mae_mean, mae_std = np.mean(metrics['mae']), np.std(metrics['mae'])
                r2_mean, r2_std = np.mean(metrics['r2']), np.std(metrics['r2'])
                transfer_data.extend([
                    {'Model': model_idx + 1, 'Fold': 'mean', 'MAE': mae_mean, 'R2': r2_mean},
                    {'Model': model_idx + 1, 'Fold': 'std', 'MAE': mae_std, 'R2': r2_std}
                ])
            
            # Save transfer learning results
            pd.DataFrame(transfer_data).to_csv(
                result_path.replace('.csv', '_transfer.csv'), index=False)

            # Save ensemble results if available
            if ensemble_path:
                ensemble_data = []
                for fold_idx in range(k_folds):
                    ensemble_data.append({
                        'Fold': fold_idx + 1,
                        'MAE': ensemble_metrics['mae'][fold_idx],
                        'R2': ensemble_metrics['r2'][fold_idx]
                    })
                # Add summary statistics
                mae_mean, mae_std = np.mean(ensemble_metrics['mae']), np.std(ensemble_metrics['mae'])
                r2_mean, r2_std = np.mean(ensemble_metrics['r2']), np.std(ensemble_metrics['r2'])
                ensemble_data.extend([
                    {'Fold': 'mean', 'MAE': mae_mean, 'R2': r2_mean},
                    {'Fold': 'std', 'MAE': mae_std, 'R2': r2_std}
                ])
                pd.DataFrame(ensemble_data).to_csv(
                    result_path.replace('.csv', '_ensemble.csv'), index=False)

        # Print final results
        print("\n=== Final Results ===")
        print("\nTransfer Learning Results:")
        for model_idx in range(len(pretrained_paths)):
            metrics = transfer_metrics[model_idx]
            mae_mean, mae_std = np.mean(metrics['mae']), np.std(metrics['mae'])
            r2_mean, r2_std = np.mean(metrics['r2']), np.std(metrics['r2'])
            print(f"\nModel {model_idx + 1}:")
            print(f"MAE: {mae_mean:.4f} ± {mae_std:.4f}")
            print(f"R2: {r2_mean:.4f} ± {r2_std:.4f}")

        if ensemble_path:
            print("\nEnsemble Results:")
            mae_mean = np.mean(ensemble_metrics['mae'])
            mae_std = np.std(ensemble_metrics['mae'])
            r2_mean = np.mean(ensemble_metrics['r2'])
            r2_std = np.std(ensemble_metrics['r2'])
            print(f"MAE: {mae_mean:.4f} ± {mae_std:.4f}")
            print(f"R2: {r2_mean:.4f} ± {r2_std:.4f}")

        return transfer_metrics, ensemble_metrics
