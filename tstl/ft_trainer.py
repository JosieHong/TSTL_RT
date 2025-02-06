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

from .dataset import MetaMolRT_Dataset, MolRT_Dataset
from .utils import csv2pkl_wfilter, LogHandler

class TSTL_FtTrainer:
	def __init__(self, model, config, device, seed=42):
		self.model = model
		self.config = config
		self.device = device
		self.seed = seed
		random.seed(seed)
		self._set_seeds(seed)

		self.all_data = []

	def _set_seeds(self, seed): 
		for seeder in [np.random.seed, torch.manual_seed, torch.cuda.manual_seed]:
			seeder(seed)

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

		unique_smiles = set()
		for idx in range(len(pkl_dict)):
			smiles = pkl_dict[idx]['smiles']
			if smiles not in unique_smiles:
				self.all_data.append(pkl_dict[idx])
				unique_smiles.add(smiles)

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
			
	def train_fold(self, model, train_loader, valid_loader, fold_idx, pretrain_idx, scaler, checkpoint_path, log_path=None): 
		with LogHandler(log_path) as log_handler: 
			print(f"\nModel {pretrain_idx + 1} - Training fold {fold_idx + 1}...")

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
						log_handler.write("Early stopping triggered")
						break

				log_handler.write(
					f"\nModel {pretrain_idx} Fold {fold_idx} Epoch {epoch}:\n"
					f"Train: MAE={train_mae:.4f}, R2={train_r2:.4f}, Loss={train_loss:.4f}\n"
					f"Valid: MAE={valid_mae:.4f}, R2={valid_r2:.4f}\n"
					f"Best valid MAE: {best_valid_mae:.4f}, R2: {best_valid_r2:.4f}\n"
					f"Early stopping patience: {early_stop_patience}/{early_stop_step}"
				)

				scheduler.step(valid_mae)

		return best_valid_mae, best_valid_r2

	@staticmethod
	def _average_model_weights(state_dicts): 
		if not state_dicts:
			raise ValueError("Empty state dictionaries list")
		
		avg_state_dict = {}
		keys = state_dicts[0].keys()
		
		if not all(set(d.keys()) == set(keys) for d in state_dicts[1:]):
			raise ValueError("Inconsistent state dictionary keys")
		
		for key in keys:
			avg_state_dict[key] = sum(d[key] for d in state_dicts) / len(state_dicts)
		
		return avg_state_dict

	def create_model_soup(self, model, train_loader, valid_loader, fold_idx, scaler, checkpoint_paths, 
					ensemble_path, log_path=None): 
		with LogHandler(log_path) as log_handler:
			checkpoints = [torch.load(path.replace('.pt', f'_fold{fold_idx}.pt'), map_location=self.device) 
						for path in checkpoint_paths]
			state_dicts = [checkpoint['model_state_dict'] for checkpoint in checkpoints]
			
			# Evaluate models
			metrics = []
			for idx, state_dict in enumerate(state_dicts): 
				model.load_state_dict(state_dict)
				train_mae, train_r2 = self.eval_step(model, train_loader, scaler)
				harmonic_mean = 2 * (1/train_mae) * train_r2 / ((1/train_mae) + train_r2)
				metrics.append(harmonic_mean - (0.8 if idx < 2 else 0))
				log_handler.write(f"Model {idx + 1} Evaluation - MAE: {train_mae:.4f}, R2: {train_r2:.4f}")
			
			# Sort and create soup
			soups = []
			best_harm = 0
			best_metrics = {'MAE': float('inf'), 'R2': float('-inf')}
			
			for state_dict, metric in sorted(zip(state_dicts, metrics), key=lambda x: x[1], reverse=True):
				model.load_state_dict(state_dict)
				train_mae, train_r2 = self.eval_step(model, train_loader, scaler)
				
				current_harm = 2 * (1/train_mae) * train_r2 / ((1/train_mae) + train_r2)
				if train_mae > 0 and train_r2 > 0 and current_harm > best_harm:
					soups.append(state_dict)
					best_harm = current_harm
					valid_mae, valid_r2 = self.eval_step(model, valid_loader, scaler)
					best_metrics = {'MAE': valid_mae, 'R2': valid_r2}
					log_handler.write(f"Added model to soup - Valid MAE: {valid_mae:.4f}, R2: {valid_r2:.4f}")
			
			# Save soup model
			if soups:
				torch.save(
					{'model_state_dict': self._average_model_weights(soups)},
					ensemble_path.replace('.pt', f'_fold{fold_idx}.pt')
				)
				log_handler.write(f"Saved model soup with {len(soups)} models")
			else:
				error_msg = "No models selected for model soup"
				log_handler.write(error_msg)
				raise ValueError(error_msg)

			return best_metrics['MAE'], best_metrics['R2']

	def fit(self, data_path, data_config, pretrained_paths=[], checkpoint_paths=[], 
		   ensemble_path='', result_path='', log_dir='', k_folds=5): 
		self.load_data(data_path, data_config) # load data to self.all_data
		kf = KFold(n_splits=k_folds, shuffle=True, random_state=self.seed)
		transfer_metrics = {i: {'MAE': [], 'R2': []} for i in range(len(pretrained_paths))}
		ensemble_metrics = {'MAE': [], 'R2': []}

		# Create save directories if needed
		all_paths = checkpoint_paths + ([ensemble_path] if ensemble_path else []) + ([result_path] if result_path else []) + ([log_dir] if log_dir else [])
		for path in all_paths:
			if path:
				os.makedirs(os.path.dirname(path), exist_ok=True)

		# Perform k-fold cross validation
		for fold_idx, (train_idx, valid_idx) in enumerate(kf.split(self.all_data)):
			print(f'\nProcessing data for Fold {fold_idx + 1}/{k_folds}')
			
			# Prepare data for this fold
			train_data = [copy.deepcopy(self.all_data[i]) for i in train_idx]
			valid_data = [copy.deepcopy(self.all_data[i]) for i in valid_idx]
			
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
			for model_idx, (resume_path, checkpoint_path) in enumerate(zip(pretrained_paths, checkpoint_paths)):
				print(f"\nTraining with pretrained model {model_idx + 1}/{len(pretrained_paths)}: {resume_path}")
				
				# Initialize model and load pretrained weights
				model = copy.deepcopy(self.model).to(self.device)
				if resume_path:
					state_dict = torch.load(resume_path, map_location=self.device)['model_state_dict']
					model.load_state_dict(
						{k: v for k, v in state_dict.items() if not k.startswith("decoder")}, 
						strict=False
					)
				
				# Train model
				if log_dir: 
					log_file = checkpoint_path.split('/')[-1].replace('.pt', f'_fold{fold_idx}.log')
					log_path = os.path.join(log_dir, log_file)
				else:
					log_path = None
				mae, r2 = self.train_fold(
					model, train_loader, valid_loader,
					fold_idx, model_idx, scaler, checkpoint_path, 
					log_path=log_path,  
				)
				
				# Store metrics
				transfer_metrics[model_idx]['MAE'].append(mae)
				transfer_metrics[model_idx]['R2'].append(r2)
				print(f"\nModel {model_idx + 1} Fold {fold_idx + 1} - MAE: {mae:.4f}, R2: {r2:.4f}")

			# Step 2: Ensemble Phase (Model Soup)
			if ensemble_path:
				print(f"\nCreating model soup for fold {fold_idx + 1}")
				mae, r2 = self.create_model_soup(
					self.model.to(self.device), 
					train_loader, valid_loader,
					fold_idx, scaler, checkpoint_paths, ensemble_path,
					log_path=ensemble_path.replace('.pt', f'_fold{fold_idx}.log')
				)
				ensemble_metrics['MAE'].append(mae)
				ensemble_metrics['R2'].append(r2)
				print(f"Ensemble Fold {fold_idx + 1} - MAE: {mae:.4f}, R2: {r2:.4f}")

		# Save results
		if result_path:
			self._save_results(transfer_metrics, ensemble_metrics, result_path, len(pretrained_paths), k_folds)

		# Print final results
		self._print_final_results(transfer_metrics, ensemble_metrics, len(pretrained_paths), ensemble_path)

		return transfer_metrics, ensemble_metrics
	
	def _save_results(self, transfer_metrics, ensemble_metrics, result_path, pretrain_number, k_folds): 
		# Prepare transfer learning results
		transfer_data = []
		for model_idx in range(pretrain_number): 
			metrics = transfer_metrics[model_idx]
			for fold_idx in range(k_folds):
				transfer_data.append({
					'Model': model_idx + 1,
					'Fold': fold_idx + 1,
					'MAE': metrics['MAE'][fold_idx],
					'R2': metrics['R2'][fold_idx]
				})
			# Add summary statistics
			mae_mean, mae_std = np.mean(metrics['MAE']), np.std(metrics['MAE'])
			r2_mean, r2_std = np.mean(metrics['R2']), np.std(metrics['R2'])
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
					'MAE': ensemble_metrics['MAE'][fold_idx],
					'R2': ensemble_metrics['R2'][fold_idx]
				})
			# Add summary statistics
			mae_mean, mae_std = np.mean(ensemble_metrics['MAE']), np.std(ensemble_metrics['MAE'])
			r2_mean, r2_std = np.mean(ensemble_metrics['R2']), np.std(ensemble_metrics['R2'])
			ensemble_data.extend([
				{'Fold': 'mean', 'MAE': mae_mean, 'R2': r2_mean},
				{'Fold': 'std', 'MAE': mae_std, 'R2': r2_std}
			])
			pd.DataFrame(ensemble_data).to_csv(
				result_path.replace('.csv', '_ensemble.csv'), index=False)

	def _print_final_results(self, transfer_metrics, ensemble_metrics, pretrain_number, ensemble_path): 
		print("\n=== Final Results ===")
		print("\nTransfer Learning Results:")
		for model_idx in range(pretrain_number):
			metrics = transfer_metrics[model_idx]
			mae_mean, mae_std = np.mean(metrics['MAE']), np.std(metrics['MAE'])
			r2_mean, r2_std = np.mean(metrics['R2']), np.std(metrics['R2'])
			print(f"Model {model_idx + 1}:")
			print(f"MAE: {mae_mean:.4f} ± {mae_std:.4f}")
			print(f"R2: {r2_mean:.4f} ± {r2_std:.4f}\n")

		if ensemble_path:
			print("Ensemble Results:")
			mae_mean = np.mean(ensemble_metrics['MAE'])
			mae_std = np.std(ensemble_metrics['MAE'])
			r2_mean = np.mean(ensemble_metrics['R2'])
			r2_std = np.std(ensemble_metrics['R2'])
			print(f"MAE: {mae_mean:.4f} ± {mae_std:.4f}")
			print(f"R2: {r2_mean:.4f} ± {r2_std:.4f}")

