import os
import copy
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import random
from torch.multiprocessing import Process, Queue
import torch.multiprocessing as mp
from typing import Dict, Any
from sklearn.model_selection import KFold

from .ft_trainer import TSTL_FtTrainer
from .dataset import MetaMolRT_Dataset, MolRT_Dataset
from .utils import csv2pkl_wfilter

class TSTL_Parallel_FtTrainer(TSTL_FtTrainer):
	def __init__(self, model, config, device, seed=42):
		super().__init__(model, config, device, seed)
		self.model = model
		self.config = config
		self.device = device
		self.seed = seed
		random.seed(seed)
		self._set_seeds(seed)

		self.all_data = []

	def _process_model(self, process_model_input: Dict[str, Any], queue: Queue): 
		model_idx = process_model_input['model_idx']
		pretrained_path = process_model_input['pretrained_path']
		checkpoint_path = process_model_input['checkpoint_path']
		log_dir = process_model_input['log_dir']
		k_folds = process_model_input['k_folds']
		kf = KFold(n_splits=k_folds, shuffle=True, random_state=self.seed)
		
		# Initialize metrics storage for this model
		model_metrics = {'MAE': [], 'R2': []} 

		# Process each fold for this model
		for fold_idx, (train_idx, valid_idx) in enumerate(kf.split(self.all_data)):
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

			print(f"\nModel {model_idx + 1} - Processing data for Fold {fold_idx + 1}/{k_folds}")
			
			# Initialize model and load pretrained weights
			model = copy.deepcopy(self.model).to(self.device)
			if pretrained_path:
				state_dict = torch.load(pretrained_path, map_location=self.device)['model_state_dict']
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
			
			# Store metrics for this fold
			model_metrics['MAE'].append(mae)
			model_metrics['R2'].append(r2)
			print(f"Model {model_idx + 1} Fold {fold_idx + 1} - MAE: {mae:.4f}, R2: {r2:.4f}")
		
		# Put results in queue
		queue.put((model_idx, model_metrics))

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

		# Enable multiprocessing for CUDA
		mp.set_start_method('spawn', force=True)
		
		# Create queue and processes
		queue = Queue()
		processes = []
		
		# Step 1: Transfer Learning Phase
		for model_idx, (pretrained_path, checkpoint_path) in enumerate(zip(pretrained_paths, checkpoint_paths)):
			process_model_input = {
				'model_idx': model_idx,
				'pretrained_path': pretrained_path,
				'checkpoint_path': checkpoint_path,
				'log_dir': log_dir, 
				'k_folds': k_folds
			}
			
			p = Process(target=self._process_model, args=(process_model_input, queue))
			processes.append(p)
			p.start()
		
		# Collect results from queue		
		for _ in range(len(pretrained_paths)):
			model_idx, model_metrics = queue.get()
			transfer_metrics[model_idx] = model_metrics
		
		# Wait for all processes to complete
		for p in processes:
			p.join()
		
		# Step 2: Ensemble Phase (Model Soup)
		if ensemble_path:
			for fold_idx, (train_idx, valid_idx) in enumerate(kf.split(self.all_data)): 
				print(f"\nCreating model soup for fold {fold_idx + 1}")
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

				mae, r2 = self.create_model_soup(
					self.model.to(self.device), 
					train_loader, valid_loader,
					fold_idx, scaler, checkpoint_paths, ensemble_path, 
					log_path=checkpoint_path.replace('.pt', f'_fold{fold_idx}.log'), 
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