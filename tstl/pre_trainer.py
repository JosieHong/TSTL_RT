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
from .utils import csv2pkl_wfilter



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