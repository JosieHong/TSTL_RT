import re
import numpy as np
import pickle
import random
import copy
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset

from utils import csv2pkl_wfilter, conformation_array

class MetaMolRT_Dataset(Dataset): 
	"""Extended dataset class for meta-learning"""
	def __init__(self, data_paths, data_config, device, k_shot=None, scalers=None): 
		self.datasets = []
		self.device = device
		self.k_shot = k_shot
		assert k_shot or len(data_paths) == 1, "Please provide only one dataset for full-batch training"
		self.scalers = {}

		# Load multiple datasets
		for path in data_paths:
			subset_id = self._get_subsetid_from_path(path)

			if path.endswith('.csv'):
				pkl_dict = csv2pkl_wfilter(path, data_config['encoding'])
				if len(pkl_dict) == 0:
					print('No valid data in {}'.format(path))
					continue
				
				# save in pkl format
				pkl_path = path.replace('.csv', '.pkl')
				with open(pkl_path, 'wb') as f:
					pickle.dump(pkl_dict, f)
				print('Save pkl file: {}'.format(pkl_path))

				# scale retention times 
				if scalers is not None:
					if subset_id not in scalers.keys(): 
						raise ValueError(f"Scaler not found for {subset_id}")

					self.scalers[subset_id] = scalers[subset_id]
					rt_values = np.array([item['rt'] for item in pkl_dict]).reshape(-1, 1)
					scaled_rt = scalers[subset_id].transform(rt_values)
					for idx, rt in enumerate(scaled_rt):
						pkl_dict[idx]['rt'] = rt.item()
				else:
					self.scalers[subset_id] = StandardScaler()
					rt_values = np.array([item['rt'] for item in pkl_dict]).reshape(-1, 1)
					scaled_rt = self.scalers[subset_id].fit_transform(rt_values)
					print('Scaler mean: {}, std: {}'.format(self.scalers[subset_id].mean_, self.scalers[subset_id].scale_))
					for idx, rt in enumerate(scaled_rt): 
						pkl_dict[idx]['rt'] = rt.item()

				# generate mask
				for idx in range(len(pkl_dict)): 
					mask = ~np.all(pkl_dict[idx]['mol'] == 0, axis=1)
					pkl_dict[idx]['mask'] = mask.astype(bool)

				self.datasets.append(pkl_dict)
			
			elif path.endswith('.pkl'):
				with open(path, 'rb') as f:
					pkl_dict = pickle.load(f)

				# scale retention times 
				if scalers is not None:
					if subset_id not in scalers.keys(): 
						raise ValueError(f"Scaler not found for {path}")

					self.scalers[subset_id] = scalers[subset_id]
					rt_values = np.array([item['rt'] for item in pkl_dict]).reshape(-1, 1)
					scaled_rt = scalers[subset_id].transform(rt_values)
					for idx, rt in enumerate(scaled_rt):
						pkl_dict[idx]['rt'] = rt.item()
				else:
					self.scalers[subset_id] = StandardScaler()
					rt_values = np.array([item['rt'] for item in pkl_dict]).reshape(-1, 1)
					scaled_rt = self.scalers[subset_id].fit_transform(rt_values)
					print('Scaler mean: {}, std: {}'.format(self.scalers[subset_id].mean_, self.scalers[subset_id].scale_))
					for idx, rt in enumerate(scaled_rt): 
						pkl_dict[idx]['rt'] = rt.item()

				# generate mask
				for idx in range(len(pkl_dict)): 
					mask = ~np.all(pkl_dict[idx]['mol'] == 0, axis=1)
					pkl_dict[idx]['mask'] = mask.astype(bool)

				self.datasets.append(pkl_dict)

		self.num_tasks = len(self.datasets)

	def _get_subsetid_from_path(self, file_path): 
		subset_id = re.search(r'/(\d{4})_', file_path)
		if subset_id:
			subset_id = subset_id.group(1)
		else:
			subset_id = 'all'
		return subset_id

	def k_samples(self, seed): 
		"""Sample k-shot tasks for inner loop"""
		random.seed(seed)

		inner_datasets = []
		for dataset in self.datasets:
			indices = random.sample(range(len(dataset)), self.k_shot)
			inner_datasets.append(self._prepare_batch([dataset[i] for i in indices]))
		
		return inner_datasets # replace the original datasets with k-shot datasets

	def _prepare_batch(self, batch_data): 
		"""Prepare a batch of data for the model"""
		titles = [item['title'] for item in batch_data]
		x = torch.FloatTensor(np.array([item['mol'] for item in batch_data]))
		x_mask = torch.BoolTensor(np.array([item['mask'] for item in batch_data]))
		y = torch.FloatTensor(np.array([item['rt'] for item in batch_data]))
		
		return titles, x, x_mask, y

	def __len__(self): # for meta-leanring only! 
		"""Get length for meta-learning"""
		assert self.k_shot == None, "Please use k_samples() for k-shot tasks"
		return len(self.datasets[0])

	def __getitem__(self, idx): # for meta-leanring only! 
		"""Get a batch of data for meta-learning"""
		assert self.k_shot == None, "Please use k_samples() for k-shot tasks"
		dataset = self.datasets[0]

		return (dataset[idx]['title'], 
				torch.FloatTensor(np.array(dataset[idx]['mol'])), # this is single sample, so no batch dimension 
				torch.FloatTensor(np.array(dataset[idx]['mask'])),
				torch.FloatTensor(np.array([dataset[idx]['rt']])))

class MolRT_Dataset(Dataset): 
	def __init__(self, x, mode='path', add_noise=0, scaler=None): 
		assert mode in ['path', 'data']
		if mode == 'path': 
			with open(x, 'rb') as file: 
				self.data = pickle.load(file)
			print('Load {} data from {}'.format(len(self.data), x))
		elif mode == 'data': 
			self.data = x

		# Scale retention times 
		if scaler is not None:
			self.scaler = scaler
			rt_values = np.array([item['rt'] for item in self.data]).reshape(-1, 1)
			scaled_rt = self.scaler.transform(rt_values)
		else:
			rt_values = np.array([item['rt'] for item in self.data]).reshape(-1, 1)
			self.scaler = StandardScaler()
			scaled_rt = self.scaler.fit_transform(rt_values)
			print('Fitted scaler with mean: {} and std: {}'.format(self.scaler.mean_, self.scaler.scale_))
		for idx, rt in enumerate(scaled_rt):
			self.data[idx]['rt'] = rt.item()

		# Generate mask
		for idx in range(len(self.data)): 
			mask = ~np.all(self.data[idx]['mol'] == 0, axis=1)
			self.data[idx]['mask'] = mask.astype(bool)

		# Add noise
		for i in range(add_noise): 
			for idx in range(len(self.data)): 
				new_data = copy.deepcopy(self.data[idx])
				new_data['mol'] = self._add_noise(new_data['mol'], new_data['mask'])
				self.data.append(new_data)

	def return_scaler(self): 
		if self.scaler is None: 
			raise ValueError('No scaler fitted!')
		return self.scaler

	def _add_noise(self, mol, mask): 
		num_atoms = np.sum(mask)
		num_noisy_atoms = max(1, int(0.2 * num_atoms)) # Try this! 
		noisy_indices = np.random.choice(num_atoms, num_noisy_atoms, replace=False)
		xyz_corr = mol[noisy_indices, :3]
		xyz_corr += np.random.normal(0, 0.02, xyz_corr.shape)
		mol[noisy_indices, :3] = xyz_corr
		return mol

	def __len__(self): 
		return len(self.data)

	def __getitem__(self, idx): 
		return (self.data[idx]['title'], 
				self.data[idx]['mol'], 
				self.data[idx]['mask'], 
				self.data[idx]['rt'])