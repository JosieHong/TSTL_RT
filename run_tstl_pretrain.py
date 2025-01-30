import os
import argparse
import yaml
import random
import numpy as np
import torch
from tstl import MolNet_RT, TSTL_PreTrainer



def init_random_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)

def main():
	parser = argparse.ArgumentParser(description='Molecular Retention Time Prediction (TSTL_RT)')
	parser.add_argument('--train_task_paths', type=str, nargs='+', required=True, 
					   help="Paths to the training data files")
	parser.add_argument('--pretrain_task_path', type=str, required=True, 
					   help="Path to the pretraining data file")
	parser.add_argument('--valid_task_paths', type=str, nargs='+', required=True,
					   help="Paths to the validation data files")
	parser.add_argument('--model_config', type=str, default='./config_pretrain.yml', 
						help='Path to model configuration')
	parser.add_argument('--data_config', type=str, default='./preprocess_etkdgv3.yml',
						help='path to data preprocessing configuration')
	parser.add_argument('--checkpoint_path', type=str, required=True, 
					   help='Path to save checkpoint')
	parser.add_argument('--seed', type=int, default=42,
					   help='Random seed')
	parser.add_argument('--device', type=int, default=0,
					   help='GPU device ID')							
	args = parser.parse_args()

	# Set random seed
	init_random_seed(args.seed)

	# Set device
	device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
	
	# Load configuration
	with open(args.model_config, 'r') as f:
		config = yaml.load(f, Loader=yaml.FullLoader)
	with open(args.data_config, 'r') as f:
		data_config = yaml.load(f, Loader=yaml.FullLoader)

	# Initialize model
	model = MolNet_RT(config['model'])

	# Initialize trainer
	trainer = TSTL_PreTrainer(
		model=model,
		config=config,
		data_config=data_config, 
		device=device
	)

	# Train model
	trainer.fit(
		train_task_paths=args.train_task_paths,
		pretrain_task_path=args.pretrain_task_path,
		valid_task_paths=args.valid_task_paths,
		checkpoint_path=args.checkpoint_path
	)

if __name__ == "__main__":
	main()