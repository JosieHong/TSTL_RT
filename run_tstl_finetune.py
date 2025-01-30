import os
import argparse
import yaml
import torch
from model import MolNet_RT
from trainer import TSTL_FtTrainer

def main():
	parser = argparse.ArgumentParser(description='Molecular Retention Time Prediction')
	parser.add_argument('--data', type=str, default='./data/0009_rt.csv',
					  help='path to data (csv)')
	parser.add_argument('--model_config', type=str, default='./config_finetune.yml', 
						help='Path to model configuration')
	parser.add_argument('--data_config', type=str, default='./preprocess_etkdgv3.yml',
						help='path to data preprocessing configuration')
	parser.add_argument('--checkpoint_paths', type=str, nargs='+', default=[],
					  help='paths to save checkpoints for each pretrained model')
	parser.add_argument('--ensemble_path', type=str, default='',
					  help='path to save ensemble model')
	parser.add_argument('--result', type=str, default='',
					  help='results save path')
	parser.add_argument('--pretrained_paths', type=str, nargs='+', default=[],
					  help='paths to different pretrained models')
	parser.add_argument('--seed', type=int, default=42,
					  help='random seed')
	parser.add_argument('--device', type=int, default=0,
					  help='GPU device ID')
	parser.add_argument('--cpu', action='store_true',
					  help='use CPU')
	parser.add_argument('--folds', type=int, default=5,
					  help='number of CV folds')
	args = parser.parse_args()

	if len(args.pretrained_paths) != len(args.checkpoint_paths):
		raise ValueError("Number of pretrained_paths must match checkpoint_paths")
	
	# Load configs
	with open(args.model_config) as f, open(args.data_config) as g:
		config = yaml.safe_load(f)
		data_config = yaml.safe_load(g)

	# Setup device
	device = torch.device('cpu' if args.cpu else f'cuda:{args.device}')
	print(f'Using device: {device}')

	# Initialize model
	model = MolNet_RT(config['model'])

	# Initialize trainer
	trainer = TSTL_FtTrainer(model, config, device, args.seed)

	# Run transfer learning followed by model soup creation
	trainer.fit(
		data_path=args.data,
		data_config=data_config, 
		pretrained_paths=args.pretrained_paths, 
		checkpoint_paths=args.checkpoint_paths,
		ensemble_path=args.ensemble_path,
		k_folds=args.folds,
		result_path=args.result,
	)

if __name__ == "__main__":
	main()