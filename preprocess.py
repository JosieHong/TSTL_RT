import os
import pandas as pd
import json
import random

from rdkit import Chem
# ignore the warning
from rdkit import RDLogger 
RDLogger.DisableLog('rdApp.*')



mannually_removed_subsets = [
	# manually discarded 23 datasets for which retention time data 
	# was only recorded fora small section of the gradient
	"0006", "0008", "0023", "0059", "0123", "0128", 
	"0130", "0131", "0132", "0133", "0136", "0137", 
	"0139", "0143", "0145", "0148", "0149", "0151", 
	"0152", "0154", "0155", "0156", "0157",

	# manually discarded two datasets that were measured under nominally 
	# the same setup, but showed a suspiciously high number of conflicting pairs
	"0056", "0057", 
	
	# One dataset uses a step-wise gradient, unlike the ramp gradient of all other datasets
	"0024"

	# Suspicious subsets in meta-data, see `./RepoRT/processed_data/studies.tsv`
	"0004", "0005", "0015", "0021", "0041", 
]

pretrain_subsets = {
	"0186": 0, "0390": 1, "0391": 2, 
}

required_columns = [
	# Column properties
	"id", "column.name", "column.length", "column.particle.size", 
	
	# Temperature
	"column.temperature", 
	
	# Flow rate
	"column.flowrate", 

	# pH
	"eluent.A.pH", "eluent.B.pH", "eluent.C.pH", "eluent.D.pH",

	# Gradient
	"gradient.start.A", "gradient.start.B", "gradient.start.C", "gradient.start.D", 
	"gradient.end.A", "gradient.end.B", "gradient.end.C", "gradient.end.D",
]

def check_metadata(path):
	df = pd.read_csv(path, sep='\t')
	for col in required_columns:
		if pd.isnull(df[col].iloc[0]): 
			return False
	return True



if __name__ == "__main__":
	root_dir = "./RepoRT/processed_data"
	out_dir = "./data/raw_all/"
	benchmark_dir = "./data/benchmark/"
	explore_dir = "./data/explore/"
	h_threshold = 200 # threshold for benchmark
	l_threshold = 100 # threshold for explore
	seed = 42
	train_ratio = 0.4 # this 'train' mean 'pretrain'

	random.seed(seed) # Set random seed

	if not os.path.exists(root_dir):
		raise FileNotFoundError("Processed data not found. Please dowload the data first.")

	os.makedirs(out_dir, exist_ok=True)
	os.makedirs(benchmark_dir, exist_ok=True)
	os.makedirs(explore_dir, exist_ok=True)

	discarded_record = {
		'manually_removed': [],
		'too_few_samples': [],
		# 'missing_required_condition': [],
		# 'missing_metadata': [],
		'missing_rt_data': [], 
		'pretrain': {}, 
		'benchmark': [],
		'explore': [],
	}
	success_num = 0
	pretrained_dfs = {}

	subset_list = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]
	for subset in subset_list: 
		# Filter 1: check if the subset is manually removed
		subset_id = subset.split('_')[0] 
		if subset in mannually_removed_subsets: 
			print('Manually removed: {}'.format(subset))
			discarded_record['manually_removed'].append(subset)
			continue

		# # Filter 2: check if the subset has the required columns in metadata
		# metadata_path = os.path.join(root_dir, subset, "{}_metadata.tsv".format(subset))
		# if not os.path.exists(metadata_path):
		#     print('Metadata not found: {}'.format(metadata_path))
		#     discarded_record['missing_metadata'].append(subset)
		#     continue
		# if not check_metadata(metadata_path):
		#     print('Missing required columns: {}'.format(metadata_path))
		#     discarded_record['missing_required_condition'].append(subset)
		#     continue

		subset_path = os.path.join(root_dir, subset, "{}_rtdata_isomeric_success.tsv".format(subset))
		if not os.path.exists(subset_path):
			print('File not found: {}'.format(subset_path))
			discarded_record['missing_rt_data'].append(subset)
			continue
		df = pd.read_csv(subset_path, sep='\t')
		
		# unify the smiles format
		df['smiles'] = df['smiles.std'].apply(lambda x: Chem.MolToSmiles(Chem.MolFromSmiles(x), isomericSmiles=True))
		
		# average the rt on the same compound
		avg_rt_df = df.groupby('smiles')['rt'].mean().reset_index()
		avg_rt_df['id'] = avg_rt_df.index

		# split the dataset
		# Randomly split the dataset into training tasks and validation tasks
		n_train = int(len(df) * train_ratio)
		train_df = df.sample(n=n_train, random_state=seed)
		valid_df = df.drop(train_df.index)

		# save the processed data
		if subset_id in pretrain_subsets.keys(): 
			# out_path = os.path.join(out_dir, "pretrain_{}_rt.csv".format(subset))
			if pretrain_subsets[subset_id] not in pretrained_dfs.keys():
				pretrained_dfs[pretrain_subsets[subset_id]] = avg_rt_df
			else: 
				pretrained_dfs[pretrain_subsets[subset_id]] = pd.concat([pretrained_dfs[pretrain_subsets[subset_id]], avg_rt_df], ignore_index=True)
			discarded_record['pretrain'][subset_id] = pretrain_subsets[subset_id]
		else: 
			if len(valid_df) >= h_threshold: 
				train_df.to_csv(os.path.join(benchmark_dir, f'{subset_id}_rt_train.csv'), index=False)
				valid_df.to_csv(os.path.join(benchmark_dir, f'{subset_id}_rt_valid.csv'), index=False)
				discarded_record['benchmark'].append(subset)
			elif len(valid_df) >= l_threshold:
				train_df.to_csv(os.path.join(explore_dir, f'{subset_id}_rt_train.csv'), index=False)
				valid_df.to_csv(os.path.join(explore_dir, f'{subset_id}_rt_valid.csv'), index=False)
				discarded_record['explore'].append(subset)
			else:
				discarded_record['too_few_samples'].append(subset)
				continue

			out_path = os.path.join(out_dir, "{}_rt.csv".format(subset))
			avg_rt_df.to_csv(out_path, index=False)
			print('Save {} rt ({} train, {} valid) to {}'.format(len(avg_rt_df), 
																	len(train_df), 
																	len(valid_df), 
																	out_path))
			success_num += 1

	# save the discarded record
	out_path = os.path.join(out_dir, "discarded_record.json")
	with open(out_path, 'w') as f:
		json.dump(discarded_record, f, indent=4)
	print('Save discarded record to {}'.format(out_path))

	# save the pretrain data
	for subset_id, df in pretrained_dfs.items():
		out_path = os.path.join(out_dir, "pretrain_{}_rt.csv".format(subset_id))
		df.to_csv(out_path, index=False)
		print('Save pretrain {} rt to {}'.format(len(df), out_path))
		
	print('='*20)
	for k, v in discarded_record.items(): 
		if k != 'pretrain':
			print('{}: {}'.format(k, len(v)))
		else:
			print('pretrain: {}'.format(v))
	print('success: {}'.format(success_num))
	print('='*20)
	print('Done.')