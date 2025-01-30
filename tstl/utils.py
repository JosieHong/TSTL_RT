import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from copy import deepcopy

from rdkit import Chem
# ignore the warning
from rdkit import RDLogger 
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import AllChem, rdDepictor



def conformation_array(smiles, conf_type): 
	# convert smiles to molecule
	if conf_type == 'etkdg': 
		mol = Chem.MolFromSmiles(smiles)
		mol_from_smiles = Chem.AddHs(mol)
		AllChem.EmbedMolecule(mol_from_smiles)

	elif conf_type == 'etkdgv3': 
		mol = Chem.MolFromSmiles(smiles)
		mol_from_smiles = Chem.AddHs(mol)
		ps = AllChem.ETKDGv3()
		ps.randomSeed = 0xf00d
		AllChem.EmbedMolecule(mol_from_smiles, ps) 

	elif conf_type == '2d':
		mol = Chem.MolFromSmiles(smiles)
		mol_from_smiles = Chem.AddHs(mol)
		rdDepictor.Compute2DCoords(mol_from_smiles)

	elif conf_type == 'omega': 
		raise ValueError('OMEGA conformation will be supported soon. ')
	else:
		raise ValueError('Unsupported conformation type. {}'.format(conf_type))

	# get the x,y,z-coordinates of atoms
	try: 
		conf = mol_from_smiles.GetConformer()
	except:
		return False, None, None
	xyz_arr = conf.GetPositions()
	# center the x,y,z-coordinates
	centroid = np.mean(xyz_arr, axis=0)
	xyz_arr -= centroid
	
	# concatenate with atom attributes
	xyz_arr = xyz_arr.tolist()
	for i, atom in enumerate(mol_from_smiles.GetAtoms()):
		xyz_arr[i] += [atom.GetDegree()]
		xyz_arr[i] += [atom.GetExplicitValence()]
		xyz_arr[i] += [atom.GetMass()/100]
		xyz_arr[i] += [atom.GetFormalCharge()]
		xyz_arr[i] += [atom.GetNumImplicitHs()]
		xyz_arr[i] += [int(atom.GetIsAromatic())]
		xyz_arr[i] += [int(atom.IsInRing())]
	xyz_arr = np.array(xyz_arr)
	
	# get the atom types of atoms
	atom_type = [atom.GetSymbol() for atom in mol_from_smiles.GetAtoms()]
	return True, xyz_arr, atom_type

def compound_filter(smiles, encoder):  
	'''
	Filter out the unsupported compounds. 
	'''
	# mol array
	good_conf, xyz_arr, atom_type = conformation_array(smiles=smiles, 
														conf_type=encoder['conf_type']) 
	if not good_conf:
		print('Can not generate correct conformation: {}'.format(smiles))
		return False
	if xyz_arr.shape[0] > encoder['max_atom_num']: 
		print('Atomic number ({}) exceed the limitation ({})'.format(encoder['max_atom_num'], xyz_arr.shape[0]))
		return False
	
	# atom types
	rare_atom_flag = False
	rare_atom = ''
	for atom in list(set(atom_type)):
		if atom not in encoder['atom_type'].keys(): 
			rare_atom_flag = True
			rare_atom = atom
			break
	if rare_atom_flag:
		print('Unsupported atom type: {}'.format(rare_atom))
		return False
	
	return True

def csv2pkl_wfilter(csv_path, encoder): 
	'''
	This function is only used in prediction, so by default, the spectra are not contained. 
	'''
	df = pd.read_csv(csv_path)
	data = []
	for idx, row in df.iterrows(): 
		# mol array
		good_conf, xyz_arr, atom_type = conformation_array(smiles=row['smiles'], 
															conf_type=encoder['conf_type']) 
		if not good_conf:
			print('Can not generate correct conformation: {} {}'.format(row['smiles'], row['id']))
			continue
		if xyz_arr.shape[0] > encoder['max_atom_num']: 
			print('Atomic number ({}) exceed the limitation ({})'.format(encoder['max_atom_num'], xyz_arr.shape[0]))
			continue
		
		rare_atom_flag = False
		rare_atom = ''
		for atom in list(set(atom_type)):
			if atom not in encoder['atom_type'].keys(): 
				rare_atom_flag = True
				rare_atom = atom
				break
		if rare_atom_flag:
			print('Unsupported atom type: {} {}'.format(rare_atom, row['id']))
			continue

		atom_type_one_hot = np.array([encoder['atom_type'][atom] for atom in atom_type])
		assert xyz_arr.shape[0] == atom_type_one_hot.shape[0]

		mol_arr = np.concatenate([xyz_arr, atom_type_one_hot], axis=1)
		mol_arr = np.pad(mol_arr, ((0, encoder['max_atom_num']-xyz_arr.shape[0]), (0, 0)), constant_values=0)

		data.append({'title': row['id'], 'smiles': row['smiles'], 'mol': mol_arr, 'rt': row['rt']})
	return data