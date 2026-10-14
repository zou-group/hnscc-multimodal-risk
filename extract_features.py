import os
import timm
import h5py
import time
import numpy as np
from tqdm import tqdm
from openslide import OpenSlide
import pandas as pd
import argparse
from huggingface_hub import login
from uni import get_encoder as get_uni_encoder
import pathologyfoundation as pf

import torch
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset, DataLoader
BICUBIC = InterpolationMode.BICUBIC

from timm.data import resolve_data_config
from timm.layers import SwiGLUPacked
from timm.data.transforms_factory import create_transform

from utils import set_seed
from preprocessing import preprocess_data_pfs

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class Whole_Slide_Bag_FP(Dataset):
	def __init__(self,
		file_path,
		wsi,
		img_transforms=None):
		"""
		Args:
			file_path (string): Path to the .h5 file containing patched data.
			img_transforms (callable, optional): Optional transform to be applied on a sample
		"""
		self.wsi = wsi
		self.roi_transforms = img_transforms

		self.file_path = file_path

		with h5py.File(self.file_path, "r") as f:
			dset = f['coords']
			self.patch_level = f['coords'].attrs['patch_level']
			self.patch_size = f['coords'].attrs['patch_size']
			self.length = len(dset)
			
		self.summary()
			
	def __len__(self):
		return self.length

	def summary(self):
		hdf5_file = h5py.File(self.file_path, "r")
		dset = hdf5_file['coords']
		for name, value in dset.attrs.items():
			print(name, value)

		print('\nfeature extraction settings')
		print('transformations: ', self.roi_transforms)

	def __getitem__(self, idx):
		with h5py.File(self.file_path,'r') as hdf5_file:
			coord = hdf5_file['coords'][idx]
		img = self.wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')

		img = self.roi_transforms(img)
		return {'img': img, 'coord': coord}

class Dataset_All_Bags(Dataset):

	def __init__(self, csv_path):
		self.df = pd.read_csv(csv_path)
	
	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):
		return self.df['slide_id'][idx]

def save_hdf5(output_path, asset_dict, attr_dict= None, mode='a', chunk_size=32):
    with h5py.File(output_path, mode) as file:
        for key, val in asset_dict.items():
            data_shape = val.shape
            if key not in file:
                data_type = val.dtype
                chunk_shape = (chunk_size, ) + data_shape[1:]
                maxshape = (None, ) + data_shape[1:]
                dset = file.create_dataset(key, shape=data_shape, maxshape=maxshape, chunks=chunk_shape, dtype=data_type)
                dset[:] = val
                if attr_dict is not None:
                    if key in attr_dict.keys():
                        for attr_key, attr_val in attr_dict[key].items():
                            dset.attrs[attr_key] = attr_val
            else:
                dset = file[key]
                dset.resize(len(dset) + data_shape[0], axis=0)
                dset[-data_shape[0]:] = val
    return output_path

def load_model(model_name):
    if model_name == 'plip':
        return pf.model_zoo("PLIP-ViT-B-32")

    elif model_name == 'optimus':
        login(token=os.environ.get('HF_TOKEN')) # set HF_TOKEN or log in when prompted

        model = timm.create_model(
            "hf-hub:bioptimus/H-optimus-0", 
            pretrained=True, 
            init_values=1e-5, 
            dynamic_img_size=False
        )
        return model.eval().to(device)
    
    elif model_name == 'virchow':
        login(token=os.environ.get('HF_TOKEN')) # set HF_TOKEN or log in when prompted

        model = timm.create_model(
            "hf-hub:paige-ai/Virchow2",
            pretrained=True,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU
        )
        return model.eval().to(device)
    
    elif model_name == 'uni':
        login(token=os.environ.get('HF_TOKEN')) # set HF_TOKEN or log in when prompted
        model, _ = get_uni_encoder(enc_name='uni2-h', device=device)
        return model.eval()

    else:
        raise ValueError(f"Unknown model {model_name}")

def compute_features(batch, model, model_name):
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        if model_name == 'plip':
            batch = batch.permute(0, 2, 3, 1)
            batch = (batch.cpu().detach().numpy() * 255).astype(np.uint8)
            return model.embed_images(batch, normalize=True).astype(np.float32)
        
        elif model_name == 'virchow':
            # batch needs to be N x 3 x 224 x 224
            output = model(batch) # size: N x 261 x 1280
            class_token = output[:, 0].cpu().numpy() # size: N x 1280
            patch_tokens = output[:, 5:].mean(1).cpu().numpy() # size: N x 1280, tokens 1-4 are register tokens so we ignore those
            return np.concatenate([class_token, patch_tokens], axis=1).astype(np.float32)

        elif model_name == 'optimus' or model_name == 'uni':
            return model(batch).cpu().numpy().astype(np.float32)
        
        else:
            raise ValueError(f"Unsupported model {model_name}")

def process_slide(model_name, model, transform, batch_size, wsi, h5_file_path, output_dir):
    time_start = time.time()

    # Dataset and Loader
    dataset = Whole_Slide_Bag_FP(
        file_path=h5_file_path,
        wsi=wsi,
        img_transforms=transform
    )
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=4 if device.type == "cuda" else 0,
        pin_memory=True if device.type == "cuda" else False
    )

    # Process batches
    output_path = os.path.join(output_dir, 'h5_files', os.path.basename(h5_file_path))
    for batch in tqdm(loader, desc=f'Extracting features using {model_name}'):
        batch_imgs = batch['img'].to(device)
        coords = batch['coord'].numpy().astype(np.int32)
        
        features = compute_features(batch_imgs, model, model_name)
        
        asset_dict = {'features': features, 'coords': coords}
        save_hdf5(output_path, asset_dict, mode='a' if os.path.exists(output_path) else 'w')

    time_elapsed = time.time() - time_start
    print('\ncomputing features for {} took {} s'.format(output_path, time_elapsed))

    with h5py.File(output_path, "r") as file:
        features = file['features'][:]
        print('features size: ', features.shape)
        print('coordinates size: ', file['coords'].shape)

    features = torch.from_numpy(features)
    torch.save(features, os.path.join(output_dir, 'pt_files', os.path.basename(h5_file_path).split('.')[0]+'.pt'))

def load_pfs_data(data_path, image_path):
    folder_names = {name for name in os.listdir(image_path) if os.path.isdir(os.path.join(image_path, name))}

    matching_folders = set()
    for folder in folder_names:
        folder_path = os.path.join(image_path, folder)
        if any(filename.endswith(('_HE.tiff', '_HE.svs')) for filename in os.listdir(folder_path)):
            matching_folders.add(folder)

    data, clinical_attrs, label_cols, all_genes, unique_patients = preprocess_data_pfs(data_path,  mode='attr', treatment_type = 'agnostic')
    
    subset_data = data[data['patient_id'].isin(matching_folders)]
    patient_ids = subset_data['patient_id'].to_list()
    
    return subset_data, patient_ids

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['plip', 'virchow', 'uni', 'optimus'])
    args = parser.parse_args()

    set_seed()

    # Initialize paths
    data_path = '' # TODO: add path to data folder here
    image_path = os.path.join(data_path, 'Images') # TODO: update to your images folder (one subfolder per patient with H&E slides)
    
    subset_data, patient_ids = load_pfs_data(data_path, image_path)
    
    # Process each slide
    csv_path = os.path.join(image_path, 'CLAM_HE_images/process_list_autogen_no_ext.csv') # TODO: update to your CLAM slide list csv
    bags_dataset = Dataset_All_Bags(csv_path)
    feat_dir = os.path.join(image_path, f'HE_features') # TODO: update to where you want extracted features saved
    os.makedirs(feat_dir, exist_ok=True)

    model_name = args.model
    model = load_model(model_name)

    # Create output directories
    output_dir = os.path.join(feat_dir, model_name)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'pt_files'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'h5_files'), exist_ok=True)

    # Initialize transforms
    if model_name == 'virchow':
        batch_size = 32
        transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    
    elif model_name == 'uni':
        batch_size = 64
        _, transform = get_uni_encoder(enc_name='uni2-h', device=device)
   
    elif model_name == 'optimus':
        batch_size = 32
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.707223, 0.578729, 0.703617), 
                std=(0.211883, 0.230117, 0.177517)
            ),
        ])

    elif model_name == 'plip':
        batch_size = 256
        def _convert_image_to_rgb(image):
            return image.convert("RGB")

        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5), 
            transforms.RandomRotation(degrees=(-90,90)), 
            # CenterZoom(scale_range=(1.0, 4.0)), 
            transforms.Resize(224, interpolation=BICUBIC),
            transforms.CenterCrop(224),
            _convert_image_to_rgb,
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])

    else:
        raise ValueError(f"Unsupported model {model_name}")

    if os.path.exists(os.path.join(feat_dir, f'{model_name}/pt_files')):
        dest_files = os.listdir(os.path.join(feat_dir, f'{model_name}/pt_files'))
    else:
        dest_files = []
    
    for slide_id in tqdm(bags_dataset, desc='Processing slides'):
        if slide_id.split('_')[0] not in patient_ids:
            print('\nnot in patient IDs: {}'.format(slide_id))
            continue

        if slide_id+'.pt' in dest_files:
            print('\nskipped {}'.format(slide_id))
            continue
            
        h5_file_path = os.path.join(image_path, f'CLAM_HE_images/patches/{slide_id}.h5') # TODO: update to your CLAM patch coordinates folder
        if not os.path.exists(h5_file_path):
            continue

        slide_path = next(
            os.path.join(image_path, f'HE_images/{slide_id}{ext}') # TODO: update to your H&E slides folder
            for ext in ['.svs', '.tiff']
            if os.path.exists(os.path.join(image_path, f'HE_images/{slide_id}{ext}'))
        )
        
        wsi = OpenSlide(slide_path)
        process_slide(model_name, model, transform, batch_size, wsi, h5_file_path, output_dir)
