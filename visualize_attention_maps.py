import os
import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
from openslide import OpenSlide
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
import matplotlib.cm as cm
from scipy.stats import percentileofscore
import pandas as pd
import glob

from utils import set_seed
from preprocessing import preprocess_data_pfs

def visualize_attention_heatmap_for_batch(slide_id, all_data, risk_data_dict):
    """
    Create an attention heatmap overlaid on WSI (heatmap only version)
    
    Args:
        slide_id: ID of the slide to process
        all_data: Preprocessed data containing patient information
        risk_data_dict: Dictionary containing risk score data from CSV files
    """
    # Setup paths and load data
    feature = 'plip'
    ext = 'imaging+molecular'
    data_path = '' # TODO: add path to data folder here
    
    # get patient's CPS score
    curr_pid = slide_id.split('_')[0]
    curr_cps = all_data.loc[all_data['patient_id'] == curr_pid, 'pdl1_cps'].values
    
    if len(curr_cps) == 0 or pd.isna(curr_cps[0]):
        curr_cps = 'None'
    else:
        curr_cps = str(curr_cps[0])

    if os.path.exists(os.path.join(data_path, 'Images/processed_HE_images', f'{slide_id}_HE.svs')): # TODO: update to your H&E slides folder (also in lines below)
        slide_path = os.path.join(data_path, 'Images/processed_HE_images', f'{slide_id}_HE.svs')
    elif os.path.exists(os.path.join(data_path, 'Images/processed_HE_images', f'{slide_id}_HE.tiff')):
        slide_path = os.path.join(data_path, 'Images/processed_HE_images', f'{slide_id}_HE.tiff')
    else:
        print(f'Slide path does not exist for {slide_id}...')
        return False
    
    h5_path_coords = os.path.join(data_path, f'Images/CLAM_HE_images/patches/{slide_id}_HE.h5') # TODO: update to your CLAM patch coordinates folder
    attention_dir = f'./outputs/avg_attention/{ext}/{feature}'
    pt_path = os.path.join(attention_dir, f'{slide_id}_HE.pt')
    
    # Check if required files exist
    if not os.path.exists(h5_path_coords):
        print(f'H5 coordinates file not found for {slide_id}')
        return False
    
    if not os.path.exists(pt_path):
        print(f'Attention file not found for {slide_id}')
        return False

    # Get treatment type and normalized risk score from CSV data
    treatment_type = 'Unknown'
    normalized_risk = 'N/A'
    risk_category = 'Unknown'
    
    slide_id_with_he = slide_id + '_HE'
    if slide_id_with_he in risk_data_dict:
        risk_info = risk_data_dict[slide_id_with_he]
        treatment_type = risk_info['treatment'].upper()  # Convert to uppercase for display
        normalized_risk = f"{risk_info['normalized_risk_score']:.2f}"
        risk_category = risk_info['risk_category']
    
    try:
        # Load WSI and patch coordinates
        wsi = OpenSlide(slide_path)
        with h5py.File(h5_path_coords, 'r') as f:
            coords = f['coords'][:]
            patch_size = f['coords'].attrs['patch_size']
            patch_level = f['coords'].attrs['patch_level']
        
        # Load averaged attention weights
        attention_data = torch.load(pt_path)
        attention_weights = attention_data['avg_attention'].numpy()
        print(f'Processing {slide_id}: {attention_weights.shape[0]} patches')
        
        # Parameters
        vis_level = wsi.get_best_level_for_downsample(32)
        print(f"Using visualization level {vis_level} for heatmap")
        
        # Get the dimensions at visualization level
        w, h = wsi.level_dimensions[vis_level]
        
        # Calculate downsampling factor
        level_downsample = wsi.level_downsamples[patch_level]
        vis_downsample = wsi.level_downsamples[vis_level]
        downsample_factor = vis_downsample / level_downsample
        
        # Calculate the patch size at the visualization level
        vis_patch_size = int(patch_size / downsample_factor)
        print(f"Patch size at level {patch_level}: {patch_size}, at level {vis_level}: {vis_patch_size}")
        
        # Create the heatmap canvas
        heatmap = np.zeros((h, w), dtype=np.float32)
        
        # Convert attention scores to percentiles
        flat_scores = attention_weights.flatten()
        percentiles = np.zeros_like(attention_weights)
        for i in range(len(attention_weights)):
            percentiles[i] = percentileofscore(flat_scores, attention_weights[i])
        percentiles = percentiles / 100.0  # Normalize to 0-1
        
        print("Placing attention scores on heatmap...")
        for idx, ((x, y), score) in enumerate(tqdm(zip(coords, percentiles), total=len(coords), desc="Processing patches")):
            # Convert coordinates to visualization level
            x_vis = int(x / level_downsample)
            y_vis = int(y / level_downsample)
            
            # Scale coordinates to visualization level
            x_vis = int(x_vis / downsample_factor)
            y_vis = int(y_vis / downsample_factor)
            
            # Ensure coordinates are within bounds
            if x_vis < 0 or y_vis < 0 or x_vis >= w or y_vis >= h:
                continue
            
            # Calculate patch bounds
            x_end = min(x_vis + vis_patch_size, w)
            y_end = min(y_vis + vis_patch_size, h)
            
            # Place the score on the heatmap
            heatmap[y_vis:y_end, x_vis:x_end] = score
        
        # Apply Gaussian blur
        print("Applying Gaussian blur...")
        sigma = 5  # Adjust as needed
        heatmap = gaussian_filter(heatmap, sigma=sigma)
        
        # Normalize the heatmap to 0-1
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        # Read the WSI at the visualization level directly for best quality
        wsi_img = wsi.read_region((0,0), vis_level, (w, h)).convert('RGB')
        
        # Apply colormap to heatmap
        cmap = plt.colormaps['bwr']
        colored_heatmap = (cmap(heatmap) * 255).astype(np.uint8)
        
        # Create the overlay
        wsi_img_array = np.array(wsi_img)
        colored_heatmap_rgb = colored_heatmap[:, :, :3]
        
        # Create mask where heatmap has non-zero values
        mask = heatmap > 0
        alpha = 0.6  # Adjust transparency
        
        # Create overlay by blending
        result = wsi_img_array.copy()
        result[mask] = (1 - alpha) * wsi_img_array[mask] + alpha * colored_heatmap_rgb[mask]
        
        # Output directory
        output_dir = './outputs/attention_heatmaps'
        os.makedirs(output_dir, exist_ok=True)
        
        # Create single heatmap visualization
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(result)
        
        # Add title
        title = f'{slide_id}\nCPS: {curr_cps} | Treatment: {treatment_type} | {risk_category} Risk'
        ax.set_title(title, fontsize=16, pad=20)
        ax.axis('off')
        
        # Add colorbar
        cbar = plt.colorbar(cm.ScalarMappable(cmap=cmap), ax=ax, shrink=0.7, pad=0.05, label='Attention Weight')
        
        plt.tight_layout()
        
        # Save the figure
        save_path = os.path.join(output_dir, f'{slide_id}_attention_heatmap_only.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f'Saved heatmap visualization to {save_path}')
        return True
        
    except Exception as e:
        print(f'Error processing {slide_id}: {str(e)}')
        return False

def load_risk_data():
    """
    Load risk score data from all CSV files (both high and low risk)
    
    Returns:
        dict: Dictionary mapping slide_id (with _HE suffix) to risk information
    """
    csv_files = [
        # High risk files
        './outputs/imaging+molecular/plip/high_risk_patients_combo_test.csv',
        './outputs/imaging+molecular/plip/high_risk_patients_combo_train&val.csv',
        './outputs/imaging+molecular/plip/high_risk_patients_io_test.csv',
        './outputs/imaging+molecular/plip/high_risk_patients_io_train&val.csv',
        # Low risk files
        './outputs/imaging+molecular/plip/low_risk_patients_combo_test.csv',
        './outputs/imaging+molecular/plip/low_risk_patients_combo_train&val.csv',
        './outputs/imaging+molecular/plip/low_risk_patients_io_test.csv',
        './outputs/imaging+molecular/plip/low_risk_patients_io_train&val.csv'
    ]
    
    risk_data_dict = {}
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            
            # Extract risk category from filename
            filename = os.path.basename(csv_file)
            if 'high_risk' in filename:
                risk_category = 'High'
            elif 'low_risk' in filename:
                risk_category = 'Low'
            else:
                risk_category = 'Unknown'
            
            print(f'Loaded {len(df)} {risk_category.lower()} risk records from {filename}')
            
            for _, row in df.iterrows():
                slide_id = row['slide_id']
                risk_data_dict[slide_id] = {
                    'treatment': row['treatment'],
                    'normalized_risk_score': row['normalized_risk_score'],
                    'patient_id': row['patient_id'],
                    'avg_risk_score': row['avg_risk_score'],
                    'duration_months': row['duration_months'],
                    'risk_category': risk_category
                }
        except Exception as e:
            print(f'Error loading {csv_file}: {str(e)}')
    
    print(f'Total slides with risk data: {len(risk_data_dict)}')
    
    # Print summary of risk categories
    high_count = sum(1 for v in risk_data_dict.values() if v['risk_category'] == 'High')
    low_count = sum(1 for v in risk_data_dict.values() if v['risk_category'] == 'Low')
    print(f'High risk patients: {high_count}')
    print(f'Low risk patients: {low_count}')
    
    return risk_data_dict

def batch_process_attention_heatmaps():
    """
    Process all .tiff/.svs HE images in the specified directory
    """
    data_path = '' # TODO: add path to data folder here
    images_path = os.path.join(data_path, 'Images/processed_HE_images') # TODO: update to your H&E slides folder

    # Load risk score data from CSV files
    print('Loading risk score data...')
    risk_data_dict = load_risk_data()
    
    # Find all .tiff and .svs files
    tiff_files = glob.glob(os.path.join(images_path, '*.tiff'))
    svs_files = glob.glob(os.path.join(images_path, '*.svs'))
    
    all_files = tiff_files + svs_files
    print(f'Found {len(all_files)} HE images to process')
    
    # Preprocess data once to get patient information
    print('Preprocessing data...')
    all_data = []
    for treatment in ['combo', 'io']:
        data, clinical_attrs, label_cols, all_genes, unique_patients = preprocess_data_pfs(
            data_path, mode='attr', treatment_type=treatment
        )
        all_data.extend([data])
    
    all_data = pd.concat(all_data, ignore_index=True)
    print(f'Loaded data for {len(all_data)} patients')
    
    # Process each file
    successful_count = 0
    failed_count = 0
    
    for file_path in tqdm(all_files, desc="Processing slides"):
        # Extract slide_id from filename
        filename = os.path.basename(file_path)
        slide_id = filename.replace('_HE.tiff', '').replace('_HE.svs', '')
        
        print(f'\nProcessing slide: {slide_id}')
        
        # Process the slide
        success = visualize_attention_heatmap_for_batch(slide_id, all_data, risk_data_dict)
        
        if success:
            successful_count += 1
        else:
            failed_count += 1
    
    print(f'\nBatch processing complete!')
    print(f'Successfully processed: {successful_count} slides')
    print(f'Failed to process: {failed_count} slides')

if __name__ == "__main__":
    set_seed()

    batch_process_attention_heatmaps()
