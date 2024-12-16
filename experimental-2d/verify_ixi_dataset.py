import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_sample(data, title, is_label=False):
    if is_label:
        # For labels, show each class in different color
        plt.imshow(np.argmax(data, axis=0), cmap='tab10')
    else:
        plt.imshow(data, cmap='gray')
    plt.title(title)
    plt.axis('off')

def verify_ixi_dataset(h5_path="task_data/ixi.h5"):
    with h5py.File(h5_path, 'r') as f:
        # Print dataset info
        print("Dataset Metadata:")
        print(f"Slice range: {f.attrs['slice_range']}")
        print(f"Image size: {f.attrs['image_size']}")
        try:
            # Handle both string and byte encodings
            modalities = [m.decode('ascii') if isinstance(m, bytes) else m 
                         for m in f.attrs['modalities']]
            print(f"Modalities: {modalities}\n")
        except AttributeError:
            print(f"Modalities: {f.attrs['modalities']}\n")
        
        # Create figure for visualization
        fig = plt.figure(figsize=(15, 12))
        
        # Sample from each task group
        for task_idx, task in enumerate(['denoising', 'segmentation', 'classification']):
            print(f"\n{task.upper()} Task:")
            task_group = f[task]
            
            # Get subjects from different sites
            sites = {'GST': [], 'HH': [], 'IOP': []}
            
            for subject in task_group:
                # For denoising task
                if task == 'denoising':
                    for mod in f.attrs['modalities']:
                        mod = mod.decode()
                        if mod in task_group[subject]:
                            site = task_group[subject][mod].attrs['site']
                            sites[site].append((subject, mod))
                
                # For segmentation task
                elif task == 'segmentation':
                    for key in task_group[subject].keys():
                        if key.startswith('image_'):
                            mod = task_group[subject][key].attrs['modality']
                            site = task_group[subject][key].attrs['site']
                            sites[site].append((subject, mod))
                
                # For classification task
                else:
                    for key in task_group[subject].keys():
                        if key.startswith('image_'):
                            mod = task_group[subject][key].attrs['modality']
                            site = task_group[subject][key].attrs['site']
                            age = task_group[subject][key].attrs['age']
                            sex = task_group[subject][key].attrs['sex']
                            sites[site].append((subject, mod, age, sex))
            
            # Print summary for each site
            for site in sites:
                print(f"\n{site}:")
                if sites[site]:
                    sample = sites[site][0]
                    if task == 'classification':
                        subj, mod, age, sex = sample
                        print(f"Sample subject {subj}: {mod}, Age: {age}, Sex: {sex}")
                    else:
                        subj, mod = sample
                        print(f"Sample subject {subj}: {mod}")
                    
                    # Plot sample
                    plt.subplot(3, 3, task_idx*3 + list(sites.keys()).index(site) + 1)
                    
                    if task == 'denoising':
                        data = task_group[subj][mod][:]
                        plot_sample(data[len(data)//2], f"{task}\n{site} - {mod}")

                        plt.axis('off')
                    elif task == 'segmentation':
                        data = task_group[subj][f'image_{mod}'][:]
                        label = task_group[subj][f'label_{mod}'][:]
                        
                        # Show image and label side by side
                        plt.subplot(3, 6, task_idx*6 + list(sites.keys()).index(site)*2 + 1)
                        plot_sample(data[len(data)//2], f"{task}\n{site} - {mod}")
                        
                        plt.subplot(3, 6, task_idx*6 + list(sites.keys()).index(site)*2 + 2)
                        plot_sample(label[len(data)//2], f"Labels\n{site} - {mod}", is_label=True)

                        plt.axis('off')
                    else:  # classification
                        data = task_group[subj][f'image_{mod}'][:]
                        plot_sample(data[len(data)//2], f"{task}\n{site} - {mod}\nAge: {age}, Sex: {sex}")

                        plt.axis('off')
                else:
                    print("No samples found")
        
        plt.tight_layout()
        plt.savefig('task_data/ixi_dataset_verification.png')
        plt.close()

if __name__ == "__main__":
    verify_ixi_dataset()