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
        print(f"Number of center slices: {f.attrs['num_slices']}")
        print(f"Image size: {f.attrs['image_size']}")
        try:
            modalities = [m.decode('ascii') if isinstance(m, bytes) else m 
                         for m in f.attrs['modalities']]
            print(f"Modalities: {modalities}\n")
        except AttributeError:
            print(f"Modalities: {f.attrs['modalities']}\n")
        
        # Create figure for visualization
        fig = plt.figure(figsize=(15, 15))
        
        # Define which site to show for each task and modality
        site_order = ['GST', 'HH', 'IOP']
        
        # Sample from each task group
        for task_idx, task in enumerate(['denoising', 'segmentation', 'classification']):
            print(f"\n{task.upper()} Task:")
            task_group = f[task]
            
            # For each site position, find a subject
            for site_idx, site in enumerate(site_order):
                found = False
                # Find first subject from this site
                for subject in task_group:
                    if task == 'denoising':
                        for mod in modalities:
                            if mod in task_group[subject]:
                                if task_group[subject][mod].attrs['site'] == site:
                                    data = task_group[subject][mod][:]
                                    middle_slice = data[len(data)//2]
                                    
                                    plt.subplot(3, 3, task_idx*3 + site_idx + 1)
                                    plot_sample(middle_slice, f"{task}\n{site} - {mod}")
                                    print(f"{site}: Sample subject {subject}: {mod}")
                                    found = True
                                    break
                    else:  # segmentation and classification
                        for key in task_group[subject].keys():
                            if key.startswith('image_'):
                                if task_group[subject][key].attrs['site'] == site:
                                    data = task_group[subject][key][:]
                                    mod = task_group[subject][key].attrs['modality']
                                    middle_slice = data[len(data)//2]
                                    
                                    if task == 'segmentation':
                                        # Show image and label side by side
                                        plt.subplot(3, 6, task_idx*6 + site_idx*2 + 1)
                                        plot_sample(middle_slice, f"{task}\n{site} - {mod}")
                                        
                                        plt.subplot(3, 6, task_idx*6 + site_idx*2 + 2)
                                        label = task_group[subject][f'label_{mod}'][:]
                                        plot_sample(label[len(label)//2], f"Labels\n{site} - {mod}", is_label=True)
                                        print(f"{site}: Sample subject {subject}: {mod}")
                                    else:  # classification
                                        plt.subplot(3, 3, task_idx*3 + site_idx + 1)
                                        age = task_group[subject][key].attrs['age']
                                        sex = task_group[subject][key].attrs['sex']
                                        plot_sample(middle_slice, f"{task}\n{site} - {mod}\nAge: {age}, Sex: {sex}")
                                        print(f"{site}: Sample subject {subject}: {mod}, Age: {age}, Sex: {sex}")
                                    found = True
                                    break
                    if found:
                        break
                
                if not found:
                    plt.subplot(3, 3, task_idx*3 + site_idx + 1)
                    plt.text(0.5, 0.5, f"No data\n{site}", ha='center', va='center')
                    plt.axis('off')
                    print(f"{site}: No samples found")
        
        plt.tight_layout()
        plt.savefig('task_data/ixi_dataset_verification.png')
        plt.close()

if __name__ == "__main__":
    verify_ixi_dataset()