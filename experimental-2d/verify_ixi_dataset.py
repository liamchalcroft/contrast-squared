import matplotlib.pyplot as plt
import torch
from downstream_preprocess import get_test_loader

def plot_sample(data, title, is_label=False):
    if is_label:
        # For labels, show each class in different color
        plt.imshow(torch.argmax(data, axis=0), cmap='tab10')
    else:
        plt.imshow(data[0], cmap='gray')
    plt.title(title)
    plt.axis('off')

def verify_dataloaders(task='denoising', modalities=['t1', 't2', 'pd'], sites=['GST', 'HH', 'IOP']):
    # Create figure for visualization
    fig = plt.figure(figsize=(30, 15)) if task == 'segmentation' else plt.figure(figsize=(15, 15))
    
    for modality_idx, modality in enumerate(modalities):
        for site_idx, site in enumerate(sites):
            # Get test loader for each modality and site
            test_loader = get_test_loader(batch_size=1, task=task, modality=modality, site=site)
            # Get the first batch
            batch = next(iter(test_loader))
            image = batch['image'][0]
            print(image.shape)
            
            if task == 'segmentation':
                # Plot the image
                plt.subplot(len(modalities), len(sites) * 2, modality_idx * len(sites) * 2 + site_idx * 2 + 1)
                plot_sample(image, f"{task}\n{site} - {modality}")
                label = batch['label'][0]
                print(label.shape)
                plt.subplot(len(modalities), len(sites) * 2, modality_idx * len(sites) * 2 + site_idx * 2 + 2)
                plot_sample(label, f"Labels\n{site} - {modality}", is_label=True)
            elif task == 'classification':
                age = batch['age'][0]
                sex = 'M' if batch['sex'][0] == 1 else 'F'
                title = f"{task}\n{site} - {modality}\nAge: {age:.1f}, Sex: {sex}"
                plt.subplot(len(modalities), len(sites), modality_idx * len(sites) + site_idx + 1)
                plot_sample(image, title)
            else:
                # Plot the image
                plt.subplot(len(modalities), len(sites), modality_idx * len(sites) + site_idx + 1)
                plot_sample(image, f"{task}\n{site} - {modality}")
    
    plt.tight_layout()
    plt.savefig(f'task_data/ixi_dataset_verification_{task}.png')
    plt.close()

if __name__ == "__main__":
    for task in [
        # 'denoising', 
        # 'segmentation', 
        'classification'
    ]:
        verify_dataloaders(task=task)