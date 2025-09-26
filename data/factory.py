from torch.utils.data import DataLoader
from typing import Tuple, List
from .dataset import EEMFNetDataset
from torch.utils.data import DataLoader, Subset
import random



# def blancedValidloader(validloader):


#     # Iterate through the DataLoader
#     for idx, (inputs, masks, targets,_) in enumerate(validloader):
#         if idx in selected_indices:
#             inputs, masks, targets = inputs.to(device), masks.to(device), targets.to(device)
#             validation_subloader.append((inputs, masks, targets))
    
#     return validation_subloader


def create_dataset(
    datadir: str, target: str, is_train: bool, to_memory: bool = False,
    file_list = None,
    resize: Tuple[int, int] = (224,224),
    texture_source_dir: str = None, structure_grid_size: str = 8,
    transparency_range: List[float] = [0.15, 1.],
    perlin_scale: int = 6, min_perlin_scale: int = 0, perlin_noise_threshold: float = 0.5,
    # use_mask: bool = True, bg_threshold: float = 100, bg_reverse: bool = False
):

    dataset = EEMFNetDataset(
        file_list =             file_list,
        datadir                = datadir,
        target                 = target, 
        is_train               = is_train,
        to_memory              = to_memory,
        resize                 = resize,
        texture_source_dir     = texture_source_dir, 
        structure_grid_size    = structure_grid_size,
        transparency_range     = transparency_range,
        perlin_scale           = perlin_scale, 
        min_perlin_scale       = min_perlin_scale, 
        perlin_noise_threshold = perlin_noise_threshold,
        # use_mask               = use_mask,
        # bg_threshold           = bg_threshold,
        # bg_reverse             = bg_reverse
    )
    if 'test' in file_list[0]:
        
        labels = [0 if 'good' in file_name else 1 for file_name in file_list]  
        
        # Separate indices for normal and abnormal images
        normal_indices = [i for i, label in enumerate(labels) if label == 0]
        abnormal_indices = [i for i, label in enumerate(labels) if label == 1]

        # Count the number of normal and abnormal images
        num_normal = len(normal_indices)
        num_abnormal = len(abnormal_indices)
        # print("...............len file_list test", len(file_list))

        # validation_subloader = blancedValidloader(validloader)
        # Choose the desired ratio (2:1 or 1:2)
        random.seed(42)  # Ensure reproducibility
        if num_normal > 2*num_abnormal:
            selected_normal = num_abnormal * 2
            selected_abnormal = num_abnormal
            normal_indices = random.sample(normal_indices, min(selected_normal, num_normal))
            selected_indices = normal_indices + abnormal_indices
            
        elif num_abnormal > 2*num_normal:
            selected_normal = num_normal
            selected_abnormal = num_normal * 2
            abnormal_indices = random.sample(abnormal_indices, min(selected_abnormal, num_abnormal))
            selected_indices = normal_indices + abnormal_indices
        
        else:
            selected_indices = normal_indices + abnormal_indices
            

        # shuffle the selected indices 
        random.shuffle(selected_indices)

        # Create a subset of the dataset
        sub_dataset = Subset(dataset, selected_indices)
           
        # print("...............len validation_subloader", len(sub_dataset))
    else:
        sub_dataset = dataset


    return sub_dataset, dataset


def create_dataloader(dataset, train: bool, batch_size: int = 16, num_workers: int = 1):
    
    
    dataloader = DataLoader(
        dataset,
        shuffle     = train,
        batch_size  = batch_size,
        num_workers = num_workers
    )

    return dataloader