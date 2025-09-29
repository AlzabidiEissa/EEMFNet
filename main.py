

import wandb
import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from timm import create_model
from data import create_dataset, create_dataloader
from models import EEMFNet #, MemoryBank
from focal_loss import FocalLoss
from train import training
from log import setup_default_logging
from utils import torch_seed
from scheduler import CosineAnnealingWarmupRestarts
import datetime
from glob import glob

_logger = logging.getLogger('train')

from torchvision import transforms

from torchvision.models import resnet18
from sklearn.decomposition import PCA
import numpy as np
from PIL import Image
import torch.optim as optim
from timm.optim import Lookahead
import itertools


def outlier_removal(data_path, feature_extractor_name):
    images_list = os.listdir(data_path)

    # Load pre-trained model for feature extraction
    model = create_model(feature_extractor_name,pretrained = True)
    model.fc = torch.nn.Identity()  # Removethe fully connected layer layer
    model.eval()

    # Preprocessing transformations (resize to model input size)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Load and preprocess images
    images = []
    image_names = []
    for img_name in images_list:
        img = Image.open(os.path.join(data_path, img_name)).convert("RGB")
        preprocessed_img = preprocess(img)
        images.append(preprocessed_img)
        image_names.append(img_name)

    images = torch.stack(images)  # Stack images into a batch

    # Extract features
    with torch.no_grad():
        features = model(images)

    # PCA for dimensionality reduction
    pca = PCA(n_components=2)
    features_reduced = pca.fit_transform(features.numpy())

    # Compute distances
    mean_feature = np.mean(features_reduced, axis=0)
    distances = np.linalg.norm(features_reduced - mean_feature, axis=1)
    threshold = np.percentile(distances, 95)  # 95th percentile for outlier detection

    # Filter outliers
    non_outliers = np.where(distances <= threshold)[0]
    # removed_images = [image_names[i] for i in outliers]
    selected_images = [image_names[i] for i in non_outliers]
    
    return selected_images

def run(cfg):
    
    # setting seed and device
    # setup_default_logging()
    torch_seed(cfg.SEED)
    
    if cfg.TRAIN.use_tpu:
        import torch_xla
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    _logger.info('Device: {}'.format(device))


    # Experiment Configuration
    
    # targets_list = ['SL10', 'SL13', 'SL16', 'SP3', 'SP5', 'SP19', 'SP24', 'CL2']
    targets_list = ['carpet']
    # targets_list = ['1', 'carpet', 'wood', 'tile', 'leather', 'grid', 'SL1', 'CL1', '0', '3', '2']
    # backbones_list = ["densenet121","resnet50","wide_resnet50_2","resnet101","resnet34","resnet18",
    #                    "efficientnet_b4","mobilenetv3_small_100","efficientnet_b3", "hrnet_w32"]

    # optimizers_list = {
    # "AdamW": optim.AdamW, 
    # "Adam": optim.Adam, 
    # # "SGD": optim.SGD, 
    # # "RMSprop": optim.RMSprop,
    # # "Adagrad": optim.Adagrad,
    # "LAMB": optim.Adam,  # LAMB typically used with transformers
    # # "LookAhead": optim.Adam,  # LookAhead with Adam
    # }
    
    # augmentations = [True, False]
    # memory_bank_options = [True, False]
    # attention_options = [True, False]
    # outlier_removal_options = [True, False] # [True, False]
    o_removal = False
    feature_extractor_name = "resnet18"
    optimizer_name = "AdamW"
    optimizers_list = {"AdamW": optim.AdamW}
    # target = 'carpet'
    # Run Experiments
    # results_summary = []

    for target in targets_list:
    # for target, feature_extractor_name, optimizer_name, augment, use_memory_bank, use_attention, o_removal in itertools.product(
        #         targets_list,
        #         backbones_list,
        #         optimizers_list,
        #         augmentations,
        #         memory_bank_options,
        #         attention_options,
        #         outlier_removal_options):

        # Run Experiments
        # results_summary = []

        # for target, optimizer_name in itertools.product(
        #         targets_list,
        #         # backbones_list,
        #         optimizers_list,
        #         # outlier_removal_options
        #         ):

        # savedir
        flolder_name = cfg.EXP_NAME + f"-{target}-WO_Mem-{feature_extractor_name[:3]}_{feature_extractor_name[-2:]}-{optimizer_name}-OR_{o_removal}-" + datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S")

        # if feature_extractor_name=='resnet18':
        #     cfg.EXP_NAME = cfg.EXP_NAME + f"-{target}-R18-" + datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S")
        # elif feature_extractor_name=='resnet50':
        #     cfg.EXP_NAME = cfg.EXP_NAME + f"-{target}-R50-" + datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S")
        # elif feature_extractor_name=='wide_resnet50_2':
        #     cfg.EXP_NAME = cfg.EXP_NAME + f"-{target}-WR50-" + datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S")
        # elif feature_extractor_name=='resnet101':
        #     cfg.EXP_NAME = cfg.EXP_NAME + f"-{target}-R101-" + datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S")
        # elif feature_extractor_name=='resnet34':
        #     cfg.EXP_NAME = cfg.EXP_NAME + f"-{target}-R34-" + datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S")
        # elif feature_extractor_name=='efficientnet_b4':
        #     cfg.EXP_NAME = cfg.EXP_NAME + f"-{target}-Eb4-" + datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S")
        # elif feature_extractor_name=='efficientnet_b3':
        #     cfg.EXP_NAME = cfg.EXP_NAME + f"-{target}-Eb3-" + datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S")                            
        # elif feature_extractor_name=='hrnet_w32':
        #     cfg.EXP_NAME = cfg.EXP_NAME + f"-{target}-Hr32-" + datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S")                            
        # elif feature_extractor_name=='mobilenetv3_small':
        #     cfg.EXP_NAME = cfg.EXP_NAME + f"-{target}-Mob3-" + datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S")                            
        # elif feature_extractor_name=='convnext_base':
        #     cfg.EXP_NAME = cfg.EXP_NAME + f"-{target}-CB-" + datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S")                            
        # elif feature_extractor_name=='densenet121':
        #     cfg.EXP_NAME = cfg.EXP_NAME + f"-{target}-Dense121-" + datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S")                            
        # elif feature_extractor_name=='swin_base_patch4_window7_224':
        #     cfg.EXP_NAME = cfg.EXP_NAME + f"-{target}-Swin224-" + datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S")                            
                
        print ('.................................................................')        
        print ('save results in folder name: ', flolder_name)        
        print ('target: ', target)        
        print ('memory_bank: False')        
        print ('backbone: ', feature_extractor_name)        
        print ('optimizer name: ', optimizer_name)        
        print ('outlier removal ', o_removal)        
        print ('.................................................................')        
        print ()        

        savedir = os.path.join(cfg.RESULT.savedir, flolder_name)
        os.makedirs(savedir, exist_ok=True)
        
        if o_removal:
            data_path = os.path.join(cfg.DATASET.datadir, target, 'train/good')
            selected_images = outlier_removal(data_path, feature_extractor_name)
            # Generate full paths
            file_list_train = [os.path.join(cfg.DATASET.datadir, target, 'train/good', img) for img in selected_images]

        else:
            # file_list_train = glob(os.path.join(cfg.DATASET.datadir, target, r'train\*\*')) 
            file_list_train = glob(os.path.join(cfg.DATASET.datadir, target, 'train/*/*')) 
        
        # file_list_test = glob(os.path.join(cfg.DATASET.datadir, target, r'test\*\*'))
        file_list_test = glob(os.path.join(cfg.DATASET.datadir, target, 'test/*/*'))

        # wandb
        if cfg.TRAIN.use_wandb:
            wandb.init(name=flolder_name, 
                        project='EEMF-Net-WoMem', 
                        config=OmegaConf.to_container(cfg),
                        reinit=True) # Allow reinitialization for multiple runs

        # build datasets
        trainset,_ = create_dataset(
            file_list              = file_list_train,
            datadir                = cfg.DATASET.datadir,
            target                 = target, 
            is_train               = True,
            resize                 = cfg.DATASET.resize,
            texture_source_dir     = cfg.DATASET.texture_source_dir,
            structure_grid_size    = cfg.DATASET.structure_grid_size,
            transparency_range     = cfg.DATASET.transparency_range,
            perlin_scale           = cfg.DATASET.perlin_scale,
            min_perlin_scale       = cfg.DATASET.min_perlin_scale,
            perlin_noise_threshold = cfg.DATASET.perlin_noise_threshold,

        )

        testset,_ = create_dataset(
            file_list = file_list_test,
            datadir  = cfg.DATASET.datadir,
            target   = target, 
            is_train = False,
            resize   = cfg.DATASET.resize
        )
        # build dataloader
        
        trainloader = create_dataloader(
            dataset     = trainset,
            train       = True,
            batch_size  = cfg.DATALOADER.batch_size,
            num_workers = cfg.DATALOADER.num_workers
        )
        
        testloader = create_dataloader(
            dataset     = testset,
            train       = False,
            batch_size  = cfg.DATALOADER.batch_size,
            num_workers = cfg.DATALOADER.num_workers
        )

        # build feature extractor
        feature_extractor = create_model(
            feature_extractor_name, 
            pretrained    = True, 
            features_only = True
        ).to(device)

        ## freeze weight of layer1,2,3
        for p in feature_extractor.parameters():
            p.requires_grad = False

        if feature_extractor_name in ['efficientnet_b3', 'efficientnet_b4']:
            for p in feature_extractor.blocks[6].parameters():
                p.requires_grad = True
        
        elif feature_extractor_name in ['mobilenetv3_small_100']:
            for name, param in feature_extractor.named_parameters():
                if 'blocks.4' in name:  # Adjust to the target block name
                    param.requires_grad = True
                if 'blocks.5' in name:  
                    param.requires_grad = True       

        elif  feature_extractor_name in ['hrnet_w32']:
            for name, param in feature_extractor.named_parameters():
                if 'incre_modules' in name:  # Adjust the name based on model inspection
                    param.requires_grad = True

        elif feature_extractor_name in ['densenet121']:
            for name, param in feature_extractor.named_parameters():
                if 'denseblock4' in name:  # Adjust to the target block name
                    param.requires_grad = True
                if 'features_norm5' in name:  
                    param.requires_grad = True          
        
        else:
            for p in feature_extractor.layer4.parameters():
                p.requires_grad = True

        # build EEMFNet
        model = EEMFNet(
            feature_extractor = feature_extractor,
            device= device
        ).to(device)

        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params}")
        param_size_in_bytes = total_params * 4  # 4 bytes per parameter for float32
        param_size_in_MB = param_size_in_bytes / (1024 ** 2)  # Convert bytes to MB
        print(f"Model size in MB: {param_size_in_MB:.2f} MB")

        model_size=f"Model size: {param_size_in_MB:.2f} MB"

        # Set training
        l1_criterion = nn.L1Loss()

        f_criterion = FocalLoss(
            smooth= cfg.TRAIN.focal_smooth,
            gamma = cfg.TRAIN.focal_gamma, 
            alpha = cfg.TRAIN.focal_alpha
        )

        optimizer = optimizers_list[optimizer_name](
            params       = filter(lambda p: p.requires_grad, model.parameters()), 
            lr           = cfg.OPTIMIZER.lr, 
            weight_decay = cfg.OPTIMIZER.weight_decay)
        
        # Learning Rate Scheduler
        if cfg['SCHEDULER']['use_scheduler']:
            scheduler = CosineAnnealingWarmupRestarts(
                optimizer, 
                first_cycle_steps = cfg.TRAIN.num_training_steps,
                max_lr = cfg.OPTIMIZER.lr,
                min_lr = cfg.SCHEDULER.min_lr,
                gamma= 1.0,
                warmup_steps   = int(cfg.TRAIN.num_training_steps * cfg.SCHEDULER.warmup_ratio))
        
        else:
            scheduler = None

        # Fitting model
        # results = training(
        training(
            model              = model, 
            num_training_steps = cfg.TRAIN.num_training_steps, 
            trainloader        = trainloader, 
            validloader        = testloader, 
            criterion          = [l1_criterion, f_criterion], 
            loss_weights       = [cfg.TRAIN.l1_weight, cfg.TRAIN.focal_weight],
            optimizer          = optimizer,
            scheduler          = scheduler,
            log_interval       = cfg.LOG.log_interval,
            eval_interval      = cfg.LOG.eval_interval,
            savedir            = savedir,
            device             = device,
            use_wandb          = cfg.TRAIN.use_wandb,
            use_tpu            = cfg.TRAIN.use_tpu,
            model_size         = model_size)


        # results_summary.append({
        # "dataset": target,
        # "backbone": feature_extractor_name,
        # "optimizer": optimizer_name,
        # # "augment": augment,
        # # "memory_bank": use_memory_bank,
        # # "attention": use_attention,
        # "outlier_removal": outlier_removal,
        # "results": results
        #  })
        
        # Finalize the current run
        wandb.finish()

if __name__=='__main__':
    args = OmegaConf.from_cli()
    # load default config
    cfg = OmegaConf.load(args.configs)
    del args['configs']
    
    # merge config with new keys
    cfg = OmegaConf.merge(cfg, args)
    
    # target cfg
    if type(cfg.DATASET.target) == int:
        # cfg.DATASET.target = '0'+str(cfg.DATASET.target)
        cfg.DATASET.target = str(cfg.DATASET.target)    
    print(OmegaConf.to_yaml(cfg))

    run(cfg)
