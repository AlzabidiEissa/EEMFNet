import cv2
import os
import numpy as np
from glob import glob
from einops import rearrange

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
# import imgaug.augmenters as iaa
from math import *
# from data import rand_perlin_2d_np
from typing import List, Tuple
import random
from .perlin import rand_perlin_2d_np
import globals
from PIL import Image
import albumentations as A

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class EEMFNetDataset(Dataset):
    def __init__(
        self, datadir: str, target: str, is_train: bool, to_memory: bool = False, 
        resize: Tuple[int, int] = (224,224), 
        file_list = None,
        texture_source_dir: str = None, structure_grid_size: str = 8,
        transparency_range: List[np.float32] = [0.15, 1.],
        perlin_scale: int = 6, min_perlin_scale: int = 0, perlin_noise_threshold: np.float32 = 0.5,
        # use_mask: bool = True, 
        # bg_threshold: float = 100, bg_reverse: bool = False
    ):
        
       
        self.f_list = glob(os.path.join(r"F:\Oguz\EISSA\anomaly_generatin_datasets\images", r'*\*\*'))

        self.augmentors_for_real = [A.RandomRotate90(),
                        # A.Flip(),
                        A.HorizontalFlip(p=1.0),
                        A.VerticalFlip(p=1.0),
                        A.Transpose(),
                        A.OpticalDistortion(p=1.0, distort_limit=1.0),
                        A.OneOf([
                            # A.IAAAdditiveGaussianNoise(),
                            A.GaussNoise(),
                        ], p=0.2),
                        A.OneOf([
                            A.MotionBlur(p=.2),
                            A.MedianBlur(blur_limit=3, p=0.1),
                            A.Blur(blur_limit=3, p=0.1),
                        ], p=0.2),
                        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
                        A.OneOf([
                            A.OpticalDistortion(p=0.3),
                            A.GridDistortion(p=.1),
                            # A.IAAPiecewiseAffine(p=0.3),
                        ], p=0.2),
                        A.OneOf([
                            A.CLAHE(clip_limit=2),
                            # A.IAASharpen(),
                            # A.IAAEmboss(),
                            A.RandomBrightnessContrast(),            
                        ], p=0.3),
                        A.HueSaturationValue(p=0.3)]
        
        
        imagesize = resize[0]
        # mode
        self.is_train = is_train 
        self.to_memory = to_memory
        # self.h, self.w = resize

        # load image file list
        self.datadir = datadir
        self.target = target
        self.file_list = file_list
        # self.file_list = glob(os.path.join(self.datadir, self.target, r'train\*\*' if is_train else r'test\*\*'))

        # synthetic anomaly
        if self.is_train and not self.to_memory:
            # load texture image file list    
            self.texture_source_file_list = glob(os.path.join(texture_source_dir,r'*\*')) if texture_source_dir else None
        
            # perlin noise
            self.perlin_scale = perlin_scale
            self.min_perlin_scale = min_perlin_scale
            self.perlin_noise_threshold = perlin_noise_threshold
            
            # structure
            self.structure_grid_size = structure_grid_size
            
            # anomaly mixing
            self.transparency_range = transparency_range
            
            # mask setting
            # self.use_mask = use_mask
            # self.bg_threshold = bg_threshold
            # self.bg_reverse = bg_reverse
            
        # transform ndarray into tensor
        # self.resize = resize
        # transform
        self.resize = list(resize)
        self.transform_img = [
            transforms.ToPILImage(),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        self.transform_img = transforms.Compose(self.transform_img)

        self.transform_mask = [
            transforms.ToPILImage(),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
        ]
        self.transform_mask = transforms.Compose(self.transform_mask)

        # sythetic anomaly switch
        self.anomaly_switch = False

    def realRandAugmenter(self):
        aug_ind = np.random.choice(np.arange(len(self.augmentors_for_real)), 3, replace=False)
        aug = A.Compose([self.augmentors_for_real[aug_ind[0]],
                         self.augmentors_for_real[aug_ind[1]],
                         self.augmentors_for_real[aug_ind[2]]])
        return aug

    def copy_paste(self, image, mask, n_image):
        aug = self.realRandAugmenter()

       
        # augmente the abnormal region
        augmentated = aug(image=image, mask=mask)
        aug_image, aug_mask = augmentated['image'], augmentated['mask']
        n_image[aug_mask == 255, :] = aug_image[aug_mask == 255, :]

        return n_image, aug_mask/255

    
    def warpAnomaly(self, img, mask, useRotate=True, useResize=True):
        y, x, _ = np.where(mask > 0)
        y0, x0, y1, x1 = y.min(), x.min(), y.max(), x.max()
        # img_result = img[y0:y1, x0:x1]
        # mask_result = mask[y0:y1, x0:x1]
        # height, width = img_result.shape[:2]

        # if useRotate:
        #     x = random.randint(0, 360)
        #     degree = x
        #     M = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)
        #     heightNew = int(
        #         width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree)))
        #     )
        #     widthNew = int(
        #         height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree)))
        #     )

        #     M[0, 2] += (widthNew - width) / 2
        #     M[1, 2] += (heightNew - height) / 2
        #     img_result = cv2.warpAffine(
        #         img_result, M, (widthNew, heightNew), borderValue=(0, 0, 0)
        #     )
        #     mask_result = cv2.warpAffine(
        #         mask_result, M, (widthNew, heightNew), borderValue=(0, 0, 0)
        #     )
        # if useResize:
        #     x = random.randint(-150, 150)
        #     x = x / 1000.0 + 1
        #     H_R = int(height * x)
        #     W_R = int(width * x)
        #     img_result = cv2.resize(img_result, dsize=(H_R, W_R))
        #     mask_result = cv2.resize(mask_result, dsize=(H_R, W_R))
        # y, x, _ = np.where(mask_result > 0)
        # y0, x0, y1, x1 = y.min(), x.min(), y.max(), x.max()
        # mask_result[mask_result < 200] = 0
        # mask_result[mask_result > 0] = 255
        # img_result, mask_result = img_result[y0:y1, x0:x1], mask_result[y0:y1, x0:x1]
        img_result, mask_result = img[y0:y1, x0:x1], mask[y0:y1, x0:x1]
        img_result = img_result * (mask_result > 0)
        return img_result, mask_result
    

    def checkLTPoint(self, empty: np.ndarray, anoamly: np.ndarray):
            H, W, C = empty.shape
            Ha, Wa, C = anoamly.shape
            Hr = H - Ha
            Wr = W - Wa
            h = random.choice(range(Hr))
            w = random.choice(range(Wr))
            return h, w 
        
    
    def copy_paste2(self, a, m, g):

        empty = np.zeros_like(g)
        empty_mask = np.zeros_like(m)
        # extract the mask region
        img_result, mask_result = self.warpAnomaly(
            a, m, useResize=True, useRotate=True
        )
        crop_anomaly = img_result
        # print(crop_anomaly.shape)
        # print(empty.shape)

        crop_mask = mask_result
        # try:
        place_h, place_w = self.checkLTPoint(empty, crop_anomaly)
        # print(place_h, place_w)
        # except:
            # pass
        m_sum = crop_anomaly > 0
        m_sum = m_sum.sum()
        empty[
            place_h : place_h + crop_anomaly.shape[0],
            place_w : place_w + crop_anomaly.shape[1],
        ] = crop_anomaly

        fro_img = (np.ones((empty.shape[0], empty.shape[1], 3), np.uint8) * 255 )
        g[empty > 0] = 0
        fusion_sum = (empty > 0) * (fro_img > 0)
        fusion_sum = fusion_sum > 0
        fusion_sum = fusion_sum.sum()

        # is covered?
        if fusion_sum == m_sum:
            g = g + empty
            empty = g
            empty_mask[
                place_h : place_h + crop_anomaly.shape[0],
                place_w : place_w + crop_anomaly.shape[1],
            ] = (
                crop_mask * 255
            )
            fro_img = fro_img.sum(axis=2)
            empty_mask = empty_mask.sum(axis=2)
            empty_mask[empty_mask > 0] = 255
            fro_img[fro_img > 0] = 127
            # binary_map = fro_img.copy()  # store
            fro_img[empty_mask > 0] = 0
            fro_img += empty_mask
            fro_img = np.stack([fro_img] * 3, axis=2)
            empty_mask = np.stack([empty_mask] * 3, axis=2)
            # binary_map = np.stack([binary_map] * 3, axis=2)
            # triple_map = fro_img.copy()
            # if ishow:  # show image
            #     plt.subplot(231)
            #     plt.imshow(empty[..., ::-1])
            #     plt.subplot(232)
            #     plt.imshow(img_result)
            #     plt.subplot(233)
            #     plt.title("triple_map")
            #     plt.imshow(triple_map)
            #     plt.subplot(234)
            #     plt.imshow(empty_mask)
            #     plt.subplot(235)
            #     plt.title("binary map")
            #     plt.imshow(binary_map)
            #     plt.subplot(236)
            #     plt.title("normal image")
            #     plt.imshow(good_img)
            #     plt.show()
            #     # exit()
            #     plt.savefig(save_path, bbox_inches='tight', dpi=300)
            # else:
            # saving the triple map
        if empty_mask.ndim == 3:
            empty_mask = empty_mask[:, :, 0].astype(np.float32)
        
        # empty_mask[empty_mask > 0] = 1 
        # print(empty_mask.max())

        return empty, empty_mask/255

    

    def __getitem__(self, idx):
        
        
        file_path = self.file_list[idx]
        
        # image
        img = Image.open(file_path).convert("RGB").resize(self.resize)
        img = np.array(img)
        
        # target
        target = 0 if 'good' in self.file_list[idx] else 1
        
        # mask
        maskmode = [6]
        if 'good' in file_path:
            mask = np.zeros(self.resize, dtype=np.float32)
        else:
            mask = Image.open(file_path.replace('test','ground_truth').replace('.png','_mask.png')).resize(self.resize)
            mask = np.array(mask)
            
        ## anomaly source
        if self.is_train and not self.to_memory:

            # data augmentions: Randomaly horizontal flip, vertical flip, rotate by 90, 270 or 180 for input images:
            # p = np.random.uniform()
            p = 1
            if p < 0.5:

                def rotate_with_rnd(img, rnd):
                    mapping = {1: 1, 2: 3, 3: 2}  # نفس القيم اللي عندك
                    k = mapping.get(rnd, 1)
                    return np.ascontiguousarray(np.rot90(img, k))
                img = rotate_with_rnd(img, rnd)

                ####################
                # # rnd = np.random.randint(1,6)
                # rnd = np.random.randint(1,4)

                # if rnd == 1:
                #     # print('data augmentions: Rotate all images by 90')
                #     # transform = A.Compose([
                #     # A.RandomRotate90(p=1.0) ])
                #     # augmented = transform(image=img)
                #     # img = augmented["image"]
                #     aug = iaa.Rot90(1)
                #     img = aug(image=img)

                # elif rnd == 2:
                #     # print('data augmentions: Rotate image by 270')
                #     aug = iaa.Rot90(2)
                #     img = aug(image=img)

                # else:
                #     # print('data augmentions: Rotate image by 180')
                #     aug = iaa.Rot90(3)
                #     img = aug(image=img)

                # # elif rnd == 4:
                # #     # print('Horizontal Flip')
                # #     img = transforms.RandomHorizontalFlip(p=1)(img)
                # # elif rnd == 5:
                # #     # print('Vertical Flip')
                # #     img = transforms.RandomVerticalFlip(p=1)(img)

            if self.anomaly_switch:
                img, mask, maskmode = self.generate_anomaly(img=img, texture_img_list=self.texture_source_file_list)
                target = 1
                self.anomaly_switch = False
            else:

                self.anomaly_switch = True

        # convert ndarray into tensor
        img = self.transform_img(img.copy())

        mask = self.transform_mask(mask).squeeze()

        return img, mask, target, maskmode
        
         
    def rand_augment(self):
        augmenters = [
            iaa.GammaContrast((0.5,2.0),per_channel=True),
            iaa.MultiplyAndAddToBrightness(mul=(0.8,1.2),add=(-30,30)),
            iaa.pillike.EnhanceSharpness(),
            iaa.AddToHueAndSaturation((-50,50),per_channel=True),
            iaa.Solarize(0.5, threshold=(32,128)),
            iaa.Posterize(),
            iaa.Invert(),
            iaa.pillike.Autocontrast(),
            iaa.pillike.Equalize(),
            iaa.Affine(rotate=(-45, 45))
        ]

        aug_idx = np.random.choice(np.arange(len(augmenters)), 3, replace=False)
        aug = iaa.Sequential([
            augmenters[aug_idx[0]],
            augmenters[aug_idx[1]],
            augmenters[aug_idx[2]]
        ])
        
        return aug

    def generate_form_mask (self):
        max_anomaly_regions = 3    # possible max number of anomaly regions
        parts = np.random.randint(1, max_anomaly_regions+1)

        mask = 0
        for part in range(parts):
            # brush params
            maxBrushWidth = np.random.randint(8, 48)
            maxLength = 32
            maxVertex = np.random.randint(12, 16)

            temp_mask = self.np_free_form_mask(maxVertex, maxLength, maxBrushWidth, maxAngle=180, h=self.resize[0], w=self.resize[1])
            mask = np.logical_or(mask, temp_mask)
            mask = mask.astype(np.float32)

        return mask


    def np_free_form_mask(self, maxVertex, maxLength, maxBrushWidth, maxAngle, h, w):
      mask = np.zeros((h, w), np.float32)
      numVertex = np.random.randint(2, maxVertex + 1)
      startY = np.random.randint(20, h-20)
      startX = np.random.randint(20, w-20)
      brushWidth = 0
      pre_angle = 0
      for i in range(numVertex):
          angle = np.random.randint(maxAngle + 1)
          angle = (angle / 360.0 * 2 * np.pi + pre_angle)/2

          if i % 2 == 0:
              angle = 2 * np.pi - angle
          length = np.random.randint(1, maxLength + 1)
          brushWidth = np.random.randint(8, maxBrushWidth + 1) // 2 * 2
          nextY = startY + length * np.cos(angle)
          nextX = startX + length * np.sin(angle)

          nextY = np.maximum(np.minimum(nextY, h - 40), 40).astype(np.int_)
          nextX = np.maximum(np.minimum(nextX, w - 40), 40).astype(np.int_)

          cv2.line(mask, (startY, startX), (nextY, nextX), 1, brushWidth)

          startY, startX = nextY, nextX
          pre_angle = angle
      return mask

    def generate_anomaly(self, img: np.ndarray, texture_img_list: list = None) -> List[np.ndarray]:


        impressions, rate_p = globals.globalimpressions1, globals.rate_p

        
        sequence = [0,0,0,1,1,1]
        # sequence = [1]*6
        prob_weights = np.exp(rate_p * impressions)
        prob_weights /= np.sum(prob_weights)  # Normalize probabilities to sum up to 1
        OE_mode = random.choices(sequence, weights=prob_weights)[0]
        # OE_mode = random.choices(sequence)[0]       

        
        mask = self.generate_perlin_noise_mask()
        # while len(mask[mask == 1]) == 0:
        #     mask = self.generate_perlin_noise_mask()
        if len(mask[mask == 1]) == 0:
            mask = self.generate_form_mask()
        
        mask_expanded = np.expand_dims(mask, axis=2)
        factor = np.random.uniform(*self.transparency_range, size=1)[0]
            
        if OE_mode == 0:
            # print("yapay anomaly")
            anomaly_source_img = self.anomaly_source(img=img, texture_img_list=texture_img_list)
            anomaly_source_img = factor * (mask_expanded * anomaly_source_img) + (1 - factor) * (mask_expanded * img)
            anomaly_img = ((- mask_expanded + 1) * img) + anomaly_source_img 

 
        else:
            # print("real anomaly")
            
            idx1 = np.random.randint(len(self.f_list))
            file_path1 = self.f_list[idx1]
            while 'self.target' in file_path1:
                idx1 = np.random.randint(len(self.f_list))
                file_path1 = self.f_list[idx1]
            
            a_img = Image.open(file_path1).convert("RGB").resize(self.resize)
            a_img = np.array(a_img)
            mask = Image.open(file_path1.replace('images','masks').replace('.png','_mask.png')).convert("RGB").resize(self.resize)
            mask = np.asarray(mask)

            # if mask.ndim == 3:
            #     mask = mask[:, :, 0]  

            # anomaly_img, mask = self.copy_paste(a_img, mask, img)
            anomaly_img, mask = self.copy_paste2(a_img, mask, img)


            


                 
        
        return anomaly_img.astype(np.uint8), mask, [OE_mode]
    

    def interpolant(self, t):
        return t * t * t * (t * (t * 6 - 15) + 10)


    def generate_perlin_noise_2d(self, shape, res, tileable=(False, False), interpolant=interpolant):
        """Generate a 2D numpy array of perlin noise.
        Args:
            shape: The shape of the generated array (tuple of two ints).
                This must be a multple of res.
            res: The number of periods of noise to generate along each
                axis (tuple of two ints). Note shape must be a multiple of
                res.
            tileable: If the noise should be tileable along each axis
                (tuple of two bools). Defaults to (False, False).
            interpolant: The interpolation function, defaults to
                t*t*t*(t*(t*6 - 15) + 10).
        Returns:
            A numpy array of shape shape with the generated noise.
        Raises:
            ValueError: If shape is not a multiple of res.
        """
        delta = (res[0] / shape[0], res[1] / shape[1])
        d = (shape[0] // res[0], shape[1] // res[1])
        grid = np.mgrid[0 : res[0] : delta[0], 0 : res[1] : delta[1]].transpose(1, 2, 0) % 1
        # Gradients
        angles = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1)
        gradients = np.dstack((np.cos(angles), np.sin(angles)))
        if tileable[0]:
            gradients[-1, :] = gradients[0, :]
        if tileable[1]:
            gradients[:, -1] = gradients[:, 0]
        gradients = gradients.repeat(d[0], 0).repeat(d[1], 1)
        g00 = gradients[: -d[0], : -d[1]]
        g10 = gradients[d[0] :, : -d[1]]
        g01 = gradients[: -d[0], d[1] :]
        g11 = gradients[d[0] :, d[1] :]
        # Ramps
        n00 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1])) * g00, 2)
        n10 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1])) * g10, 2)
        n01 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * g01, 2)
        n11 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)) * g11, 2)
        # Interpolation
        t = self.interpolant(grid)
        n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
        n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11
        return np.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1)

   
    
    def generate_perlin_noise_mask(self) -> np.ndarray:
        # define perlin noise scale
        perlin_scalex = 2 ** (torch.randint(self.min_perlin_scale, self.perlin_scale, (1,)).numpy()[0])
        perlin_scaley = 2 ** (torch.randint(self.min_perlin_scale, self.perlin_scale, (1,)).numpy()[0])

        # generate perlin noise
        perlin_noise = rand_perlin_2d_np((self.resize[0], self.resize[1]), (perlin_scalex, perlin_scaley))

        
        
        # apply affine transform
        rot = iaa.Affine(rotate=(-90, 90))
        perlin_noise = rot(image=perlin_noise)
        
        # make a mask by applying threshold
        mask_noise = np.where(
            perlin_noise > self.perlin_noise_threshold, 
            np.ones_like(perlin_noise), 
            np.zeros_like(perlin_noise)
        )
        
        return mask_noise
    
    def anomaly_source(self, img: np.ndarray, texture_img_list: list = None) -> np.ndarray:
        p = np.random.uniform() if texture_img_list else 1.0
        if p < 0.5:
            idx = np.random.choice(len(texture_img_list))
            anomaly_source_img = self._texture_source(texture_img_path=texture_img_list[idx])

        else:
            anomaly_source_img = self._structure_source(img=img)
            
        return anomaly_source_img
        
    def _texture_source(self, texture_img_path: str) -> np.ndarray:
        texture_source_img = cv2.imread(texture_img_path)
        texture_source_img = cv2.cvtColor(texture_source_img, cv2.COLOR_BGR2RGB)
        texture_source_img = cv2.resize(texture_source_img, dsize=(self.resize[1], self.resize[0])).astype(np.float32)
        
        return texture_source_img
        
    def _structure_source(self, img: np.ndarray) -> np.ndarray:
        structure_source_img = self.rand_augment()(image=img)
        
        assert self.resize[0] % self.structure_grid_size == 0, 'structure should be devided by grid size accurately'
        grid_w = self.resize[1] // self.structure_grid_size
        grid_h = self.resize[0] // self.structure_grid_size
        
        structure_source_img = rearrange(
            tensor  = structure_source_img, 
            pattern = '(h gh) (w gw) c -> (h w) gw gh c',
            gw      = grid_w, 
            gh      = grid_h
        )
        disordered_idx = np.arange(structure_source_img.shape[0])
        np.random.shuffle(disordered_idx)

        structure_source_img = rearrange(
            tensor  = structure_source_img[disordered_idx], 
            pattern = '(h w) gw gh c -> (h gh) (w gw) c',
            h       = self.structure_grid_size,
            w       = self.structure_grid_size
        ).astype(np.float32)
        
        return structure_source_img
        
    def __len__(self):
        return len(self.file_list)
    
