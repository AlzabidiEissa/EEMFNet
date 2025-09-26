import torch.nn as nn
from .decoder import Decoder
from .msff import MSFF

import torch
import torch.nn as nn
import torch.nn.functional as F


class EEMFNet(nn.Module):
    # def __init__(self, memory_bank, feature_extractor, feature_extractor1, device='cpu'):
    def __init__(self, feature_extractor, device='cpu'):
        super(EEMFNet, self).__init__()
        
        # self.transform = transforms.Compose([
        #     transforms.ToPILImage()])
        
        # self.a = T_Attention()
        in_channels = []
        for o in feature_extractor(torch.randn(1, 3, 224, 224).to(device)):
                    in_channels.append(o.size(1))
        
        # self.r = FI()

        # self.memory_bank = memory_bank
        self.feature_extractor = feature_extractor
        # self.feature_extractor1 = feature_extractor1
        self.msff = MSFF(in_channels[1:-1])
        self.decoder = Decoder(in_channels)
        
        ############
        # self.edge_cat = ConcatNet(nn.BatchNorm2d)
        # self.edge = Sub_MGAI2(nn.BatchNorm2d, dim=256, num_clusters=32, dropout=0.1)        

    def forward(self, x):
        
        
        features = self.feature_extractor(x)
        f_in = features[0]
        f_out = features[-1]
        f_ii = features[1:-1]

        # # extract concatenated information(CI)
        # concat_features = self.memory_bank.select(features = f_ii)

        # Multi-scale Feature Fusion(MSFF) Module
        # msff_outputs = self.msff(features = concat_features)
        msff_outputs = self.msff(features = f_ii)

        # decoder
        predicted_mask = self.decoder(
            encoder_output  = f_out,
            concat_features = [f_in] + msff_outputs
        )
        return predicted_mask

        # # print(predicted_mask[:,1,:].unsqueeze(1).shape,'.....................................')
        # pred_mask = self.r(predicted_mask,predicted_mask1)
        # # pred_mask = self.r(predicted_mask[:,1,:].unsqueeze(1),predicted_mask1[:,1,:].unsqueeze(1))
        # # print(pred_mask.shape,'..............................................')
        # return pred_mask
