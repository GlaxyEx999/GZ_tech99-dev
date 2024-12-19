import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torchvision.models as models
from modelscope.msdatasets import MsDataset
from utils import download
from torchsummary import summary

TRAIN_MODES = ["linear_probe", "full_finetune", "no_pretrain"]

'''
    classifier是一个conv layer，output shape: [7, 258(T)]
    corresponding label shape: [7, 258]
    threshold 0.5 to binarize the output
    loss func is BCE
'''
class Interpolate(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='bilinear', align_corners=False):
        super(Interpolate, self).__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return F.interpolate(x, size=self.size, scale_factor=self.scale_factor,
                             mode=self.mode, align_corners=self.align_corners)

def compute_f1(preds, labels):
    threshold = 0.5
    preds = (preds >= threshold).int()
    labels = (labels >= threshold).int()
    TP_per_frame = (preds & labels).sum(dim=1)  # Sum over classes, shape: [bsz, T]
    FP_per_frame = (preds & ~labels).sum(dim=1)  # False positives, shape: [bsz, T]
    FN_per_frame = (~preds & labels).sum(dim=1)  

    epsilon = 1e-7  # To avoid division by zero
    precision_per_frame = TP_per_frame / (TP_per_frame + FP_per_frame + epsilon)  # Shape: [bsz, T]
    recall_per_frame = TP_per_frame / (TP_per_frame + FN_per_frame + epsilon)  # Shape: [bsz, T]
    f1_per_frame = 2 * precision_per_frame * recall_per_frame / (precision_per_frame + recall_per_frame + epsilon)  # Shape: [bsz, T]
    frame_f1 = f1_per_frame.mean()
    
    return frame_f1

def get_weight(Ytr):  # (2493, 258, 6)
    mp = Ytr.transpose(0, 2, 1)[:].sum(0).sum(0)  # (6,)
    mmp = mp.astype(np.float32) / mp.sum()
    cc = ((mmp.mean() / mmp) * ((1 - mmp) / (1 - mmp.mean()))) ** 0.3
    inverse_feq = torch.from_numpy(cc)
    return inverse_feq


def sp_loss(fla_pred, target, gwe):
    # class-wise weight vector?
    we = gwe.to("cuda" if torch.cuda.is_available() else "cpu")
    wwe = 1
    we *= wwe
    loss = 0
    for _, (out, fl_target) in enumerate(zip(fla_pred, target)):
        twe = we.view(-1, 1).repeat(1, fl_target.size(1)).type(torch.cuda.FloatTensor)
        ttwe = twe * fl_target.data + (1 - fl_target.data) * wwe

        # 已经包含sigmoid和BCE
        loss_fn = nn.BCEWithLogitsLoss(weight=ttwe, size_average=True)
        # print(target.shape)
        loss += loss_fn(torch.squeeze(out), fl_target)

    return loss




class Net:
    def __init__(
        self,
        backbone: str,
        train_mode: int,
        cls_num: int,
        ori_T: int,
        imgnet_ver="v1",
        weight_path="",
    ):
        if not train_mode in range(len(TRAIN_MODES)):
            raise ValueError(f"Unsupported training mode {train_mode}.")

        if not hasattr(models, backbone):
            raise ValueError(f"Unsupported model {backbone}.")

        self.imgnet_ver = imgnet_ver
        self.training = bool(weight_path == "")
        self.full_finetune = bool(train_mode > 0)
        self.type, self.weight_url, self.input_size = self._model_info(backbone)
        self.model: torch.nn.Module = eval("models.%s()" % backbone)
        self.ori_T = ori_T

        self.out_channel_before_classifier = 0
        self._set_outsize() # set out channel size

        self.cls_num = cls_num
        if self.training:
            if train_mode < 2:
                weight_path = self._download_model(self.weight_url)
                checkpoint = (
                    torch.load(weight_path)
                    if torch.cuda.is_available()
                    else torch.load(weight_path, map_location="cpu")
                )
                self.model.load_state_dict(checkpoint, False)

            for parma in self.model.parameters():
                parma.requires_grad = self.full_finetune
            # classfier is a Conv layer outputs [7, 258]
            self._set_classifier()
   
            self.model.train()

        else:
            self._set_classifier()
            checkpoint = (
                torch.load(weight_path)
                if torch.cuda.is_available()
                else torch.load(weight_path, map_location="cpu")
            )
            self.model.load_state_dict(checkpoint, False)

            self.model.eval()

    def _get_backbone(self, backbone_ver, backbone_list):
        for backbone_info in backbone_list:
            if backbone_ver == backbone_info["ver"]:
                return backbone_info

        raise ValueError("[Backbone not found] Please check if --model is correct!")

    def _model_info(self, backbone: str):
        backbone_list = MsDataset.load(
            "monetjoe/cv_backbones",
            split=self.imgnet_ver,
            cache_dir="./__pycache__",
            # download_mode="force_redownload",
        )
        backbone_info = self._get_backbone(backbone, backbone_list)
        return (
            str(backbone_info["type"]),
            str(backbone_info["url"]),
            int(backbone_info["input_size"]),
        )

    def _download_model(self, weight_url: str, model_dir="./__pycache__"):
        weight_path = f'{model_dir}/{weight_url.split("/")[-1]}'
        os.makedirs(model_dir, exist_ok=True)
        if not os.path.exists(weight_path):
            download(weight_url, weight_path)

        return weight_path

    def _create_classifier(self):
        # classifier is a Conv2d, output: [cls_num, T]
        # frequency dim -> cls_num
        # temp_nn = nn.Conv2d(Fequency, cls_num, (3,1), padding=?)
        '''
            channel -> cls_num
            T -> original T size
            F -> 1

            Question: How many deconvolution layer is needed?
                      Maybe we should start with 4.

            Deconvolution + ReLU + BN
        '''
        original_T_size = self.ori_T
        upsample_module = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, None)), # F -> 1
            
            nn.ConvTranspose2d(self.out_channel_before_classifier, 256, kernel_size=(1,4), stride=(1,2), padding=(0,1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),

            nn.ConvTranspose2d(256, 128, kernel_size=(1,4), stride=(1,2), padding=(0,1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),

            nn.ConvTranspose2d(128, 64, kernel_size=(1,4), stride=(1,2), padding=(0,1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),

            nn.ConvTranspose2d(64, 32, kernel_size=(1,4), stride=(1,2), padding=(0,1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),

            # input for Interp: [bsz, C, 1, T]
            Interpolate(size=(1, original_T_size), mode='bilinear', align_corners=False),
            # classifier
            nn.Conv2d(32, 32, kernel_size=(1,1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, self.cls_num, kernel_size=(1,1)) 
        )

        
        return upsample_module

    def _set_outsize(self):
        #### get the output size before classifier ####
        conv2d_out_ch = []
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                conv2d_out_ch.append(module.out_channels) 

            if (
                str(name).__contains__("classifier")
                or str(name).__eq__("fc")
                or str(name).__contains__("head")
                # or hasattr(module, "classifier")
            ):
                if isinstance(module, torch.nn.Conv2d): 
                    conv2d_out_ch.append(module.in_channels)
                    break

        self.out_channel_before_classifier = conv2d_out_ch[-1]


    def _set_classifier(self):
        #### set custom classifier ####
        if self.type == "convnext":
            del self.model.classifier[2]
            self.model.classifier = nn.Sequential(
                *list(self.model.classifier)
                + list(self._create_classifier())
            )
            self.classifier = self.model.classifier

        elif self.type == "maxvit":
            del self.model.classifier[5]
            self.model.classifier = nn.Sequential(
                *list(self.model.classifier)
                + list(self._create_classifier())
            )
            self.classifier = self.model.classifier

        elif hasattr(self.model, "classifier"):
            '''e.g., squeezenet1_1'''
            # self.model.classifier = self._create_classifier(cls_num, linear_output)
            self.model.classifier = self._create_classifier()
            self.classifier = self.model.classifier

        elif hasattr(self.model, "fc"):
            self.model.fc = self._create_classifier()
            self.classifier = self.model.fc

        elif hasattr(self.model, "head"):
            self.model.head = self._create_classifier()
            self.classifier = self.model.head

        else:
            self.model.heads.head = self._create_classifier()
            self.classifier = self.model.heads.head

        for parma in self.classifier.parameters():
            parma.requires_grad = True

    def get_input_size(self):
        return self.input_size

    def forward(self, x):
        if torch.cuda.is_available():
            x = x.cuda()
            self.model = self.model.cuda()

        if self.type == "googlenet" and self.training:
            return self.model(x)[0]
        else:
            # return self.model[:-2](x)
            out = self.model(x)
            return out if out.dim()==3 else out.view(out.size(0), self.cls_num, self.ori_T)

    def parameters(self):
        if self.full_finetune:
            return self.model.parameters()
        else:
            return self.classifier.parameters()

    def state_dict(self):
        return self.model.state_dict()
