#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os

"""
PLEASE READ paths.md FOR INFORMATION TO HOW TO SET THIS UP
"""

#nnUNet_raw = os.environ.get('nnUNet_raw')
#nnUNet_preprocessed = os.environ.get('nnUNet_preprocessed')
#nnUNet_results = os.environ.get('nnUNet_results')

monai_raw = "/home/user/seong_test/monai_tutorial/NCCT_UNet/dataset/monai_raw"
monai_preprocessed = "/home/user/seong_test/nnUnetFrame/dataset/nnUNet_preprocessed"
monai_results = "/home/user/seong_test/nnUnetFrame/dataset/nnUNet_trained_models"


if monai_raw is None:
    print("monai_raw is not defined and nnU-Net can only be used on data for which preprocessed files "
          "are already present on your system. nnU-Net cannot be used for experiment planning and preprocessing like "
          "this. If this is not intended, please read documentation/setting_up_paths.md for information on how to set "
          "this up properly.")

if monai_preprocessed is None:
    print("monai_preprocessed is not defined and nnU-Net can not be used for preprocessing "
          "or training. If this is not intended, please read documentation/setting_up_paths.md for information on how "
          "to set this up.")

if monai_results is None:
    print("monai_results is not defined and nnU-Net cannot be used for training or "
          "inference. If this is not intended behavior, please read documentation/setting_up_paths.md for information "
          "on how to set this up.")
