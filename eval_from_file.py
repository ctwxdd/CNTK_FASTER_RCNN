# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os
import numpy as np
import cntk
from cntk import load_model
from FasterRCNN_train import prepare, train_faster_rcnn, store_eval_model_with_native_udf
from FasterRCNN_eval import compute_test_set_aps, FasterRCNN_Evaluator
from utils.config_helpers import merge_configs
from utils.plot_helpers import plot_test_set_results, plot_test_file_results

def get_configuration():
    # load configs for detector, base network and data set
    from FasterRCNN_config import cfg as detector_cfg
    # for VGG16 base model use:         from utils.configs.VGG16_config import cfg as network_cfg
    # for AlexNet base model use:       from utils.configs.AlexNet_config import cfg as network_cfg
    from utils.configs.VGG16_config import cfg as network_cfg
    # for Pascal VOC 2007 data set use: from utils.configs.Pascal_config import cfg as dataset_cfg
    # for the Grocery data set use:     from utils.configs.Grocery_config import cfg as dataset_cfg
    from utils.configs.Building100_config import cfg as dataset_cfg

    return merge_configs([detector_cfg, network_cfg, dataset_cfg])

# trains and evaluates a Fast R-CNN model.
if __name__ == '__main__':
    cfg = get_configuration()
    prepare(cfg, False)
    #cntk.device.try_set_default_device(cntk.device.gpu(cfg.GPU_ID))
    cntk.device.try_set_default_device(cntk.device.cpu())
    model_path = cfg['MODEL_PATH']
    print(model_path)

    if os.path.exists(model_path) and cfg["CNTK"].MAKE_MODE:
        print("Loading existing model from %s" % model_path)
        eval_model = load_model(model_path)
    else:
        print("No trained model found.")
        exit()

    # Plot results on test set images
    results_folder = os.path.join(cfg.OUTPUT_PATH, cfg["DATA"].DATASET)
    evaluator = FasterRCNN_Evaluator(eval_model, cfg)

    for i in range(2):
        a = input("filename:")
        plot_test_file_results(evaluator, 'D:\\src\\CNTK_Faster_RCNN\\test_img\\' + a, 'D:\\src\\CNTK_Faster_RCNN\\test_img\\', cfg)


