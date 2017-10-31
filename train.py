import os
import sys
import argparse
import numpy as np
import cntk as C
from cntk import UnitType, Trainer, CloneMethod, combine, plus, softmax, element_times
from cntk.learners import momentum_schedule, learning_rate_schedule, momentum_sgd
from cntk.logging import log_number_of_parameters, ProgressPrinter
from cntk.metrics import classification_error

from utils.od_mb_source import ObjectDetectionMinibatchSource
from utils.fast_rcnn_utils import clone_model, clone_conv_layers, create_fast_rcnn_predictor, create_detection_losses
from utils.rpn.rpn_helpers import create_rpn, create_proposal_target_layer, create_proposal_layer
from utils.config_helpers import merge_configs

from FasterRCNN_train import prepare

abs_path = os.path.dirname(os.path.abspath(__file__)) 
sys.path.append(abs_path) 

model_path = os.path.join(abs_path, "model")

def get_configuration():
    # load configs for detector, base network and data set
    from FasterRCNN_config import cfg as detector_cfg
    from utils.configs.VGG16_config import cfg as network_cfg
    from utils.configs.Building100_config import cfg as dataset_cfg

    return merge_configs([detector_cfg, network_cfg, dataset_cfg])

# Creates a reader for base64 images
def create_train_reader(cfg):

    return ObjectDetectionMinibatchSource(
        cfg["DATA"].TRAIN_MAP_FILE,
        cfg["DATA"].TRAIN_ROI_FILE,
        num_classes=cfg["DATA"].NUM_CLASSES,
        max_annotations_per_image=cfg.INPUT_ROIS_PER_IMAGE,
        pad_width=cfg.IMAGE_WIDTH,
        pad_height=cfg.IMAGE_HEIGHT,
        pad_value=cfg["MODEL"].IMG_PAD_COLOR,
        randomize=True,
        use_flipping=cfg["TRAIN"].USE_FLIPPED,
        max_images=cfg["DATA"].NUM_TRAIN_IMAGES,
        proposal_provider=None)

def create_network(cfg):

    ##create input variables
    features = C.input_variable(shape=(cfg.NUM_CHANNELS, cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH),
                                 dynamic_axes=[C.Axis.default_batch_axis()],
                                 name=cfg["MODEL"].FEATURE_NODE_NAME)
    ##roi_iput
    scaled_gt_boxes = C.input_variable((cfg.INPUT_ROIS_PER_IMAGE, 5), dynamic_axes=[C.Axis.default_batch_axis()])
    dims_in = C.input_variable((6), dynamic_axes=[C.Axis.default_batch_axis()])
    dims_input = C.alias(dims_in, name='dims_input')

    # Load the pre-trained classification net and clone layers
    base_model = C.load_model(cfg['BASE_MODEL_PATH'])
    conv_layers = clone_conv_layers(base_model, cfg)
    fc_layers = clone_model(base_model, [cfg["MODEL"].POOL_NODE_NAME], [cfg["MODEL"].LAST_HIDDEN_NODE_NAME], clone_method=CloneMethod.clone)

    # Normalization and conv layers
    feat_norm = features - C.Constant([[[v]] for v in cfg["MODEL"].IMG_PAD_COLOR])
    conv_out = conv_layers(feat_norm)

    # RPN and prediction targets
    rpn_rois, rpn_losses = create_rpn(conv_out, scaled_gt_boxes, dims_input, cfg)

    rois, label_targets, bbox_targets, bbox_inside_weights = create_proposal_target_layer(rpn_rois, scaled_gt_boxes, cfg)

    # Fast RCNN and losses
    cls_score, bbox_pred = create_fast_rcnn_predictor(conv_out, rois, fc_layers, cfg)
    detection_losses = create_detection_losses(cls_score, label_targets, bbox_pred, rois, bbox_targets, bbox_inside_weights, cfg)
    loss = rpn_losses + detection_losses
    pred_error = classification_error(cls_score, label_targets, axis=1)


    e2e_lr_factor = cfg["MODEL"].E2E_LR_FACTOR
    e2e_lr_per_sample_scaled = [x * e2e_lr_factor for x in cfg["CNTK"].E2E_LR_PER_SAMPLE]
    mm_schedule = momentum_schedule(cfg["CNTK"].MOMENTUM_PER_MB)

    print("Using base model:   {}".format(cfg["MODEL"].BASE_MODEL))
    print("lr_per_sample:      {}".format(e2e_lr_per_sample_scaled))
    
    return {
        'features' : features,
        'roi_input' : scaled_gt_boxes,
        'loss' : loss,
        'pred_error' : pred_error,
        'dim_input': dims_in
    }

def train(network, trainer, od_minibatch_source, reader_dev, minibatch_size, epochs_to_train, cfg):
    
    input_map = {
        od_minibatch_source.image_si: network['features'],
        od_minibatch_source.roi_si: network['roi_input'],
        od_minibatch_source.dims_si: network['dim_input']
    }
    # Train all minibatches 

    progress_printer = ProgressPrinter(tag='Training', num_epochs=epochs_to_train)
    
    for epoch in range(epochs_to_train):       # loop over epochs
        sample_count = 0
        while sample_count < cfg["DATA"].NUM_TRAIN_IMAGES:  # loop over minibatches in the epoch
            data = od_minibatch_source.next_minibatch(min(cfg.MB_SIZE, cfg["DATA"].NUM_TRAIN_IMAGES-sample_count), input_map=input_map)
            trainer.train_minibatch(data)                                    # update model with it
            sample_count += trainer.previous_minibatch_sample_count          # count samples processed so far
            progress_printer.update_with_trainer(trainer, with_metric=True)  # log progress
            if sample_count % 100 == 0:
                print("Processed {} samples".format(sample_count))

        progress_printer.epoch_summary(with_metric=True)

    return create_faster_rcnn_eval_model(network['loss'], network['features'], network['dim_input'], cfg)

def create_faster_rcnn_eval_model(model, image_input, dims_input, cfg, rpn_model=None):
    
    print("creating eval model")
    last_conv_node_name = cfg["MODEL"].LAST_CONV_NODE_NAME
    conv_layers = clone_model(model, [cfg["MODEL"].FEATURE_NODE_NAME], [last_conv_node_name], CloneMethod.freeze)
    conv_out = conv_layers(image_input)

    model_with_rpn = model if rpn_model is None else rpn_model
    rpn = clone_model(model_with_rpn, [last_conv_node_name], ["rpn_cls_prob_reshape", "rpn_bbox_pred"], CloneMethod.freeze)
    rpn_out = rpn(conv_out)
    # we need to add the proposal layer anew to account for changing configs when buffering proposals in 4-stage training
    rpn_rois = create_proposal_layer(rpn_out.outputs[0], rpn_out.outputs[1], dims_input, cfg)

    roi_fc_layers = clone_model(model, [last_conv_node_name, "rpn_target_rois"], ["cls_score", "bbox_regr"], CloneMethod.freeze)
    pred_net = roi_fc_layers(conv_out, rpn_rois)
    cls_score = pred_net.outputs[0]
    bbox_regr = pred_net.outputs[1]

    if cfg.BBOX_NORMALIZE_TARGETS:
        num_boxes = int(bbox_regr.shape[1] / 4)
        bbox_normalize_means = np.array(cfg.BBOX_NORMALIZE_MEANS * num_boxes)
        bbox_normalize_stds = np.array(cfg.BBOX_NORMALIZE_STDS * num_boxes)
        bbox_regr = plus(element_times(bbox_regr, bbox_normalize_stds), bbox_normalize_means, name='bbox_regr')

    cls_pred = softmax(cls_score, axis=1, name='cls_pred')
    eval_model = combine([cls_pred, rpn_rois, bbox_regr])

    return eval_model


# Create trainer 
def create_trainer(loss, pred_error, lr_per_sample, mm_schedule, l2_reg_weight, epochs_to_train, cfg): 

    # Set learning parameters 
    if isinstance(loss, C.Variable):
        loss = C.combine([loss])

    params = loss.parameters
    biases = [p for p in params if '.b' in p.name or 'b' == p.name]
    others = [p for p in params if not p in biases]
    
    bias_lr_mult = cfg["CNTK"].BIAS_LR_MULT

    lr_schedule = learning_rate_schedule(lr_per_sample, unit=UnitType.sample)
    learner = momentum_sgd(others, lr_schedule, mm_schedule, l2_regularization_weight=l2_reg_weight,
                           unit_gain=False, use_mean_gradient=True)

    bias_lr_per_sample = [v * bias_lr_mult for v in lr_per_sample]
    bias_lr_schedule = learning_rate_schedule(bias_lr_per_sample, unit=UnitType.sample)
    bias_learner = momentum_sgd(biases, bias_lr_schedule, mm_schedule, l2_regularization_weight=l2_reg_weight, unit_gain=False, use_mean_gradient=True)
    return Trainer(None, (loss, pred_error), [learner, bias_learner])


## Train and evaluate the network.
def faster_rcnn(max_epochs=100, l1_regularization_weight=0, l2_regularization_weight=0):

    cfg = get_configuration()

    prepare(cfg, False)
    network = create_network(cfg)
    model_path = cfg['MODEL_PATH']

    print(model_path)

    e2e_lr_factor = cfg["MODEL"].E2E_LR_FACTOR
    e2e_lr_per_sample_scaled = [x * e2e_lr_factor for x in cfg["CNTK"].E2E_LR_PER_SAMPLE]
    mm_schedule = momentum_schedule(cfg["CNTK"].MOMENTUM_PER_MB)

    trainer = create_trainer(network['loss'], network['pred_error'], e2e_lr_per_sample_scaled, mm_schedule, cfg["CNTK"].L2_REG_WEIGHT, cfg["CNTK"].E2E_MAX_EPOCHS, cfg) 
    
    reader_train = create_train_reader(cfg)

    eval_model = train(network, trainer, reader_train, None, cfg.MB_SIZE, cfg["CNTK"].E2E_MAX_EPOCHS, cfg) 
    eval_model.save(model_path)

## Main function
if __name__=='__main__':

    data_path  = os.path.join(abs_path, "DataSets", "building100") 
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-datadir', '--datadir', help='Data directory where the CIFAR dataset is located', required=False, default=data_path) 
    parser.add_argument('-outputdir', '--outputdir', help='Output directory for checkpoints and models', required=False, default=None) 
    parser.add_argument('-n', '--num_epochs', help='Total number of epochs to train', type=int, required=False, default='100') 
    parser.add_argument('-m', '--minibatch_size', help='Minibatch size', type=int, required=False, default='32') 
    parser.add_argument('-e', '--epoch_size', help='Epoch size', type=int, required=False, default='50000') 
    parser.add_argument('-device', '--device', type=int, help="Force to run the script on a specified device", required=False, default=None) 

    args = vars(parser.parse_args())

    if args['outputdir'] is not None: 
        model_path = args['outputdir'] + "/models" 
    if args['device'] is not None: 
        C.device.try_set_default_device(C.device.gpu(args['device'])) 
 
    C.device.try_set_default_device(C.device.cpu())
    faster_rcnn( max_epochs=args['num_epochs'] )