import cntk
from cntk.layers import placeholder, Constant, Sequential
from cntk import parameter, plus, roipooling, normal, times, combine, CloneMethod, cross_entropy_with_softmax, reduce_sum
from utils.rpn.cntk_smoothL1_loss import SmoothL1Loss
from cntk.logging.graph import find_by_name, plot

def create_fast_rcnn_predictor(conv_out, rois, fc_layers, cfg):
    # RCNN
    roi_out = roipooling(conv_out, rois, cntk.MAX_POOLING, (cfg["MODEL"].ROI_DIM, cfg["MODEL"].ROI_DIM), spatial_scale=1/16.0)
    fc_out = fc_layers(roi_out)

    # prediction head
    W_pred = parameter(shape=(4096, cfg["DATA"].NUM_CLASSES), init=normal(scale=0.01), name="cls_score.W")
    b_pred = parameter(shape=cfg["DATA"].NUM_CLASSES, init=0, name="cls_score.b")
    cls_score = plus(times(fc_out, W_pred), b_pred, name='cls_score')

    # regression head
    W_regr = parameter(shape=(4096, cfg["DATA"].NUM_CLASSES*4), init=normal(scale=0.001), name="bbox_regr.W")
    b_regr = parameter(shape=cfg["DATA"].NUM_CLASSES*4, init=0, name="bbox_regr.b")
    bbox_pred = plus(times(fc_out, W_regr), b_regr, name='bbox_regr')

    return cls_score, bbox_pred


def clone_model(base_model, from_node_names, to_node_names, clone_method):
    from_nodes = [find_by_name(base_model, node_name) for node_name in from_node_names]
    if None in from_nodes:
        print("Error: could not find all specified 'from_nodes' in clone. Looking for {}, found {}"
              .format(from_node_names, from_nodes))
    to_nodes = [find_by_name(base_model, node_name) for node_name in to_node_names]
    if None in to_nodes:
        print("Error: could not find all specified 'to_nodes' in clone. Looking for {}, found {}"
              .format(to_node_names, to_nodes))
    input_placeholders = dict(zip(from_nodes, [placeholder() for x in from_nodes]))
    cloned_net = combine(to_nodes).clone(clone_method, input_placeholders)
    return cloned_net

def clone_conv_layers(base_model, cfg):
    feature_node_name = cfg["MODEL"].FEATURE_NODE_NAME
    start_train_conv_node_name = cfg["MODEL"].START_TRAIN_CONV_NODE_NAME
    last_conv_node_name = cfg["MODEL"].LAST_CONV_NODE_NAME
    if not cfg.TRAIN_CONV_LAYERS:
        conv_layers = clone_model(base_model, [feature_node_name], [last_conv_node_name], CloneMethod.freeze)
    elif feature_node_name == start_train_conv_node_name:
        conv_layers = clone_model(base_model, [feature_node_name], [last_conv_node_name], CloneMethod.clone)
    else:
        fixed_conv_layers = clone_model(base_model, [feature_node_name], [start_train_conv_node_name],
                                        CloneMethod.freeze)
        train_conv_layers = clone_model(base_model, [start_train_conv_node_name], [last_conv_node_name],
                                        CloneMethod.clone)
        conv_layers = Sequential([fixed_conv_layers, train_conv_layers])
    return conv_layers

def create_detection_losses(cls_score, label_targets, bbox_pred, rois, bbox_targets, bbox_inside_weights, cfg):
    # The losses are normalized by the batch size
    # classification loss
    p_cls_score = placeholder()
    p_label_targets = placeholder()
    cls_loss = cross_entropy_with_softmax(p_cls_score, p_label_targets, axis=1)
    cls_normalization_factor = 1.0 / cfg.NUM_ROI_PROPOSALS
    normalized_cls_loss = reduce_sum(cls_loss) * cls_normalization_factor

    reduced_cls_loss = cntk.as_block(normalized_cls_loss,
                                     [(p_cls_score, cls_score), (p_label_targets, label_targets)],
                                     'CrossEntropyWithSoftmax', 'norm_cls_loss')

    # regression loss
    p_bbox_pred = placeholder()
    p_bbox_targets = placeholder()
    p_bbox_inside_weights = placeholder()
    bbox_loss = SmoothL1Loss(cfg.SIGMA_DET_L1, p_bbox_pred, p_bbox_targets, p_bbox_inside_weights, 1.0)
    bbox_normalization_factor = 1.0 / cfg.NUM_ROI_PROPOSALS
    normalized_bbox_loss = reduce_sum(bbox_loss) * bbox_normalization_factor

    reduced_bbox_loss = cntk.as_block(normalized_bbox_loss,
                                     [(p_bbox_pred, bbox_pred), (p_bbox_targets, bbox_targets), (p_bbox_inside_weights, bbox_inside_weights)],
                                     'SmoothL1Loss', 'norm_bbox_loss')

    detection_losses = plus(reduced_cls_loss, reduced_bbox_loss, name="detection_losses")

    return detection_losses