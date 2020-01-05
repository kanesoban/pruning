import tensorflow as tf
from tensorflow.python.eager.backprop import GradientTape
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.keras import backend
from tensorflow.keras.models import Model
from tqdm import tqdm
import numpy as np


'''
def insert_prune_masks(model: tf.keras.Model, position='after'):
    # Auxiliary dictionary to describe the network graph
    network_dict = {'input_layers_of': {}, 'new_output_tensor_of': {}}
    mask_tensors = []

    # Set the input layers of each layer
    for layer in model.layers:
        for node in layer.outbound_nodes:
            layer_name = node.outbound_layer.name
            if layer_name not in network_dict['input_layers_of']:
                network_dict['input_layers_of'].update(
                    {layer_name: [layer.name]})
            else:
                network_dict['input_layers_of'][layer_name].append(layer.name)

    # Set the output tensor of the input layer
    network_dict['new_output_tensor_of'].update(
        {model.layers[0].name: model.input})

    # Iterate over all layers after the input
    for layer in model.layers[1:]:

        # Determine input tensors
        layer_input = [network_dict['new_output_tensor_of'][layer_aux]
                       for layer_aux in network_dict['input_layers_of'][layer.name]]
        if len(layer_input) == 1:
            layer_input = layer_input[0]

        # Insert layer if type is CNN
        if isinstance(layer, tf.keras.layers.Conv2D):
            if position == 'replace':
                x = layer_input
            elif position == 'after':
                x = layer(layer_input)
            elif position == 'before':
                pass
            else:
                raise ValueError('position must be: before, after or replace')

            mask_tensor = tf.ones(layer.output.shape[1:])
            mask_tensors.append(mask_tensor)
            new_layer = tf.keras.layers.Lambda(lambda x: x * mask_tensor)
            x = new_layer(x)
            print('Layer {} inserted after layer {}'.format(new_layer.name, layer.name))
        else:
            x = layer(layer_input)

        # Set new output tensor (the original one, or the one of the inserted
        # layer)
        network_dict['new_output_tensor_of'].update({layer.name: x})

    return Model(inputs=model.inputs, outputs=x), mask_tensors
'''


def insert_prune_masks(model: tf.keras.Model):
    mask_tensors = []
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            def decorator(f):
                mask_tensor = np.ones(layer.output.shape[1:])

                def call(*args, **kwargs):
                    result = f(*args, **kwargs)
                    result = result * mask_tensor
                    return result
                return call, mask_tensor
            c, mask_tensor = decorator(layer.call)
            layer.call = c
            mask_tensors.append(mask_tensor)
    return model, mask_tensors


def watch_layer(layer, tape):
    def decorator(f):
        def call(*args, **kwargs):
            result = f(*args, **kwargs)
            layer.layer_result = result
            tape.watch(result)
            return result
        return call

    layer.old_call = layer.call
    layer.call = decorator(layer.call)


def unwatch_layer(layer):
    layer.call = layer.old_call
    delattr(layer, 'old_call')


def unwatch_all_layers(layers):
    for layer in layers:
        if hasattr(layer, 'old_call'):
            unwatch_layer(layer)


def calculate_z_derivatives(model: Model, inputs, targets, cnn_layers):
    omega_te = []
    for l, cnn_layer in enumerate(cnn_layers):
        total_loss = 0
        with GradientTape() as tape:
            if l > 0:
                unwatch_layer(cnn_layers[l-1])
            watch_layer(cnn_layer, tape)
            outs = model(inputs)

            loss_fns = [
                loss_fn for loss_fn in model.loss_functions if loss_fn is not None
            ]

            i = 0
            loss_fn = loss_fns[0]
            weights = None
            with backend.name_scope(model.output_names[i] + '_loss'):
                per_sample_losses = loss_fn.call(targets[i], outs[i])
                weighted_losses = losses_utils.compute_weighted_loss(
                    per_sample_losses,
                    sample_weight=weights,
                    reduction=losses_utils.ReductionV2.NONE)
                loss_reduction = loss_fn.reduction

                # `AUTO` loss reduction defaults to `SUM_OVER_BATCH_SIZE` for all
                # compile use cases.
                if loss_reduction == losses_utils.ReductionV2.AUTO:
                    loss_reduction = losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE

                # Compute the stateless loss value.
                output_loss = losses_utils.reduce_weighted_loss(
                    weighted_losses, reduction=loss_reduction)

            total_loss += model._loss_weights_list[i] * output_loss

        r = tape.gradient(total_loss, cnn_layer.layer_result) * cnn_layer.layer_result
        shape = r.shape[:-1]
        r = np.sum(r, axis=0)
        r = np.sum(r, axis=0)
        r = np.sum(r, axis=0)
        r /= shape[0] * shape[1] * shape[2]
        omega_te.append(r)
    unwatch_layer(cnn_layers[-1])
    return np.array(omega_te)


def get_cnn_layers(model: tf.keras.Model):
    cnn_layers = []
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            cnn_layers.append(layer)
    return cnn_layers


def nullify_channel(omega_te, mask_tensors):
    min_layer_id = 0
    min_id = None
    min_val = np.inf
    for l in range(len(omega_te)):
        id = np.argmin(omega_te[l])
        # Check if this channel is already zeroed out
        if mask_tensors[l][0, 0, id] == 0:
            continue
        val = omega_te[l][id]
        if val < min_val:
            min_layer_id = l
            min_id = id
            min_val = val
    mask_tensors[min_layer_id][:, :, min_id] = 0


def prune_model(model: tf.keras.Model, data_train: np.ndarray, label_train: np.ndarray, data_test: np.ndarray, label_test: np.ndarray, batch_size: int, prune_iterations: int):
    print('Inserting pruning masks')
    model, mask_tensors = insert_prune_masks(model)
    cnn_layers = get_cnn_layers(model)
    batches = data_train.shape[0] // batch_size
    for i in tqdm(range(prune_iterations)):
        data_batch = data_train[:batch_size]
        label_batch = label_train[:batch_size]
        omega_te = calculate_z_derivatives(model, data_batch, label_batch, cnn_layers)
        for batch in range(1, batches):
            data_batch = data_train[i * batch_size:(i+1) * batch_size]
            label_batch = label_train[i * batch_size:(i+1) * batch_size]
            omega_te += calculate_z_derivatives(model, data_batch, label_batch, cnn_layers)
        omega_te /= batches

        print('Pruning...')
        nullify_channel(omega_te, mask_tensors)
        print('Retraining...')
        model.fit(data_train, label_train, validation_data=(data_test, label_test), batch_size=batch_size, epochs=1)
    # Create new model based on mask
    # TODO
    return model
