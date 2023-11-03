# Includes

import tensorflow as tf
import numpy as np

# Further imports are NOT allowed, please use the APIs in `tf`, `tf.keras` and `tf.keras.layers`!


def create_micro_kws_student_model(
    model_settings: dict, model_name: str = "micro_kws_student"
) -> tf.keras.Model:
    """Builds a MicroKWS model with the keras API.

    Arguments
    ---------
    model_settings : dict
        Dict of different settings for model training.

    Returns
    -------
    model : tf.keras.Model
        Model of the 'Student' architecture.
    """

    # Get relevant model setting.
    input_frequency_size = model_settings["dct_coefficient_count"]
    input_time_size = model_settings["spectrogram_length"]

    inputs = tf.keras.Input(shape=(model_settings["fingerprint_size"]), name="input")

    ### ENTER STUDENT CODE BELOW ###

    # Hint: The following code is just an example, for the final challenge,
    # you will need to add more layers here.

    # Reshape the flattened input.
    x = tf.reshape(inputs, shape=(-1, input_time_size, input_frequency_size, 1))

    # First convolution.
    x = tf.keras.layers.DepthwiseConv2D(
        depth_multiplier=4,
        kernel_size=(5, 4),
        strides=(2, 2),
        padding="SAME",
        activation="relu",
    )(x)

    # Flatten for fully connected layers.
    x = tf.keras.layers.Flatten()(x)

    # Output fully connected.
    output = tf.keras.layers.Dense(units=model_settings["label_count"], activation="softmax")(x)

    ### ENTER STUDENT CODE ABOVE ###

    model = tf.keras.Model(inputs, output)
    model._name = model_name
    return model
