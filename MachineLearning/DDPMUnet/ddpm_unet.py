# fmt: off
# flake8: noqa

"""
Original code: https://github.com/keras-team/keras-io/blob/3463ea5cbc650eeabdec92ce754cf1aa1f5acc66/examples/generative/ddpm.py

@license Apache License 2.0 https://github.com/keras-team/keras-io/blob/3463ea5cbc650eeabdec92ce754cf1aa1f5acc66/LICENSE
"""

import keras
from keras import layers
import tensorflow as tf
import math

# Kernel initializer to use
def kernel_init(scale):
    scale = max(scale, 1e-10)
    return keras.initializers.VarianceScaling(
        scale, mode="fan_avg", distribution="uniform"
    )


class AttentionBlock(layers.Layer):
    """Applies self-attention.

    Args:
        units: Number of units in the dense layers
        groups: Number of groups to be used for GroupNormalization layer
    """

    def __init__(self, units, groups=8, **kwargs):
        self.units = units
        self.groups = groups
        super().__init__(**kwargs)

        self.norm = layers.GroupNormalization(groups=groups)
        self.query = layers.Dense(units, kernel_initializer=kernel_init(1.0))
        self.key = layers.Dense(units, kernel_initializer=kernel_init(1.0))
        self.value = layers.Dense(units, kernel_initializer=kernel_init(1.0))
        self.proj = layers.Dense(units, kernel_initializer=kernel_init(0.0))

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]
        scale = tf.cast(self.units, tf.float32) ** (-0.5)

        inputs = self.norm(inputs)
        q = self.query(inputs)
        k = self.key(inputs)
        v = self.value(inputs)

        attn_score = tf.einsum("bhwc, bHWc->bhwHW", q, k) * scale
        attn_score = tf.reshape(attn_score, [batch_size, height, width, height * width])

        attn_score = tf.nn.softmax(attn_score, -1)
        attn_score = tf.reshape(attn_score, [batch_size, height, width, height, width])

        proj = tf.einsum("bhwHW,bHWc->bhwc", attn_score, v)
        proj = self.proj(proj)
        return inputs + proj


class TimeEmbedding(layers.Layer):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.half_dim = dim // 2
        self.emb = math.log(10000) / (self.half_dim - 1)
        self.emb = tf.exp(tf.range(self.half_dim, dtype=tf.float32) * -self.emb)

    def call(self, inputs):
        inputs = tf.cast(inputs, dtype=tf.float32)
        emb = inputs[:, None] * self.emb[None, :]
        emb = tf.concat([tf.sin(emb), tf.cos(emb)], axis=-1)
        return emb


def ResidualBlock(width, groups=8, activation_fn=keras.activations.swish, include_temb=True):
    def apply(inputs):

        if include_temb:
            x, t = inputs
        else:
            x = inputs

        input_width = x.shape[3]

        if input_width == width:
            residual = x
        else:
            residual = layers.Conv2D(
                width, kernel_size=1, kernel_initializer=kernel_init(1.0)
            )(x)

        if include_temb:
            temb = activation_fn(t)
            temb = layers.Dense(width, kernel_initializer=kernel_init(1.0))(temb)[
                :, None, None, :
            ]

        x = layers.GroupNormalization(groups=groups)(x)
        x = activation_fn(x)
        x = layers.Conv2D(
            width, kernel_size=3, padding="same", kernel_initializer=kernel_init(1.0)
        )(x)

        if include_temb:
            x = layers.Add()([x, temb])
        x = layers.GroupNormalization(groups=groups)(x)
        x = activation_fn(x)

        x = layers.Conv2D(
            width, kernel_size=3, padding="same", kernel_initializer=kernel_init(0.0)
        )(x)
        x = layers.Add()([x, residual])
        return x

    return apply


def DownSample(width):
    def apply(x):
        x = layers.Conv2D(
            width,
            kernel_size=3,
            strides=2,
            padding="same",
            kernel_initializer=kernel_init(1.0),
        )(x)
        return x

    return apply


def UpSample(width, interpolation="nearest"):
    def apply(x):
        x = layers.UpSampling2D(size=2, interpolation=interpolation)(x)
        x = layers.Conv2D(
            width, kernel_size=3, padding="same", kernel_initializer=kernel_init(1.0)
        )(x)
        return x

    return apply


def TimeMLP(units, activation_fn=keras.activations.swish):
    def apply(inputs):
        temb = layers.Dense(
            units, activation=activation_fn, kernel_initializer=kernel_init(1.0)
        )(inputs)
        temb = layers.Dense(units, kernel_initializer=kernel_init(1.0))(temb)
        return temb

    return apply


def build_model(
    img_size,
    output_size,
    widths,
    has_attention,
    num_res_blocks=2,
    norm_groups=8,
    interpolation="nearest",
    include_temb=True,
    activation_fn=keras.activations.swish,
):
    image_input = layers.Input(
        shape=img_size, name="image_input"
    )

    if include_temb:
        time_input = keras.Input(shape=(), dtype=tf.int64, name="time_input")

    #x = layers.concatenate([image_input, conditional_image_input])

    x = layers.Conv2D(
        widths[0],
        kernel_size=(3, 3),
        padding="same",
        kernel_initializer=kernel_init(1.0),
    )(image_input)

    if include_temb:
        temb = TimeEmbedding(dim=widths[0] * 4)(time_input)
        temb = TimeMLP(units=widths[0] * 4, activation_fn=activation_fn)(temb)

    skips = [x]

    # DownBlock
    for i in range(len(widths)):
        for _ in range(num_res_blocks):

            if include_temb:
                inputs = [x, temb]
            else:
                inputs = x
            x = ResidualBlock(
                widths[i], groups=norm_groups, activation_fn=activation_fn, include_temb=include_temb
            )(inputs)
            if has_attention[i]:
                x = AttentionBlock(widths[i], groups=norm_groups)(x)
            skips.append(x)

        if widths[i] != widths[-1]:
            x = DownSample(widths[i])(x)
            skips.append(x)

    if include_temb:
        inputs = [x, temb]
    else:
        inputs = x
    # MiddleBlock
    x = ResidualBlock(widths[-1], groups=norm_groups, activation_fn=activation_fn, include_temb=include_temb)(
        inputs
    )
    x = AttentionBlock(widths[-1], groups=norm_groups)(x)

    if include_temb:
        inputs = [x, temb]
    else:
        inputs = x

    x = ResidualBlock(widths[-1], groups=norm_groups, activation_fn=activation_fn, include_temb=include_temb)(
        inputs
    )

    # UpBlock
    for i in reversed(range(len(widths))):
        for _ in range(num_res_blocks + 1):
            x = layers.Concatenate(axis=-1)([x, skips.pop()])

            if include_temb:
                inputs = [x, temb]
            else:
                inputs = x

            x = ResidualBlock(
                widths[i], groups=norm_groups, activation_fn=activation_fn, include_temb=include_temb
            )(inputs)
            if has_attention[i]:
                x = AttentionBlock(widths[i], groups=norm_groups)(x)

        if i != 0:
            x = UpSample(widths[i], interpolation=interpolation)(x)

    # End block
    x = layers.GroupNormalization(groups=norm_groups)(x)
    x = activation_fn(x)
    x = layers.Conv2D(1, (3, 3), padding="same", kernel_initializer=kernel_init(0.0))(x)

    output = x
    
    if include_temb:
        inputs = [image_input, time_input]
    else:
        inputs = [image_input]
    return keras.Model(inputs, output, name="ddpm_unet")