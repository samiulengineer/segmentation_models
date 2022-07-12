
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from einops import rearrange

learning_rate = 0.001
weight_decay = 0.0001
batch_size = 2
num_epochs = 1
image_size = 512  # We'll resize input images to this size
patch_size = 16  # Size of the patches to be extract from the input images
num_patches = (image_size // patch_size) ** 2
projection_dim = 192
num_heads = 4
input_shape = (512, 512, 3)
num_class = 2
transformer_units = [
    projection_dim * 3,
    projection_dim,
]  # Size of the transformer layers
transformer_layers = 8


def Patches(images, patch_size):
    batch_size = tf.shape(images)[0]
    patches = tf.image.extract_patches(
            images=images,
            sizes=[1, patch_size, patch_size, 1],
            strides=[1, patch_size, patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
    patch_dims = patches.shape[-1]
    patches = tf.reshape(patches, [batch_size, -1, patch_dims])
    return patches

def PatchEncoder(patch, num_patches, projection_dim):
    positions = tf.range(start=0, limit=num_patches, delta=1)
    projection = layers.Dense(units=projection_dim)(patch)
    position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )(positions)
    return projection + position_embedding

def Block(encoded_patches, num_heads, projection_dim,  transformer_units, dropout=0.1):
    x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=dropout
        )(x1, x1)
    x2 = layers.Add()([attention_output, encoded_patches])
    x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
    dense1 = layers.Dense(transformer_units, activation=tf.nn.gelu)(x3)
    drop = layers.Dropout(dropout)(dense1)
    dense2 = layers.Dense(projection_dim, activation=tf.nn.gelu)(drop)
    drop2 = layers.Dropout(dropout)(dense2)
    x4 = layers.Add()([drop2,x2])
    return x4

def MaskTransformer(x, n_cls, patch_size, d_encoder, n_layers, n_heads, d_model, dropout, im_size):
    H, W = im_size
    GS = H // patch_size
    scale = d_model ** -0.5

    cls_emb = tf.Variable(tf.random.truncated_normal((1, n_cls, d_model), stddev=0.2, seed=123), trainable=False, name="cls_emb")
    proj_patch = tf.Variable(scale * tf.random.normal((d_model, d_model)), name="proj_patch")
    proj_classes = tf.Variable(scale * tf.random.normal((d_model, d_model)), name="proj_classes")

    proj_dec = layers.Dense(d_model)(x)
    x = tf.concat([proj_dec, cls_emb], 1)
    for _ in n_layers:
        x = Block(x, n_heads, d_model, d_model*4, dropout)
    
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    patches, cls_seg_feat = x[:, : -n_cls], x[:, -n_cls :]
    patches = patches @ proj_patch
    cls_seg_feat = cls_seg_feat @ proj_classes
    patches = patches / tf.norm(patches, axis=-1, keepdims=True)
    cls_seg_feat = cls_seg_feat / tf.norm(cls_seg_feat, axis=-1, keepdims=True)

    masks = patches @ tf.transpose(cls_seg_feat, perm=[0,2,1])
    mask_norm = layers.LayerNormalization(epsilon=1e-6)(masks)
    masks = rearrange(mask_norm, "b (h w) n -> b h w n", h=int(GS))
    return masks

def segmenter(config):
    inputs = layers.Input(shape=input_shape)
    # Augment data.
    # Create patches.
    patches = Patches(patch_size)(inputs)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        encoded_patches = Block(num_heads, projection_dim, projection_dim*3, 0.1)(encoded_patches)

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    head = layers.Dense(1000)(representation)
    #logits = DecoderLinear(n_cls=num_class, patch_size=patch_size)(head)
    masks = MaskTransformer(n_cls=num_class, patch_size=patch_size, d_encoder=projection_dim, 
                            n_layers=2, n_heads=3, d_model=projection_dim, dropout=0.1)(head)
    logits = layers.UpSampling2D((16,16), interpolation="bilinear")(masks)
    

    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits)
    return model
