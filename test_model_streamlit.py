import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers # For custom layer definitions
import numpy as np
from PIL import Image
import json
import os

# --- RepVGGBlock Class Definition (Latest Verified Version) ---
# Users will need this definition if it's a custom layer in your model.
class RepVGGBlock(layers.Layer):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 groups=1, deploy=False, use_se=False, **kwargs):
        super(RepVGGBlock, self).__init__(**kwargs)
        self.config_initial_in_channels = in_channels
        self.config_out_channels = out_channels
        self.config_kernel_size = kernel_size
        self.config_strides_val = stride
        self.config_groups = groups
        self._deploy_mode_internal = deploy
        self.config_use_se = use_se
        self.actual_in_channels = None

        self.rbr_dense_conv = layers.Conv2D(
            filters=self.config_out_channels, kernel_size=self.config_kernel_size,
            strides=self.config_strides_val, padding='same',
            groups=self.config_groups, use_bias=False, name=self.name + '_dense_conv'
        )
        self.rbr_dense_bn = layers.BatchNormalization(name=self.name + '_dense_bn')
        self.rbr_1x1_conv = layers.Conv2D(
            filters=self.config_out_channels, kernel_size=1,
            strides=self.config_strides_val, padding='valid',
            groups=self.config_groups, use_bias=False, name=self.name + '_1x1_conv'
        )
        self.rbr_1x1_bn = layers.BatchNormalization(name=self.name + '_1x1_bn')
        self.rbr_identity_bn = None
        self.rbr_reparam = layers.Conv2D(
            filters=self.config_out_channels, kernel_size=self.config_kernel_size,
            strides=self.config_strides_val, padding='same',
            groups=self.config_groups, use_bias=True, name=self.name + '_reparam_conv'
        )

    def build(self, input_shape):
        self.actual_in_channels = input_shape[-1]
        if self.config_initial_in_channels is None:
            self.config_initial_in_channels = self.actual_in_channels
        elif self.config_initial_in_channels != self.actual_in_channels:
            raise ValueError(f"Input channel mismatch for layer {self.name}: Expected {self.config_initial_in_channels}, got {self.actual_in_channels}")

        if self.rbr_identity_bn is None and \
           self.actual_in_channels == self.config_out_channels and self.config_strides_val == 1:
            self.rbr_identity_bn = layers.BatchNormalization(name=self.name + '_identity_bn')
        
        super(RepVGGBlock, self).build(input_shape)

        if not self.rbr_dense_conv.built: self.rbr_dense_conv.build(input_shape)
        if not self.rbr_dense_bn.built: self.rbr_dense_bn.build(self.rbr_dense_conv.compute_output_shape(input_shape))
        if not self.rbr_1x1_conv.built: self.rbr_1x1_conv.build(input_shape)
        if not self.rbr_1x1_bn.built: self.rbr_1x1_bn.build(self.rbr_1x1_conv.compute_output_shape(input_shape))
        if self.rbr_identity_bn is not None and not self.rbr_identity_bn.built:
            self.rbr_identity_bn.build(input_shape)
        if not self.rbr_reparam.built:
            self.rbr_reparam.build(input_shape)

    def call(self, inputs):
        if self._deploy_mode_internal:
            return self.rbr_reparam(inputs)
        else:
            out_dense = self.rbr_dense_bn(self.rbr_dense_conv(inputs))
            out_1x1 = self.rbr_1x1_bn(self.rbr_1x1_conv(inputs))
            if self.rbr_identity_bn is not None:
                out_identity = self.rbr_identity_bn(inputs)
                return out_dense + out_1x1 + out_identity
            else: return out_dense + out_1x1

    def _fuse_bn_tensor(self, conv_layer, bn_layer): # Not called during inference with deploy=True model
        kernel = conv_layer.kernel; dtype = kernel.dtype; out_channels = kernel.shape[-1]
        gamma = getattr(bn_layer, 'gamma', tf.ones(out_channels, dtype=dtype))
        beta = getattr(bn_layer, 'beta', tf.zeros(out_channels, dtype=dtype))
        running_mean = getattr(bn_layer, 'moving_mean', tf.zeros(out_channels, dtype=dtype))
        running_var = getattr(bn_layer, 'moving_variance', tf.ones(out_channels, dtype=dtype))
        epsilon = bn_layer.epsilon; std = tf.sqrt(running_var + epsilon)
        fused_kernel = kernel * (gamma / std)
        if conv_layer.use_bias: fused_bias = beta + (gamma * (conv_layer.bias - running_mean)) / std
        else: fused_bias = beta - (running_mean * gamma) / std
        return fused_kernel, fused_bias

    def reparameterize(self): # Not called during inference with deploy=True model
        if self._deploy_mode_internal: return
        branches_to_check = [self.rbr_dense_conv, self.rbr_dense_bn, self.rbr_1x1_conv, self.rbr_1x1_bn]
        if self.rbr_identity_bn: branches_to_check.append(self.rbr_identity_bn)
        for branch_layer in branches_to_check:
            if not branch_layer.built: raise Exception(f"ERROR: Branch layer {branch_layer.name} for {self.name} not built.")
        kernel_dense, bias_dense = self._fuse_bn_tensor(self.rbr_dense_conv, self.rbr_dense_bn)
        kernel_1x1_unpadded, bias_1x1 = self._fuse_bn_tensor(self.rbr_1x1_conv, self.rbr_1x1_bn)
        pad_amount = self.config_kernel_size // 2
        kernel_1x1_padded = tf.pad(kernel_1x1_unpadded, [[pad_amount,pad_amount],[pad_amount,pad_amount],[0,0],[0,0]])
        final_kernel = kernel_dense + kernel_1x1_padded; final_bias = bias_dense + bias_1x1
        if self.rbr_identity_bn is not None:
            running_mean_id = self.rbr_identity_bn.moving_mean; running_var_id = self.rbr_identity_bn.moving_variance
            gamma_id = self.rbr_identity_bn.gamma; beta_id = self.rbr_identity_bn.beta
            epsilon_id = self.rbr_identity_bn.epsilon; std_id = tf.sqrt(running_var_id + epsilon_id)
            kernel_id_scaler = gamma_id / std_id
            bias_id_term = beta_id - (running_mean_id * gamma_id) / std_id
            identity_kernel_np = np.zeros((self.config_kernel_size,self.config_kernel_size,self.actual_in_channels,self.config_out_channels),dtype=np.float32)
            for i in range(self.actual_in_channels): identity_kernel_np[pad_amount,pad_amount,i,i] = kernel_id_scaler[i].numpy()
            kernel_id_final = tf.convert_to_tensor(identity_kernel_np, dtype=tf.float32)
            final_kernel += kernel_id_final; final_bias += bias_id_term
        if not self.rbr_reparam.built: raise Exception(f"CRITICAL ERROR: {self.rbr_reparam.name} not built before set_weights.")
        self.rbr_reparam.set_weights([final_kernel, final_bias]); self._deploy_mode_internal = True

    def get_config(self):
        config = super(RepVGGBlock, self).get_config()
        config.update({
            "in_channels": self.config_initial_in_channels, "out_channels": self.config_out_channels,
            "kernel_size": self.config_kernel_size, "stride": self.config_strides_val,
            "groups": self.config_groups, "deploy": self._deploy_mode_internal, "use_se": self.config_use_se
        }); return config
    @classmethod
    def from_config(cls, config): return cls(**config)
# --- End of RepVGGBlock ---

# --- NECALayer Class Definition (Verified Version) ---
class NECALayer(layers.Layer):
    def __init__(self, channels, gamma=2, b=1, **kwargs):
        super(NECALayer, self).__init__(**kwargs)
        self.channels = channels; self.gamma = gamma; self.b = b
        tf_channels = tf.cast(self.channels, tf.float32)
        k_float = (tf.math.log(tf_channels) / tf.math.log(2.0) + self.b) / self.gamma
        k_int = tf.cast(tf.round(k_float), tf.int32)
        if tf.equal(k_int % 2, 0): self.k_scalar_val = k_int + 1
        else: self.k_scalar_val = k_int
        self.k_scalar_val = tf.maximum(1, self.k_scalar_val)
        kernel_size_for_conv1d = (int(self.k_scalar_val.numpy()),)
        self.gap = layers.GlobalAveragePooling2D(keepdims=True)
        self.conv1d = layers.Conv1D(filters=1, kernel_size=kernel_size_for_conv1d, padding='same', use_bias=False, name=self.name + '_eca_conv1d')
        self.sigmoid = layers.Activation('sigmoid')
    def call(self, inputs):
        if self.channels != inputs.shape[-1]: raise ValueError(f"Input channels {inputs.shape[-1]} != layer channels {self.channels} for {self.name}")
        x = self.gap(inputs); x = tf.squeeze(x, axis=[1,2]); x = tf.expand_dims(x, axis=-1)
        x = self.conv1d(x); x = tf.squeeze(x, axis=-1); attention = self.sigmoid(x)
        return inputs * tf.reshape(attention, [-1, 1, 1, self.channels])
    def get_config(self):
        config = super(NECALayer, self).get_config()
        config.update({"channels": self.channels, "gamma": self.gamma, "b": self.b}); return config
    @classmethod
    def from_config(cls, config): return cls(**config)
# --- End of NECALayer ---


# --- Streamlit App Configuration ---
MODEL_FILENAME = 'genera_cic_v1.keras'
LABEL_MAPPING_FILENAME = 'label_mapping.json'
IMG_WIDTH = 299
IMG_HEIGHT = 299

st.set_page_config(page_title="Genera Cloud Classifier", layout="wide")

# --- Load Model and Label Mapping (Cached for performance) ---
@st.cache_resource
def load_keras_model(model_path):
    """Loads the Keras model with custom layer definitions."""
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        st.error(f"Please ensure '{model_path}' is in the same directory as this script, or update the path.")
        return None
    try:
        custom_objects = {'RepVGGBlock': RepVGGBlock, 'NECALayer': NECALayer}
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        st.error(f"Error loading Keras model from '{model_path}': {e}")
        st.error("Make sure the custom layer definitions (RepVGGBlock, NECALayer) are correct and match the saved model.")
        return None

@st.cache_data
def load_label_map(mapping_path):
    """Loads the label mapping from a JSON file."""
    if not os.path.exists(mapping_path):
        st.error(f"Label mapping file not found: {mapping_path}")
        st.error(f"Please ensure '{mapping_path}' is in the same directory as this script, or update the path.")
        return None
    try:
        with open(mapping_path, 'r') as f:
            label_data = json.load(f)
        # Ensure int_to_label keys are integers, as they might be saved as strings in JSON
        int_to_label = {int(k): v for k, v in label_data['int_to_label'].items()}
        return int_to_label
    except Exception as e:
        st.error(f"Error loading label mapping from '{mapping_path}': {e}")
        return None

# Load resources
model = load_keras_model(MODEL_FILENAME)
int_to_label = load_label_map(LABEL_MAPPING_FILENAME)

# --- Image Preprocessing Function ---
def preprocess_for_prediction(image_pil, target_size=(IMG_HEIGHT, IMG_WIDTH)):
    """Prepares a PIL image for model prediction."""
    img = image_pil.convert('RGB') # Ensure 3 channels
    img_resized = img.resize(target_size)
    img_array = np.array(img_resized, dtype=np.float32)
    img_array = img_array / 255.0   # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
    return img_array

# --- Streamlit App UI ---
st.title("☁️ Genera - Cloud Classifier 🌥️")
st.markdown("Upload an image of the sky, and this app will predict the dominant cloud genus.")

# Check if model and labels loaded successfully before proceeding
if model is None or int_to_label is None:
    st.error("Application cannot start due to errors loading model or label mapping. Please check the console/logs for details.")
else:
    uploaded_file = st.file_uploader("Choose a cloud image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            image_pil = Image.open(uploaded_file)
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(image_pil, caption='Uploaded Image.', use_container_width=True)
            
            # Preprocess and predict
            with st.spinner('Analyzing the sky...'):
                processed_image_tensor = preprocess_for_prediction(image_pil)
                predictions = model.predict(processed_image_tensor)
                pred_probabilities = predictions[0] # Get probabilities for the single uploaded image

            with col2:
                st.subheader("🔍 Prediction Results:")
                # Display top N predictions with confidence
                top_n = 5 # Show top 5 predictions
                # Get indices of sorted probabilities (highest first)
                sorted_indices = np.argsort(pred_probabilities)[::-1]

                for i in range(min(top_n, len(pred_probabilities))):
                    class_index = sorted_indices[i]
                    class_name = int_to_label.get(class_index, f"Unknown Class ({class_index})")
                    confidence = pred_probabilities[class_index]
                    st.markdown(f"**{class_name}**: `{confidence*100:.2f}%`")

                # Highlight the top prediction
                top_pred_idx = sorted_indices[0]
                top_class_name = int_to_label.get(top_pred_idx, "Unknown Class")
                top_confidence = pred_probabilities[top_pred_idx]
                st.success(f"**Top Prediction: {top_class_name} ({top_confidence*100:.2f}%)**")

        except Exception as e:
            st.error(f"An error occurred during image processing or prediction: {e}")
            st.error("Please ensure the uploaded file is a valid image format (JPG, JPEG, PNG).")
    else:
        st.info("Please upload an image to classify.")

st.markdown("---")
st.markdown("Developed as part of the Personalized Weather Intelligence project.")