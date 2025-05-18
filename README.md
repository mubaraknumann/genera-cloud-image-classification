# Genera - Cloud Image Classification Model

**Version:** 1.0.0
**Last Updated:** May 18 2025
**Contact:** [numanmubarak@protonmail.com][Github - https://github.com/mubaraknumann]

## Table of Contents
1. [Model Description](#model-description)
2. [Intended Uses & Limitations](#intended-uses--limitations)
3. [How to Use](#how-to-use)
    - [Prerequisites](#prerequisites)
    - [Loading the Model](#loading-the-model)
    - [Making Predictions](#making-predictions)
4. [Training Procedure](#training-procedure)
    - [Dataset: SkyGen](#dataset-skygen)
    - [Architecture: RepVGG with NECA Attention](#architecture-repvgg-with-neca-attention)
    - [Data Preprocessing & Augmentation](#data-preprocessing--augmentation)
    - [Training Details](#training-details)
5. [Evaluation Results](#evaluation-results)
6. [Custom Layers](#custom-layers)
7. [Roadblocks & Solutions During Development](#roadblocks--solutions-during-development)
8. [Future Work](#future-work)
9. [Citation](#citation)
10. [License](#license)
11. [Acknowledgements](#acknowledgements)

## 1. Model Description

**Genera** is a deep learning model designed for the classification of twelve distinct cloud genera from ground-based sky imagery. This model is an implementation of a custom RepVGG-style architecture enhanced with a New Efficient Channel Attention (NECA) mechanism, inspired by recent advancements in computer vision for atmospheric science (specifically drawing ideas from Shi et al., 2024, "Improved RepVGG ground-based cloud image classification with attention convolution").

The model was trained on the **UGCI (Ultimate Ground-level Cloud Image) dataset**, a custom-collected dataset of ground-based cloud images. It aims to provide a robust and efficient solution for automated cloud type identification, which can be a foundational component for personalized weather intelligence systems, meteorological research, and citizen science applications.

**Key Features:**
*   Classifies 12 cloud types: Altocumulus, Altostratus, Cirrocumulus, Cirrostratus, Cirrus, Clear Sky, Contrail, Cumulonimbus, Cumulus, Nimbostratus, Stratocumulus, and Stratus.
*   Utilizes a RepVGG architecture for efficient inference via structural re-parameterization.
*   Incorporates NECA-style channel attention for improved feature extraction.

## 2. Intended Uses & Limitations

**Intended Uses:**
*   Automated classification of cloud types from ground-based, full-sky or partial-sky images.
*   Supporting meteorological observations and local weather analysis.
*   Educational tool for learning about cloud formations.
*   Potential component for citizen science projects related to atmospheric monitoring.
*   Research baseline for further development in image-based atmospheric science.

**Limitations:**
*   The model is trained on the UGCI dataset. Its performance may vary on images with significantly different characteristics (e.g., camera types not represented in UGCI, extreme lighting conditions, heavy obstructions).
*   Currently classifies 12 cloud genera; it does not identify specific cloud species, varieties, or supplementary features beyond the primary genus.
*   Does not predict weather phenomena directly (e.g., rain, snow), only the cloud type which may be associated with such phenomena.
*   The accuracy for minority classes in the UGCI dataset (e.g., Altostratus, Stratus before dataset expansion) was initially lower, though significantly improved with more data. Performance on rare cloud types may still be less robust.

## 3. How to Use

### Prerequisites
*   Python 3.8+
*   TensorFlow 2.10+ (or the version you used)
*   NumPy
*   Pillow (for image manipulation)
*   (Optional for full environment: pandas, scikit-learn, seaborn, matplotlib for data handling and visualization as in the training scripts)

## You can install necessary packages using pip:

     pip install tensorflow numpy Pillow

## Loading the Model

The model is saved in the Keras native format (.keras). You will need to provide the definitions of the custom layers (RepVGGBlock and NECALayer) when loading.

**IMPORTANT: You must have the RepVGGBlock and NECALayer class definitions available in your Python environment before running this.**

**--- CUSTOM LAYER DEFINITIONS ---**


**--- RepVGGBlock Class Definition ---**


       
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
        self.config_use_se = use_se # Placeholder, not used in this version of RepVGGBlock
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
            raise ValueError(f"Input channel mismatch for {self.name}: Expected {self.config_initial_in_channels}, got {self.actual_in_channels}")

        if self.rbr_identity_bn is None and \
           self.actual_in_channels == self.config_out_channels and self.config_strides_val == 1:
            self.rbr_identity_bn = layers.BatchNormalization(name=self.name + '_identity_bn')

        super(RepVGGBlock, self).build(input_shape) # Call super build first

        # Ensure all sub-layers are built
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
        else: # Training mode
            out_dense = self.rbr_dense_bn(self.rbr_dense_conv(inputs))
            out_1x1 = self.rbr_1x1_bn(self.rbr_1x1_conv(inputs))
            if self.rbr_identity_bn is not None:
                out_identity = self.rbr_identity_bn(inputs)
                return out_dense + out_1x1 + out_identity
            else: return out_dense + out_1x1

    def _fuse_bn_tensor(self, conv_layer, bn_layer):
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

    def reparameterize(self):
        if self._deploy_mode_internal: return
        branches_to_check = [self.rbr_dense_conv, self.rbr_dense_bn, self.rbr_1x1_conv, self.rbr_1x1_bn]
        if self.rbr_identity_bn: branches_to_check.append(self.rbr_identity_bn)
        for branch_layer in branches_to_check:
            if not branch_layer.built: # Or len(branch_layer.weights) == 0
                raise Exception(f"ERROR: Branch layer {branch_layer.name} for {self.name} not built. Call model with data first.")

        kernel_dense, bias_dense = self._fuse_bn_tensor(self.rbr_dense_conv, self.rbr_dense_bn)
        kernel_1x1_unpadded, bias_1x1 = self._fuse_bn_tensor(self.rbr_1x1_conv, self.rbr_1x1_bn)
        pad_amount = self.config_kernel_size // 2
        kernel_1x1_padded = tf.pad(kernel_1x1_unpadded, [[pad_amount,pad_amount],[pad_amount,pad_amount],[0,0],[0,0]])
        final_kernel = kernel_dense + kernel_1x1_padded
        final_bias = bias_dense + bias_1x1
        if self.rbr_identity_bn is not None:
            running_mean_id = self.rbr_identity_bn.moving_mean; running_var_id = self.rbr_identity_bn.moving_variance
            gamma_id = self.rbr_identity_bn.gamma; beta_id = self.rbr_identity_bn.beta
            epsilon_id = self.rbr_identity_bn.epsilon; std_id = tf.sqrt(running_var_id + epsilon_id)
            kernel_id_scaler = gamma_id / std_id
            bias_id_term = beta_id - (running_mean_id * gamma_id) / std_id
            identity_kernel_np = np.zeros((self.config_kernel_size, self.config_kernel_size, self.actual_in_channels, self.config_out_channels), dtype=np.float32)
            for i in range(self.actual_in_channels): identity_kernel_np[pad_amount, pad_amount, i, i] = kernel_id_scaler[i].numpy()
            kernel_id_final = tf.convert_to_tensor(identity_kernel_np, dtype=tf.float32)
            final_kernel += kernel_id_final; final_bias += bias_id_term
        if not self.rbr_reparam.built:
            raise Exception(f"CRITICAL ERROR: {self.rbr_reparam.name} of {self.name} not built before set_weights.")
        self.rbr_reparam.set_weights([final_kernel, final_bias])
        self._deploy_mode_internal = True

    def get_config(self):
        config = super(RepVGGBlock, self).get_config()
        config.update({
            "in_channels": self.config_initial_in_channels, "out_channels": self.config_out_channels,
            "kernel_size": self.config_kernel_size, "stride": self.config_strides_val,
            "groups": self.config_groups, "deploy": self._deploy_mode_internal, "use_se": self.config_use_se
        })
        return config
    @classmethod
    def from_config(cls, config): return cls(**config)
**--- End of RepVGGBlock ---**

**--- NECALayer Class Definition ---**

    class NECALayer(layers.Layer):
    def __init__(self, channels, gamma=2, b=1, **kwargs):
        super(NECALayer, self).__init__(**kwargs)
        self.channels = channels
        self.gamma = gamma
        self.b = b
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
        x = self.gap(inputs)
        x = tf.squeeze(x, axis=[1, 2])
        x = tf.expand_dims(x, axis=-1)
        x = self.conv1d(x)
        x = tf.squeeze(x, axis=-1)
        attention = self.sigmoid(x)
        attention_reshaped = tf.reshape(attention, [-1, 1, 1, self.channels])
        return inputs * attention_reshaped

    def get_config(self):
        config = super(NECALayer, self).get_config()
        config.update({"channels": self.channels, "gamma": self.gamma, "b": self.b})
        return config
    @classmethod
    def from_config(cls, config): return cls(**config)
    
**--- End of NECALayer ---**


**--- END OF CUSTOM LAYER DEFINITIONS ---**

    import tensorflow as tf
    from tensorflow import keras
    
    MODEL_FILE = 'path/to/your/repvgg_neca_deploy_final.keras' # Replace with actual path
    LABEL_MAPPING_FILE = 'path/to/your/label_mapping.json' # Replace with actual path
    
    custom_objects = {'RepVGGBlock': RepVGGBlock, 'NECALayer': NECALayer}
    loaded_model = tf.keras.models.load_model(MODEL_FILE, custom_objects=custom_objects, compile=False)
    print("Model loaded successfully!")
    loaded_model.summary() # Optional: to see the loaded architecture

# Load label mapping
    import json
    with open(LABEL_MAPPING_FILE, 'r') as f:
        label_map_data = json.load(f)
    int_to_label = {int(k): v for k, v in label_map_data['int_to_label'].items()}

**Making Predictions**
        
    from PIL import Image    
    import numpy as np
    
    def preprocess_image_for_prediction(image_path_or_pil_image, target_size=(299, 299)):
        if isinstance(image_path_or_pil_image, str):
            img = Image.open(image_path_or_pil_image)
        else: # Assuming PIL image
            img = image_path_or_pil_image
        
    img = img.convert('RGB') # Ensure 3 channels
    img = img.resize(target_size)
    img_array = np.array(img, dtype=np.float32)
    img_array = img_array / 255.0   # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
    return img_array

    # Example prediction:
    image_path = 'path/to/your/cloud_image.jpg' # Replace with your image path
    input_tensor = preprocess_image_for_prediction(image_path)
    predictions = loaded_model.predict(input_tensor)
    predicted_probabilities = predictions[0]
    
    # Get top prediction
    predicted_class_index = np.argmax(predicted_probabilities)
    predicted_class_name = int_to_label.get(predicted_class_index, "Unknown Class")
    confidence = predicted_probabilities[predicted_class_index]
    
    print(f"Predicted Cloud Type: {predicted_class_name}")
    print(f"Confidence: {confidence*100:.2f}%")
    
    # Display all class probabilities (optional)
     for i, prob in enumerate(predicted_probabilities):
         class_name = int_to_label.get(i, f"Class_{i}")
         print(f"- {class_name}: {prob*100:.2f}%")

## 4. Training Procedure
Dataset: UGCI

The model was trained on UGCI, a custom dataset of ground-based cloud images collected by [https://github.com/mubaraknumann].

Total Images (after expansion): ~32,742 images (Train: 22,918, Val: 4,912, Test: 4,912).

Classes (12): Altocumulus, Altostratus, Cirrocumulus, Cirrostratus, Cirrus, Clear Sky, Contrail, Cumulonimbus, Cumulus, Nimbostratus, Stratocumulus, Stratus.

**Image Characteristics:** [Briefly describe: e.g., "Images were captured using various mobile and stationary cameras, representing diverse lighting conditions and geographical locations, primarily focusing on full-sky or wide-angle views."]

**Labeling Process:** [Briefly describe: e.g., "Images were manually labeled by meteorological enthusiasts/experts."]

**Data Splitting:** Stratified 70% training, 15% validation, 15% testing.

**Architecture:** RepVGG with NECA Attention

The model architecture is based on RepVGG, which features structural re-parameterization (multi-branch for training, single fused 3x3 convolution per block for inference). Each RepVGG block is followed by a New Efficient Channel Attention (NECA) module and a ReLU activation.

**RepVGG Configuration:**

Stages: 4

Blocks per stage: [1, 2, 4, 1]

Channels per stage: [64, 128, 256, 512]

NECA Parameters: gamma=2, b=1 for adaptive kernel size calculation in the 1D convolution.

Data Preprocessing & Augmentation

Input Size: Images were resized to 299x299 pixels.

Normalization: Pixel values were scaled to the `` range.

Training Augmentations (Stronger):

Random Horizontal Flips

Random Rotations (up to ~30 degrees, factor 0.1)

Random Zoom (up to 10%)

Random Translation (up to 5%)

Random Brightness adjustments (factor 0.3)

Random Contrast adjustments (factor 0.3)

**Training Details**

Framework: TensorFlow/Keras

Optimizer: AdamW (learning_rate=1e-4, weight_decay=5e-5)

Loss Function: Sparse Categorical Crossentropy

Class Imbalance: Addressed using balanced class weights during training.

Callbacks:

ModelCheckpoint (saving best model based on val_accuracy)

EarlyStopping (monitoring val_loss, patience 20, restore_best_weights=True)

ReduceLROnPlateau (monitoring val_loss, patience 10)

Epochs: Trained for up to 150 epochs (EarlyStopping intervened). The best model was restored from Epoch 171 of the final run (adjust if different).

Batch Size: 32

## 5. Evaluation Results

The final RepVGG+NECA deploy model (loaded from file) achieved the following on the UGCI test set:

Overall Test Accuracy: ~90.15%

Overall Test Loss: ~0.3968

Macro Average F1-score: ~0.88

Weighted Average F1-score: ~0.90

Per-Class Performance (F1-score from final run):

Altocumulus: 0.94

Altostratus: 0.89

Cirrocumulus: 0.85

Cirrostratus: 0.82

Cirrus: 0.91

Clear Sky: 1.00

Contrail: 0.68

Cumulonimbus: 0.94

Cumulus: 0.95

Nimbostratus: 0.85

Stratocumulus: 0.88

Stratus: 0.87

**Training/Validation Graph**

![image/png](https://cdn-uploads.huggingface.co/production/uploads/68299aa8c18ecd32686e7e9e/f-1txKQ1IWWWgzLB_csJQ.png)

**Confusion Matrix**

![image/png](https://cdn-uploads.huggingface.co/production/uploads/68299aa8c18ecd32686e7e9e/BvnOq4zk6BQgyoh92BFml.png)

**Per Class F1 Scores**

![image/png](https://cdn-uploads.huggingface.co/production/uploads/68299aa8c18ecd32686e7e9e/FrX6m5V5Cpcl622EUVM9r.png)


## 6. Custom Layers

This model utilizes two custom Keras layers. Their Python class definitions are required to load and use the model.

RepVGGBlock: Implements the re-parameterizable block.

NECALayer: Implements the New Efficient Channel Attention mechanism.

(You would typically provide the code for these layers in a separate .py file or directly in notebooks/scripts that use the model).

## 7. Roadblocks & Solutions During Development

The development process involved several key challenges, primarily related to the custom RepVGGBlock:

Initial Training Instability: Early RepVGG training attempts suffered from exploding validation losses, addressed by significantly reducing the learning rate and using AdamW with weight decay.

Reparameterization Errors: Numerous ValueError and AttributeError issues occurred when trying to convert the trained multi-branch RepVGG blocks to their single-branch inference form. This was due to Keras's layer build lifecycle and how it handles sub-layers that are not in the active computation graph during training mode.

Solution: The RepVGGBlock class was iteratively refined. The final robust solution involved:

Defining all potential sub-layers (training branches AND the deploy-mode fused Conv2D layer) in __init__.

Implementing a build(self, input_shape) method in RepVGGBlock that explicitly ensures all these sub-layers (including the deploy-mode rbr_reparam Conv2D) are built when the RepVGGBlock itself is built by Keras (e.g., when data first flows through the model).

The reparameterize() method then calculates fused weights and sets them onto the already existing and built rbr_reparam sub-layer.

Save/Load Consistency: Ensuring that the saved deploy-mode model correctly loaded and performed identically to the in-memory reparameterized version also required careful management of the custom layer's config and state. The final approach proved successful.

Data Augmentation Issues: Initial problems with augmented images appearing black were traced to RandomBrightness layer defaults and fixed by specifying value_range=(0.0, 1.0).

A "quick test" script was developed to rapidly iterate on and debug the RepVGGBlock's reparameterization and save/load mechanism without requiring full model training cycles.

## 8. Future Work

Further hyperparameter tuning and exploration of more advanced data augmentation.

Experimentation with different RepVGG architectural variants (depth/width).

Continued refinement of minority class performance (contrail).

## 9. Citation

If you use this model or code in your research, please consider citing this repository (and any associated paper, if applicable).

Mohammed Numan Mubarak. (2025). Genera - Cloud Image Classification Model. Retrieved from [huggingface.co/mubaraknumann/genera-cloud-image-classification]

## 10. License

This project, including the model weights and source code, is licensed under the MIT License. See the LICENSE file for more details.

## 11. Acknowledgements

This work was inspired by the methodologies presented in "Improved RepVGG ground-based cloud image classification with attention convolution" by Shi et al. (2024).

---
tags:
- cloud
- weather
- cloud genera
- cloud types
- wmo
- cloud image classification
---
