import os
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Paths to the directories
image_dir = "/train_images"
mask_dir = "/train_masks"

# Create directories to store augmented data
augmented_image_dir = "/augmented_images"
augmented_mask_dir = "/augmented_masks"

os.makedirs(augmented_image_dir, exist_ok=True)
os.makedirs(augmented_mask_dir, exist_ok=True)

def random_flip(image, mask):
    """Randomly flip image and mask."""
    if random.random() > 0.5:
        image = ImageOps.mirror(image)
        mask = ImageOps.mirror(mask)
    if random.random() > 0.5:
        image = ImageOps.flip(image)
        mask = ImageOps.flip(mask)
    return image, mask

def random_rotation(image, mask):
    """Randomly rotate image and mask by 90, 180, or 270 degrees."""
    angle = random.choice([0, 90, 180, 270])
    return image.rotate(angle), mask.rotate(angle)

def random_brightness(image):
    """Randomly adjust brightness of the image."""
    enhancer = ImageEnhance.Brightness(image)
    factor = random.uniform(0.7, 1.3)  # Random brightness factor
    return enhancer.enhance(factor)

def augment_and_save(image_path, mask_path, prefix):
    """Perform augmentations and save augmented images and masks."""
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")  # Keep mask as grayscale

    # Apply augmentations
    image, mask = random_flip(image, mask)
    image, mask = random_rotation(image, mask)
    image = random_brightness(image)
    mask = random_brightness(mask)

    # Save augmented image and mask
    image_name = os.path.basename(image_path)
    mask_name = os.path.basename(mask_path)

    image.save(os.path.join(augmented_image_dir, f"{prefix}_{image_name}"))
    mask.save(os.path.join(augmented_mask_dir, f"{prefix}_{mask_name}"))

# Iterate through all images and masks for augmentation
for filename in os.listdir(image_dir):
    if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".tif"):
        image_path = os.path.join(image_dir, filename)
        mask_path = os.path.join(mask_dir, filename)

        # Perform multiple augmentations for each file
        for i in range(5):  # Create 5 augmented versions of each image and mask
            augment_and_save(image_path, mask_path, prefix=f"aug_{i}")

print("Data augmentation completed! Check 'augmented_images_256/' and 'augmented_masks_256/'.")



from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Input
from tensorflow.keras.models import Model

def unet_model(input_size=(128, 128, 3)):
    inputs = Input(input_size)

    # Encoder: Contracting path
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    # Bottleneck
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(c4)

    # Decoder: Expansive path
    u5 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c4)
    u5 = concatenate([u5, c3])
    c5 = Conv2D(256, (3, 3), activation='relu', padding='same')(u5)
    c5 = Conv2D(256, (3, 3), activation='relu', padding='same')(c5)

    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c2])
    c6 = Conv2D(128, (3, 3), activation='relu', padding='same')(u6)
    c6 = Conv2D(128, (3, 3), activation='relu', padding='same')(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c1])
    c7 = Conv2D(64, (3, 3), activation='relu', padding='same')(u7)
    c7 = Conv2D(64, (3, 3), activation='relu', padding='same')(c7)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c7)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model

# Compile the model
model = unet_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())

import os
import numpy as np
import cv2  # For reading images
from sklearn.model_selection import train_test_split

# Load and resize images and masks to (256, 256)
def load_data(image_dir, mask_dir):
    images = []
    masks = []

    for filename in os.listdir(image_dir):
        # Load image and mask
        img = cv2.imread(os.path.join(image_dir, filename))

        # Normalize image and mask
        img = img/255.0
        images.append(img.reshape(128, 128, 3))

    for filename in os.listdir(mask_dir):
        # Load image and mask
        mask = cv2.imread(os.path.join(mask_dir, filename), cv2.IMREAD_GRAYSCALE)

        # Normalize image and mask
        mask = mask/255.0
        masks.append(mask.reshape(128, 128, 1))  # Reshape mask to (128,128, 1)

    return np.array(images), np.array(masks)

# Load data
X, Y = load_data(augmented_image_dir, augmented_mask_dir)

# Split data into training and validation sets (80% train, 20% validation)
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

# Train the model
history = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    batch_size=8,
    epochs=20,
    verbose=1
)


#Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

model.save("/saved_model/U-Net.keras")

