import tensorflow as tf
import os

MODEL_PATH = "saved_models/original_model.h5"

# Check file exists
if not os.path.exists(MODEL_PATH):
    print("Model file NOT FOUND:", MODEL_PATH)
    exit()

# Load old model safely
model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

# Save ONLY weights
model.save_weights("saved_models/nids_weights.h5")

print("Weights file created successfully")
