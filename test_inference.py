import numpy as np
import tritonclient.http as httpclient
from PIL import Image, ImageFilter
import sys

# 1. Connect to the Triton Server
try:
    client = httpclient.InferenceServerClient(url="localhost:8000")
    if not client.is_server_live():
        print("Triton server is not live. Check your Docker container.")
        sys.exit(1)
    print("Successfully connected to Triton!")
except Exception as e:
    print(f"Connection failed: {e}")
    sys.exit(1)

# 2. Prepare Input for Generator (Latent Noise)
print("Preparing latent noise...")
latent_size = 128
batch_size = 1

# Generate 4D noise: [Batch, Channels, Height, Width]
latent_noise = np.random.randn(batch_size, latent_size, 1, 1).astype(np.float32)

gen_input = httpclient.InferInput("input__0", [batch_size, latent_size, 1, 1], "FP32")
gen_input.set_data_from_numpy(latent_noise)

# 3. Call Generator
try:
    print("Generating base 32x32 image...")
    gen_response = client.infer(model_name="generator", inputs=[gen_input])
    generated_image = gen_response.as_numpy("output__0") 
except Exception as e:
    print(f"Generator inference failed: {e}")
    sys.exit(1)

# 4. Prepare Input for EBM
try:
    print("Scoring image quality with EBM...")
    ebm_input = httpclient.InferInput("input__0", [batch_size, 3, 32, 32], "FP32")
    ebm_input.set_data_from_numpy(generated_image)

    # 5. Call EBM
    ebm_response = client.infer(model_name="ebm", inputs=[ebm_input])
    energy_score = ebm_response.as_numpy("output__0")
except Exception as e:
    print(f"EBM inference failed: {e}")
    sys.exit(1)

# 6. Post-processing & High-Quality Upscaling
print(f"Upscaling to 4K resolution (3840x2160)...")

# Convert from [-1, 1] to [0, 255]
img_data = generated_image[0]
img_data = ((img_data + 1) * 127.5).clip(0, 255).astype(np.uint8)

# Transpose for PIL: (C, H, W) -> (H, W, C)
img_final = np.transpose(img_data, (1, 2, 0))
img_pil = Image.fromarray(img_final)

# UPSCALE LOGIC
# target 4K dimensions
width_4k = 3840
height_4k = 2160

# We use LANCZOS for the best possible mathematical interpolation
img_upscaled = img_pil.resize((width_4k, height_4k), resample=Image.Resampling.LANCZOS)

# Apply a slight sharpening filter to counteract the "stretch blur"
img_sharpened = img_upscaled.filter(ImageFilter.SHARPEN)

# 7. Final Output
print("-" * 35)
print(f"Base Output Shape: {generated_image.shape}")
print(f"Upscaled Dimensions: {width_4k}x{height_4k}")
print(f"EBM Energy Score: {energy_score.flatten()[0]}")
print("-" * 35)

output_path = "generated_4k.png"
img_sharpened.save(output_path)
print(f"Success! High-resolution image saved as '{output_path}'")