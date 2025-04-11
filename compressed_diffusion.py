import torch
from PIL import Image, ImageDraw
import traceback
import os

class CompressedDiffusionModel:
    """
    Simplified stable diffusion model with better error handling and lower memory footprint
    """
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5", device=None):
        # Use a smaller model by default
        self.model_id = model_id
        
        # Determine the best available device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
                print(f"Using CUDA for image generation (GPU: {torch.cuda.get_device_name(0)})")
            else:
                self.device = "cpu"
                print("Using CPU for image generation (will be slow)")
        else:
            self.device = device
            
        self.pipe = None
        self.initialized = False
    
    def compress_model(self):
        """Load a smaller model with better memory handling"""
        try:
            print(f"Loading diffusion model {self.model_id}...")
            
            # Import here to avoid loading dependencies unless needed
            try:
                from diffusers import StableDiffusionPipeline
                print("Successfully imported diffusers")
            except ImportError as e:
                print(f"Error importing diffusers: {e}")
                print("Please install with: pip install diffusers")
                return False
                
            # Choose appropriate settings based on device
            if self.device == "cuda":
                print("Setting up with CUDA optimizations...")
                torch_dtype = torch.float16
            else:
                torch_dtype = torch.float32
            
            # Try smaller model first if on CPU to avoid memory issues
            model_id = "CompVis/stable-diffusion-v1-2" if self.device == "cpu" else self.model_id
                
            try:
                # Basic diffusion pipeline without safety checker to reduce memory
                pipe = StableDiffusionPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch_dtype,
                    safety_checker=None
                )
                
                # Enable memory efficient attention
                pipe.enable_attention_slicing(slice_size="auto")
                
                # Move to device
                pipe = pipe.to(self.device)
                self.pipe = pipe
                self.initialized = True
                print("Model loaded successfully")
                return True
                
            except Exception as e:
                print(f"Error loading model: {e}")
                traceback.print_exc()
                return False
            
        except Exception as e:
            print(f"Error initializing diffusion model: {e}")
            traceback.print_exc()
            self.initialized = False
            return False
    
    def generate_food_image(self, prompt, negative_prompt=None, guidance_scale=7.5, num_inference_steps=25):
        """
        Generate a food image based on the prompt
        
        Args:
            prompt: Text description of the food
            negative_prompt: Things to avoid in generation
            guidance_scale: Classifier-free guidance scale
            num_inference_steps: Number of denoising steps
            
        Returns:
            Generated image as a PIL Image object
        """
        if not self.initialized:
            success = self.compress_model()
            if not success:
                return self._create_placeholder_image(f"Failed to initialize model\n{prompt}")
        
        # Enhance prompt for food image generation
        enhanced_prompt = f"professional food photography of {prompt}, high resolution, mouth-watering, gourmet, beautifully plated, soft lighting"
        print(f"Using prompt: {enhanced_prompt[:100]}...")
        
        # Default negative prompt for food images
        if negative_prompt is None:
            negative_prompt = "blurry, low quality, distorted, deformed, disfigured, watermark, text, logo, poorly lit, bad composition"
        
        # Reduce inference steps on CPU for speed
        if self.device == "cpu":
            num_inference_steps = min(15, num_inference_steps)  # Even fewer steps for CPU
            print(f"Using {num_inference_steps} inference steps on CPU")
        
        # Generate image
        try:
            print("Generating image...")
            
            # First try to empty CUDA cache if on GPU
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            # Generate the image with appropriate settings
            outputs = self.pipe(
                prompt=enhanced_prompt,
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps
            )
            
            # Get the image
            if hasattr(outputs, "images") and len(outputs.images) > 0:
                print("Image generated successfully")
                output_image = outputs.images[0]
                
                # Ensure it's a valid PIL Image
                if isinstance(output_image, Image.Image):
                    # Save a copy directly to disk as a test
                    try:
                        test_path = "generated_food_direct_test.png"
                        output_image.save(test_path)
                        print(f"Direct test image saved to {test_path}")
                    except Exception as e:
                        print(f"Warning: Could not save test image: {e}")
                    
                    # Return a copy of the image
                    return output_image.copy()
                else:
                    print(f"WARNING: output is not a PIL Image but {type(output_image)}")
                    return self._create_placeholder_image(f"Invalid image type\n{prompt}")
            else:
                print("No image in output")
                return self._create_placeholder_image(f"No image in output\n{prompt}")
                
        except Exception as e:
            print(f"Error generating image: {e}")
            traceback.print_exc()
            return self._create_placeholder_image(f"Error: {str(e)[:50]}\n{prompt}")
    
    def _create_placeholder_image(self, text):
        """Create a placeholder image with error message"""
        img = Image.new('RGB', (512, 512), color=(255, 255, 255))
        d = ImageDraw.Draw(img)
        d.text((10, 10), text, fill=(0, 0, 0))
        print(f"Created placeholder image with text: {text[:50]}...")
        return img


# Test function (can be called directly)
def test_model(prompt="delicious chocolate cake", output_path="food_image_test.png"):
    model = CompressedDiffusionModel()
    success = model.compress_model()
    
    if success:
        image = model.generate_food_image(prompt)
        if image:
            image.save(output_path)
            print(f"Test image saved to {output_path}")
            return True
    
    return False


if __name__ == "__main__":
    # Run test if this file is executed directly
    test_model()