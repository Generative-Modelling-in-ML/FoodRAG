"""
Script to modify recipe_adapter.py file to use the standalone diffusion model
This avoids circular imports and dependency issues
"""
import os

def fix_initialize_diffusion_model():
    """Replace the _initialize_diffusion_model method in recipe_adapter.py"""
    file_path = 'recipe_adapter.py'
    
    # Create backup if not already exists
    backup_path = 'recipe_adapter.py.bak'
    if not os.path.exists(backup_path):
        import shutil
        shutil.copy2(file_path, backup_path)
        print(f"Created backup at {backup_path}")
    
    # Read the file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find the method definition
    import re
    method_pattern = r'def _initialize_diffusion_model\(self\):(.*?)(?=\n    def |\n\n)'
    
    # New method implementation that avoids circular imports
    new_method = """
    def _initialize_diffusion_model(self):
        \"\"\"Initialize the diffusion model on demand to save memory\"\"\"
        if self.diffusion_model is None:
            try:
                # Import the model directly to avoid circular imports
                from compressed_diffusion import CompressedDiffusionModel
                
                print("Initializing diffusion model...")
                self.diffusion_model = CompressedDiffusionModel()
                success = self.diffusion_model.compress_model()
                
                if not success:
                    print("WARNING: Diffusion model initialization failed")
                    self.diffusion_model = None
            except Exception as e:
                print(f"ERROR initializing diffusion model: {e}")
                import traceback
                traceback.print_exc()
                print("Continuing without image generation capability")
                self.diffusion_model = None
    """
    
    # Replace the method
    if re.search(method_pattern, content, re.DOTALL):
        new_content = re.sub(method_pattern, f"def _initialize_diffusion_model(self):{new_method}", content, flags=re.DOTALL)
        
        # Write back to the file
        with open(file_path, 'w') as f:
            f.write(new_content)
        
        print(f"Successfully fixed _initialize_diffusion_model method in {file_path}")
        return True
    else:
        print(f"Could not find _initialize_diffusion_model method in {file_path}")
        return False

if __name__ == "__main__":
    fix_initialize_diffusion_model()