import torch
import numpy as np
import time
import psutil
import json
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

class RecipeAdaptationEvaluator:
    """
    Evaluates the performance of the recipe adaptation system
    """
    def __init__(self, diet_model_path=None):
        """
        Initialize the evaluator
        
        Args:
            diet_model_path: Path to the dietary constraint model
        """
        # Initialize CLIP for image-text alignment evaluation
        try:
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        except Exception as e:
            print(f"Warning: Could not initialize CLIP model: {e}")
            self.clip_model = None
            self.clip_processor = None
        
        # Initialize diet model for constraint satisfaction evaluation
        if diet_model_path:
            from diet_model import DietaryConstraintEmbedding
            self.diet_model = DietaryConstraintEmbedding(model_path=diet_model_path)
        else:
            self.diet_model = None
    
    def evaluate_constraint_satisfaction(self, original_recipe, adapted_recipe, diet_constraints):
        """
        Evaluate how well the adapted recipe satisfies dietary constraints
        
        Args:
            original_recipe: Original recipe before adaptation
            adapted_recipe: Recipe after adaptation
            diet_constraints: List of dietary constraints
            
        Returns:
            Dictionary with constraint satisfaction metrics
        """
        if not self.diet_model:
            return {"error": "Diet model not initialized"}
        
        # Standardize diet constraint names
        standardized_constraints = []
        for constraint in diet_constraints:
            if not constraint.lower().endswith("diet"):
                constraint = f"{constraint} diet"
            standardized_constraints.append(constraint.lower())
        
        # Check original recipe compatibility
        original_compatible_count = 0
        original_ingredient_count = len(original_recipe['ingredients'])
        original_compatibility_scores = []
        
        for ingredient_obj in original_recipe['ingredients']:
            ingredient_name = ingredient_obj['text']
            is_compatible = True
            
            for diet in standardized_constraints:
                score, compatible = self.diet_model.check_food_compatibility(diet, ingredient_name)
                if not compatible:
                    is_compatible = False
                original_compatibility_scores.append(score)
            
            if is_compatible:
                original_compatible_count += 1
        
        # Check adapted recipe compatibility
        adapted_compatible_count = 0
        adapted_ingredient_count = len(adapted_recipe['ingredients'])
        adapted_compatibility_scores = []
        
        for ingredient_obj in adapted_recipe['ingredients']:
            ingredient_name = ingredient_obj['text']
            is_compatible = True
            
            for diet in standardized_constraints:
                score, compatible = self.diet_model.check_food_compatibility(diet, ingredient_name)
                if not compatible:
                    is_compatible = False
                adapted_compatibility_scores.append(score)
            
            if is_compatible:
                adapted_compatible_count += 1
        
        # Calculate metrics
        original_compliance_rate = original_compatible_count / original_ingredient_count if original_ingredient_count > 0 else 0
        adapted_compliance_rate = adapted_compatible_count / adapted_ingredient_count if adapted_ingredient_count > 0 else 0
        
        # Calculate average compatibility scores
        original_avg_score = sum(original_compatibility_scores) / len(original_compatibility_scores) if original_compatibility_scores else 0
        adapted_avg_score = sum(adapted_compatibility_scores) / len(adapted_compatibility_scores) if adapted_compatibility_scores else 0
        
        # Calculate improvement
        compliance_improvement = adapted_compliance_rate - original_compliance_rate
        score_improvement = adapted_avg_score - original_avg_score
        
        return {
            "original_compliance_rate": original_compliance_rate,
            "adapted_compliance_rate": adapted_compliance_rate,
            "compliance_improvement": compliance_improvement,
            "original_compatibility_score": original_avg_score,
            "adapted_compatibility_score": adapted_avg_score,
            "score_improvement": score_improvement,
            "fully_compatible": adapted_compatible_count == adapted_ingredient_count
        }
    
    def evaluate_clip_score(self, recipe, image):
        """
        Evaluate alignment between recipe and generated image using CLIP
        
        Args:
            recipe: Recipe dictionary
            image: Generated image
            
        Returns:
            CLIP score (higher means better alignment)
        """
        if not self.clip_model or not self.clip_processor:
            return {"error": "CLIP model not initialized"}
        
        # Create text description from recipe
        ingredients_text = ", ".join([ing['text'] for ing in recipe['ingredients'][:5]])
        text = f"{recipe['title']}: {ingredients_text}"
        
        # Prepare inputs for CLIP
        inputs = self.clip_processor(
            text=[text],
            images=image,
            return_tensors="pt",
            padding=True
        )
        
        # Calculate similarity score
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            logits_per_image = outputs.logits_per_image
            clip_score = logits_per_image.item()
        
        return {
            "clip_score": clip_score,
            "text_prompt": text
        }
    
    def evaluate_performance_metrics(self, func, *args, **kwargs):
        """
        Measure performance metrics (time, memory) of a function
        
        Args:
            func: Function to measure
            *args, **kwargs: Arguments to pass to the function
            
        Returns:
            Performance metrics
        """
        # Measure memory before
        process = psutil.Process()
        memory_before = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Measure time
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        # Measure memory after
        memory_after = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Calculate metrics
        execution_time = end_time - start_time
        memory_usage = memory_after - memory_before
        
        return {
            "execution_time_seconds": execution_time,
            "memory_usage_mb": memory_usage,
            "result": result
        }
    
    def evaluate_adaptation(self, recipe_adapter, recipe, diet_constraints, generate_image=False):
        """
        Comprehensive evaluation of recipe adaptation
        
        Args:
            recipe_adapter: Recipe adaptation engine
            recipe: Original recipe to adapt
            diet_constraints: Dietary constraints
            generate_image: Whether to generate an image
            
        Returns:
            Evaluation results
        """
        # Measure performance
        perf_metrics = self.evaluate_performance_metrics(
            recipe_adapter.adapt_recipe,
            recipe=recipe,
            diet_constraints=diet_constraints,
            generate_image=generate_image
        )
        
        adapted_recipe = perf_metrics["result"]
        
        # Evaluate constraint satisfaction
        constraint_metrics = self.evaluate_constraint_satisfaction(
            original_recipe=recipe,
            adapted_recipe=adapted_recipe,
            diet_constraints=diet_constraints
        )
        
        # Evaluate image if generated
        clip_metrics = {}
        if generate_image and "image" in adapted_recipe:
            clip_metrics = self.evaluate_clip_score(
                recipe=adapted_recipe,
                image=adapted_recipe["image"]
            )
        
        # Combine all metrics
        evaluation_results = {
            "performance_metrics": {
                "execution_time_seconds": perf_metrics["execution_time_seconds"],
                "memory_usage_mb": perf_metrics["memory_usage_mb"]
            },
            "constraint_satisfaction": constraint_metrics,
            "image_text_alignment": clip_metrics,
            "adapted_recipe": {
                "title": adapted_recipe["title"],
                "ingredients_count": len(adapted_recipe["ingredients"]),
                "adaptation_notes": adapted_recipe.get("adaptation_notes", [])
            }
        }
        
        return evaluation_results
    
    def evaluate_foodgpt_comparison(self, our_adapted_recipe, foodgpt_recipe, diet_constraints):
        """
        Compare our adaptation with FoodGPT baseline
        
        Args:
            our_adapted_recipe: Recipe adapted by our system
            foodgpt_recipe: Recipe adapted by FoodGPT
            diet_constraints: Dietary constraints
            
        Returns:
            Comparison metrics
        """
        # Evaluate constraint satisfaction for both
        our_constraints = self.evaluate_constraint_satisfaction(
            original_recipe=our_adapted_recipe,  # Using as original since we need structure only
            adapted_recipe=our_adapted_recipe,
            diet_constraints=diet_constraints
        )
        
        foodgpt_constraints = self.evaluate_constraint_satisfaction(
            original_recipe=foodgpt_recipe,  # Using as original since we need structure only
            adapted_recipe=foodgpt_recipe,
            diet_constraints=diet_constraints
        )
        
        # Compare CLIP scores if images are available
        our_clip_score = None
        foodgpt_clip_score = None
        
        if "image" in our_adapted_recipe:
            our_clip_results = self.evaluate_clip_score(
                recipe=our_adapted_recipe,
                image=our_adapted_recipe["image"]
            )
            our_clip_score = our_clip_results.get("clip_score")
        
        if "image" in foodgpt_recipe:
            foodgpt_clip_results = self.evaluate_clip_score(
                recipe=foodgpt_recipe,
                image=foodgpt_recipe["image"]
            )
            foodgpt_clip_score = foodgpt_clip_results.get("clip_score")
        
        # Create comparison metrics
        comparison = {
            "constraint_satisfaction": {
                "our_system": our_constraints.get("adapted_compliance_rate", 0),
                "foodgpt": foodgpt_constraints.get("adapted_compliance_rate", 0),
                "difference": our_constraints.get("adapted_compliance_rate", 0) - foodgpt_constraints.get("adapted_compliance_rate", 0)
            },
            "compatibility_score": {
                "our_system": our_constraints.get("adapted_compatibility_score", 0),
                "foodgpt": foodgpt_constraints.get("adapted_compatibility_score", 0),
                "difference": our_constraints.get("adapted_compatibility_score", 0) - foodgpt_constraints.get("adapted_compatibility_score", 0)
            }
        }
        
        if our_clip_score and foodgpt_clip_score:
            comparison["clip_score"] = {
                "our_system": our_clip_score,
                "foodgpt": foodgpt_clip_score,
                "difference": our_clip_score - foodgpt_clip_score
            }
        
        return comparison