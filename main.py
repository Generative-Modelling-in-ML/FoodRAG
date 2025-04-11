import argparse
import json
import os
from PIL import Image
import traceback

def parse_args():
    parser = argparse.ArgumentParser(description="Personalized Recipe Adaptation Engine")
    parser.add_argument("--recipe_data", type=str, default="recipes_with_nutritional_info.json",
                        help="Path to recipe data JSON file")
    parser.add_argument("--nutrient_data", type=str, default=None,
                        help="Path to nutrient data file (optional)")
    parser.add_argument("--diet_model", type=str, default="diet_food_contrastive_model.pth",
                        help="Path to pre-trained diet model")
    parser.add_argument("--ingredient_graph", type=str, default=None,
                        help="Path to pre-built ingredient graph (optional)")
    parser.add_argument("--recipe_title", type=str, required=True,
                        help="Title or search query for the recipe to adapt")
    parser.add_argument("--diet_constraints", type=str, nargs="+", required=True,
                        help="Dietary constraints to adapt to (e.g., keto vegan)")
    parser.add_argument("--generate_image", action="store_true",
                        help="Generate an image of the adapted recipe")
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate the adaptation performance")
    parser.add_argument("--output", type=str, default="adapted_recipe.json",
                        help="Path to save the adapted recipe")
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Import the recipe adapter
    try:
        from recipe_adapter import RecipeAdaptationEngine
    except ImportError as e:
        print(f"Error importing RecipeAdaptationEngine: {e}")
        print("Make sure all required modules are installed and available in your Python path.")
        return
    
    # Initialize recipe adaptation engine
    try:
        print(f"Initializing Recipe Adaptation Engine with data from {args.recipe_data}...")
        recipe_adapter = RecipeAdaptationEngine(
            recipe_data_path=args.recipe_data,
            nutrient_data_path=args.nutrient_data,
            diet_model_path=args.diet_model,
            ingredient_graph_path=args.ingredient_graph
        )
    except Exception as e:
        print(f"Error initializing Recipe Adaptation Engine: {e}")
        traceback.print_exc()
        return
    
    # Find recipe by title
    try:
        print(f"Searching for recipe: {args.recipe_title}")
        recipe = recipe_adapter.find_recipe_by_title(args.recipe_title)
        
        if not recipe:
            print(f"Recipe '{args.recipe_title}' not found.")
            print("Suggesting recipes that match your dietary constraints instead...")
            suggestions = recipe_adapter.suggest_recipes_for_diet(args.diet_constraints, top_k=3)
            
            if suggestions:
                print("\nHere are some recipe suggestions that match your constraints:")
                for i, suggestion in enumerate(suggestions, 1):
                    print(f"{i}. {suggestion['title']}")
                
                # Use the first suggestion as our recipe
                recipe = suggestions[0]
                print(f"\nUsing '{recipe['title']}' for adaptation.")
            else:
                print("No recipes found matching your constraints.")
                return
    except Exception as e:
        print(f"Error finding recipe: {e}")
        traceback.print_exc()
        return
    
    # Adapt recipe
    try:
        print(f"Adapting recipe to dietary constraints: {', '.join(args.diet_constraints)}")
        adapted_recipe = recipe_adapter.adapt_recipe(
            recipe=recipe,
            diet_constraints=args.diet_constraints,
            generate_image=args.generate_image
        )
        
        # Display adaptation results
        print("\n" + "="*50)
        print(f"Adapted Recipe: {adapted_recipe['title']}")
        print("="*50)
        
        print("\nIngredients:")
        for ing in adapted_recipe['ingredients']:
            if '_original' in ing:
                print(f"- {ing['text']} (substituted for {ing['_original']})")
            else:
                print(f"- {ing['text']}")
        
        if 'adaptation_notes' in adapted_recipe and adapted_recipe['adaptation_notes']:
            print("\nAdaptation Notes:")
            for note in adapted_recipe['adaptation_notes']:
                print(f"- {note}")
        
        # Save adapted recipe
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Handle image separately if present
        if 'image' in adapted_recipe:
            try:
                image = adapted_recipe['image']
                print(f"\nImage object type: {type(image)}")
                
                # Create directory if needed
                output_dir = os.path.dirname(args.output)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                    
                # Construct absolute path for image
                image_path = os.path.splitext(args.output)[0] + '.png'
                absolute_path = os.path.abspath(image_path)
                print(f"Attempting to save image to: {absolute_path}")
                
                # Check if image is a PIL Image
                from PIL import Image as PILImage
                if isinstance(image, PILImage.Image):
                    # Save with explicit format
                    image.save(image_path, format='PNG')
                    print(f"Image saved successfully to: {image_path}")
                else:
                    print(f"WARNING: image is not a PIL Image object but a {type(image)}")
                    # Try to convert if it's another type
                    try:
                        PILImage.fromarray(image).save(image_path)
                        print(f"Successfully converted and saved image to: {image_path}")
                    except Exception as e:
                        print(f"Error converting image: {e}")
                
                # Simple direct test as fallback
                try:
                    test_path = "direct_test_image.png"
                    if isinstance(image, PILImage.Image):
                        image.save(test_path, format='PNG')
                    print(f"Direct save test result: {'Success' if os.path.exists(test_path) else 'Failed'}")
                except Exception as e:
                    print(f"Direct save test error: {e}")
                
                # Replace image with path in JSON
                adapted_recipe['image_path'] = image_path
                del adapted_recipe['image']
                
            except Exception as e:
                import traceback
                print(f"\nERROR saving image: {e}")
                traceback.print_exc()
                print("Continuing without saving image.")
                
                # Still add a reference to show there was an attempt to generate an image
                adapted_recipe['image_generation_attempted'] = True
                if 'image' in adapted_recipe:
                    del adapted_recipe['image']
        
        # Save to JSON
        with open(args.output, 'w') as f:
            json.dump(adapted_recipe, f, indent=2)
        
        print(f"\nAdapted recipe saved to: {args.output}")
        
        # Evaluate if requested
        if args.evaluate:
            try:
                from evaluation import RecipeAdaptationEvaluator
                
                print("\nEvaluating adaptation performance...")
                evaluator = RecipeAdaptationEvaluator(diet_model_path=args.diet_model)
                
                # Evaluate constraint satisfaction
                constraint_metrics = evaluator.evaluate_constraint_satisfaction(
                    original_recipe=recipe,
                    adapted_recipe=adapted_recipe,
                    diet_constraints=args.diet_constraints
                )
                
                print("\nConstraint Satisfaction Metrics:")
                print(f"- Original compliance rate: {constraint_metrics['original_compliance_rate']:.2f}")
                print(f"- Adapted compliance rate: {constraint_metrics['adapted_compliance_rate']:.2f}")
                print(f"- Improvement: {constraint_metrics['compliance_improvement']:.2f}")
                
                if constraint_metrics['fully_compatible']:
                    print("✓ The adapted recipe is fully compatible with your dietary constraints!")
                else:
                    print("! The adapted recipe may not be fully compatible with all constraints.")
                
                # Evaluate CLIP score if image was generated
                if args.generate_image and 'image_path' in adapted_recipe:
                    try:
                        image = Image.open(adapted_recipe['image_path'])
                        clip_metrics = evaluator.evaluate_clip_score(
                            recipe=adapted_recipe,
                            image=image
                        )
                        
                        if 'error' not in clip_metrics:
                            print(f"\nImage-Text Alignment (CLIP) Score: {clip_metrics['clip_score']:.4f}")
                    except Exception as e:
                        print(f"Could not evaluate image-text alignment: {e}")
                
                # Save evaluation results
                eval_output = os.path.splitext(args.output)[0] + '_evaluation.json'
                with open(eval_output, 'w') as f:
                    json.dump(constraint_metrics, f, indent=2)
                
                print(f"Evaluation metrics saved to: {eval_output}")
                
            except Exception as e:
                print(f"Error during evaluation: {e}")
                traceback.print_exc()
    
    except Exception as e:
        print(f"Error adapting recipe: {e}")
        traceback.print_exc()

def example_usage():
    """
    Shows how to use the recipe adaptation engine programmatically
    """
    from recipe_adapter import RecipeAdaptationEngine
    
    # Initialize the engine
    adapter = RecipeAdaptationEngine(
        recipe_data_path="recipes_with_nutritional_info.json",
        diet_model_path="diet_food_contrastive_model.pth"
    )
    
    # Define a sample recipe
    recipe = {
        "title": "Classic Beef Lasagna",
        "ingredients": [
            {"text": "ground beef, 1 pound"},
            {"text": "lasagna noodles, 12 sheets"},
            {"text": "ricotta cheese, 16 ounces"},
            {"text": "mozzarella cheese, 16 ounces, shredded"},
            {"text": "parmesan cheese, 1/2 cup, grated"},
            {"text": "eggs, 2 large"},
            {"text": "tomato sauce, 24 ounces"},
            {"text": "olive oil, 2 tablespoons"},
            {"text": "garlic, 4 cloves, minced"},
            {"text": "onion, 1 medium, chopped"},
            {"text": "Italian seasoning, 1 tablespoon"},
            {"text": "salt, 1 teaspoon"},
            {"text": "black pepper, 1/2 teaspoon"}
        ],
        "instructions": [
            "Preheat oven to 375°F (190°C).",
            "Cook lasagna noodles according to package directions.",
            "In a large skillet, heat olive oil over medium heat. Add onion and garlic, sauté until softened.",
            "Add ground beef, cook until browned. Add tomato sauce and seasonings, simmer for 10 minutes.",
            "In a bowl, mix ricotta cheese, eggs, and 1/4 cup parmesan cheese.",
            "In a 9x13 inch baking dish, layer sauce, noodles, ricotta mixture, and mozzarella.",
            "Repeat layers, ending with sauce and remaining mozzarella and parmesan on top.",
            "Cover with foil and bake for 25 minutes. Remove foil and bake for additional 25 minutes.",
            "Let stand for 15 minutes before serving."
        ]
    }
    
    # Define dietary constraints
    diet_constraints = ["keto", "dairy-free"]
    
    # Adapt the recipe
    adapted_recipe = adapter.adapt_recipe(
        recipe=recipe,
        diet_constraints=diet_constraints,
        generate_image=True
    )
    
    # Print adaptation results
    print(f"Adapted Recipe: {adapted_recipe['title']}")
    print("\nIngredients:")
    for ing in adapted_recipe['ingredients']:
        if '_original' in ing:
            print(f"- {ing['text']} (substituted for {ing['_original']})")
        else:
            print(f"- {ing['text']}")
    
    # Save image if generated
    if 'image' in adapted_recipe:
        try:
            from PIL import Image as PILImage
            image = adapted_recipe['image']
            
            if isinstance(image, PILImage.Image):
                image.save("adapted_lasagna.png", format='PNG')
                print("\nImage saved to: adapted_lasagna.png")
            else:
                print(f"\nWARNING: Unable to save image - not a PIL Image object: {type(image)}")
        except Exception as e:
            print(f"\nError saving image: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    main()