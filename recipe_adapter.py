import json
import numpy as np
from collections import defaultdict
import copy
import traceback

# Import components
from recipe_graph import build_ingredient_graph, find_ingredient_substitutions
from rag_retriever import RecipeKnowledgeRetriever

# Uses DietaryConstraintEmbedding from diet_model.py
# Uses CompressedDiffusionModel from compressed_diffusion.py

class RecipeAdaptationEngine:
    """
    Main engine for adapting recipes based on dietary constraints
    """
    def __init__(
        self, 
        recipe_data_path, 
        nutrient_data_path=None,
        diet_model_path="diet_food_contrastive_model.pth",
        ingredient_graph_path=None
    ):
        """
        Initialize the recipe adaptation engine
        
        Args:
            recipe_data_path: Path to recipe data JSON
            nutrient_data_path: Path to nutrient data (optional)
            diet_model_path: Path to pre-trained diet model
            ingredient_graph_path: Path to pre-built ingredient graph (or None to build on init)
        """
        print("Initializing Recipe Adaptation Engine...")
        
        # Load recipe data
        with open(recipe_data_path, 'r') as f:
            self.recipes = json.load(f)
        
        # Initialize knowledge retriever
        self.retriever = RecipeKnowledgeRetriever(
            recipe_data_path=recipe_data_path,
            nutrient_data_path=nutrient_data_path
        )
        
        # Build or load ingredient graph
        if ingredient_graph_path:
            print(f"Loading ingredient graph from {ingredient_graph_path}")
            # Load pre-built graph
            import pickle
            with open(ingredient_graph_path, 'rb') as f:
                self.ingredient_graph = pickle.load(f)
        else:
            print("Building ingredient graph (this may take a while)...")
            # Build graph from recipe data
            self.ingredient_graph, _, _ = build_ingredient_graph(
                recipe_data_path,
                min_co_occurrence=2,
                alpha=0.6  # Weight between co-occurrence and nutritional similarity
            )
        
        # Initialize dietary constraint model
        from diet_model import DietaryConstraintEmbedding
        print(f"Loading dietary constraint model from {diet_model_path}")
        self.diet_model = DietaryConstraintEmbedding(model_path=diet_model_path)
        
        # Initialize image generation model (only when needed to save memory)
        self.diffusion_model = None
        
        print("Recipe Adaptation Engine initialized")
    
    def _initialize_diffusion_model(self):
        """Initialize the diffusion model on demand to save memory"""
        if self.diffusion_model is None:
            try:
                # First try to import the fixed version if it exists
                try:
                    from fixed_diffusion import CompressedDiffusionModel
                    print("Using fixed diffusion model implementation")
                except ImportError:
                    # Fall back to original implementation
                    from compressed_diffusion import CompressedDiffusionModel
                    print("Using original diffusion model implementation")
                    
                print("Initializing compressed diffusion model (this may take a while)...")
                self.diffusion_model = CompressedDiffusionModel(model_id="runwayml/stable-diffusion-v1-5")
                success = self.diffusion_model.compress_model()
                
                if not success and hasattr(self.diffusion_model, 'initialized') and not self.diffusion_model.initialized:
                    print("WARNING: Diffusion model initialization may have failed")
            except Exception as e:
                print(f"ERROR initializing diffusion model: {e}")
                traceback.print_exc()
                print("Continuing without image generation capability")
                self.diffusion_model = None
    
    def adapt_recipe(self, recipe, diet_constraints, generate_image=False):
        """
        Adapt a recipe to meet dietary constraints
        
        Args:
            recipe: Recipe to adapt (dict with title, ingredients, instructions)
            diet_constraints: List of dietary constraints (e.g., ["keto", "dairy-free"])
            generate_image: Whether to generate an image of the adapted recipe
            
        Returns:
            Adapted recipe
        """
        print(f"Adapting recipe '{recipe['title']}' to {diet_constraints}")
        
        # Make a deep copy of the recipe to avoid modifying the original
        adapted_recipe = copy.deepcopy(recipe)
        
        # Standardize diet constraint names
        standardized_constraints = []
        for constraint in diet_constraints:
            if not constraint.lower().endswith("diet"):
                constraint = f"{constraint} diet"
            standardized_constraints.append(constraint.lower())
        
        # Retrieve knowledge for adaptation
        print("Retrieving adaptation knowledge...")
        adaptation_knowledge = self.retriever.retrieve_adaptation_knowledge(recipe, diet_constraints)
        
        # Identify incompatible ingredients
        print("Identifying incompatible ingredients...")
        incompatible_ingredients = []
        compatibility_scores = {}
        
        for ingredient_obj in recipe['ingredients']:
            ingredient_name = ingredient_obj['text']
            
            # Check compatibility with all diet constraints
            is_compatible = True
            avg_score = 0
            
            for diet in standardized_constraints:
                score, compatible = self.diet_model.check_food_compatibility(diet, ingredient_name)
                avg_score += score
                if not compatible:
                    is_compatible = False
            
            avg_score /= len(standardized_constraints)
            compatibility_scores[ingredient_name] = avg_score
            
            if not is_compatible:
                incompatible_ingredients.append(ingredient_name)
        
        print(f"Found {len(incompatible_ingredients)} incompatible ingredients")
        
        # Find suitable substitutions for incompatible ingredients
        substitutions = {}
        
        for ingredient in incompatible_ingredients:
            print(f"Finding substitutes for '{ingredient}'...")
            
            # Try graph-based substitution first
            graph_substitutes = []
            if ingredient in self.ingredient_graph:
                graph_substitutes = find_ingredient_substitutions(
                    self.ingredient_graph, 
                    ingredient,
                    top_n=10
                )
            
            # Filter substitutes by diet compatibility
            compatible_substitutes = []
            for sub, score in graph_substitutes:
                is_compatible = True
                for diet in standardized_constraints:
                    _, compatible = self.diet_model.check_food_compatibility(diet, sub)
                    if not compatible:
                        is_compatible = False
                        break
                
                if is_compatible:
                    compatible_substitutes.append((sub, score))
            
            # If no compatible substitutes found from graph, use knowledge retriever
            if not compatible_substitutes:
                # Get ingredient knowledge
                ing_knowledge = adaptation_knowledge['ingredient_substitutions'].get(ingredient, {})
                
                # Try common pairings as potential substitutes
                common_pairings = ing_knowledge.get('common_pairings', [])
                
                for pairing in common_pairings:
                    is_compatible = True
                    for diet in standardized_constraints:
                        _, compatible = self.diet_model.check_food_compatibility(diet, pairing)
                        if not compatible:
                            is_compatible = False
                            break
                    
                    if is_compatible:
                        # Add to substitutes with a default score
                        compatible_substitutes.append((pairing, 0.5))
            
            # If still no compatible substitutes, use general diet-friendly ingredients
            if not compatible_substitutes:
                for advice in adaptation_knowledge['general_advice']:
                    for common_ing in advice['common_ingredients']:
                        is_compatible = True
                        for diet in standardized_constraints:
                            _, compatible = self.diet_model.check_food_compatibility(diet, common_ing)
                            if not compatible:
                                is_compatible = False
                                break
                        
                        if is_compatible:
                            compatible_substitutes.append((common_ing, 0.3))
            
            # Remove duplicates and sort by score
            seen = set()
            unique_substitutes = []
            for sub, score in compatible_substitutes:
                if sub not in seen:
                    seen.add(sub)
                    unique_substitutes.append((sub, score))
            
            unique_substitutes.sort(key=lambda x: x[1], reverse=True)
            
            # Store top substitutions
            if unique_substitutes:
                substitutions[ingredient] = unique_substitutes[:3]  # Top 3 substitutes
            else:
                # If no substitutes found, recommend omitting
                substitutions[ingredient] = [("omit", 1.0)]
        
        # Apply substitutions to recipe
        for i, ingredient_obj in enumerate(adapted_recipe['ingredients']):
            ingredient_name = ingredient_obj['text']
            
            if ingredient_name in substitutions and substitutions[ingredient_name]:
                # Replace with top substitute
                top_substitute, _ = substitutions[ingredient_name][0]
                
                if top_substitute == "omit":
                    # Mark for removal
                    adapted_recipe['ingredients'][i]['text'] = f"[REMOVED] {ingredient_name}"
                    adapted_recipe['ingredients'][i]['_original'] = ingredient_name
                else:
                    # Replace with substitute
                    adapted_recipe['ingredients'][i]['text'] = top_substitute
                    adapted_recipe['ingredients'][i]['_original'] = ingredient_name
                    adapted_recipe['ingredients'][i]['_substitute'] = True
        
        # Remove ingredients marked for removal
        adapted_recipe['ingredients'] = [
            ing for ing in adapted_recipe['ingredients']
            if not ing['text'].startswith('[REMOVED]')
        ]
        
        # Update recipe title
        diet_suffix = " and ".join(diet_constraints)
        adapted_recipe['title'] = f"{recipe['title']} ({diet_suffix})"
        
        # Add adaptation notes
        adapted_recipe['adaptation_notes'] = []
        for original, substitutes in substitutions.items():
            if substitutes[0][0] == "omit":
                adapted_recipe['adaptation_notes'].append(f"Removed {original} to comply with dietary restrictions")
            else:
                adapted_recipe['adaptation_notes'].append(
                    f"Substituted {original} with {substitutes[0][0]} to comply with dietary restrictions"
                )
        
        # Generate image if requested
        if generate_image:
            print("Generating image for adapted recipe...")
            try:
                self._initialize_diffusion_model()
                
                if self.diffusion_model is not None:
                    # Create prompt from recipe title and main ingredients
                    main_ingredients = [ing['text'] for ing in adapted_recipe['ingredients'][:5]]
                    prompt = f"{adapted_recipe['title']}: {', '.join(main_ingredients)}"
                    print(f"Using prompt: {prompt}")
                    
                    image = self.diffusion_model.generate_food_image(prompt)
                    if image is not None:
                        adapted_recipe['image'] = image
                        print("Successfully generated and added image to recipe")
                    else:
                        print("Image generation returned None")
                else:
                    print("Diffusion model not available, skipping image generation")
            except Exception as e:
                print(f"ERROR during image generation: {e}")
                traceback.print_exc()
                print("Continuing without image")
        
        return adapted_recipe
    
    def find_recipe_by_title(self, title_query):
        """Find a recipe by title search"""
        results = self.retriever.keyword_search(title_query)
        return results[0]['recipe'] if results else None
    
    def suggest_recipes_for_diet(self, diet_constraints, top_k=5):
        """Suggest recipes that are compatible with diet constraints"""
        # Standardize diet constraint names
        standardized_constraints = []
        for constraint in diet_constraints:
            if not constraint.lower().endswith("diet"):
                constraint = f"{constraint} diet"
            standardized_constraints.append(constraint.lower())
        
        # Get diet information
        all_suggestions = []
        for diet in standardized_constraints:
            diet_info = self.retriever.retrieve_dietary_information(diet)
            if diet_info['example_recipes']:
                all_suggestions.extend(diet_info['example_recipes'])
        
        # Score suggestions based on compatibility with all diets
        scored_suggestions = []
        for recipe in all_suggestions:
            if isinstance(recipe, dict) and 'title' in recipe and 'ingredients' in recipe:
                # Calculate average compatibility score
                avg_score = 0
                for diet in standardized_constraints:
                    diet_score = 0
                    for ingredient in recipe['ingredients']:
                        score, _ = self.diet_model.check_food_compatibility(diet, ingredient)
                        diet_score += score
                    diet_score /= len(recipe['ingredients']) if recipe['ingredients'] else 1
                    avg_score += diet_score
                
                avg_score /= len(standardized_constraints)
                scored_suggestions.append((recipe, avg_score))
        
        # Sort by score and return top_k
        scored_suggestions.sort(key=lambda x: x[1], reverse=True)
        return [recipe for recipe, _ in scored_suggestions[:top_k]]