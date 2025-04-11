import json
import os
import csv
import re
import requests
from tqdm import tqdm
import torch
import numpy as np

class RecipeDataLoader:
    """
    Utilities for loading and processing recipe data
    """
    def __init__(self, recipe_data_path=None, nutrient_data_path=None):
        """
        Initialize the data loader
        
        Args:
            recipe_data_path: Path to recipe JSON data
            nutrient_data_path: Path to nutrient data (optional)
        """
        self.recipe_data_path = recipe_data_path
        self.nutrient_data_path = nutrient_data_path
        
        # Load data if paths are provided
        self.recipes = None
        self.nutrient_data = None
        
        if recipe_data_path and os.path.exists(recipe_data_path):
            self.load_recipe_data()
        
        if nutrient_data_path and os.path.exists(nutrient_data_path):
            self.load_nutrient_data()
    
    def load_recipe_data(self):
        """Load recipe data from JSON file"""
        print(f"Loading recipe data from {self.recipe_data_path}")
        with open(self.recipe_data_path, 'r') as f:
            self.recipes = json.load(f)
        print(f"Loaded {len(self.recipes)} recipes")
        return self.recipes
    
    def load_nutrient_data(self):
        """Load nutrient data"""
        print(f"Loading nutrient data from {self.nutrient_data_path}")
        with open(self.nutrient_data_path, 'r') as f:
            self.nutrient_data = json.load(f)
        print(f"Loaded nutrient data for {len(self.nutrient_data)} items")
        return self.nutrient_data
    
    def download_usda_data(self, output_path="usda_nutrients.json"):
        """
        Download and process USDA nutritional database
        
        Args:
            output_path: Path to save processed data
            
        Returns:
            Processed nutrient data
        """
        try:
            # URLs for USDA data files
            food_url = "https://fdc.nal.usda.gov/fdc-app.html#/food-details/747447/nutrients"
            nutrient_url = "https://fdc.nal.usda.gov/portal-data/external/nutrients.csv"
            
            # Create temp directory
            os.makedirs("temp_usda", exist_ok=True)
            
            # Download nutrient data
            print("Downloading USDA nutrient data...")
            response = requests.get(nutrient_url)
            with open("temp_usda/nutrients.csv", 'wb') as f:
                f.write(response.content)
            
            # Process nutrient data
            print("Processing nutrient data...")
            nutrients = {}
            with open("temp_usda/nutrients.csv", 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    nutrient_id = row.get('id', '')
                    if nutrient_id:
                        nutrients[nutrient_id] = {
                            'name': row.get('name', ''),
                            'unit': row.get('unit_name', '')
                        }
            
            # Process food data
            print("Processing food data...")
            processed_data = []
            
            # Save processed data
            with open(output_path, 'w') as f:
                json.dump(processed_data, f, indent=2)
            
            print(f"Processed USDA data saved to {output_path}")
            self.nutrient_data_path = output_path
            self.nutrient_data = processed_data
            
            # Clean up temp files
            import shutil
            shutil.rmtree("temp_usda")
            
            return processed_data
            
        except Exception as e:
            print(f"Error downloading USDA data: {e}")
            return None
    
    def preprocess_recipes(self, output_path=None):
        """
        Preprocess recipes for better compatibility with the system
        
        Args:
            output_path: Path to save processed recipes
            
        Returns:
            Processed recipe data
        """
        if not self.recipes:
            print("No recipe data loaded. Load recipe data first.")
            return None
        
        print("Preprocessing recipes...")
        processed_recipes = []
        
        for recipe in tqdm(self.recipes):
            # Create processed recipe
            processed_recipe = {
                'id': recipe.get('id', str(len(processed_recipes))),
                'title': recipe.get('title', 'Untitled Recipe'),
                'ingredients': [],
                'instructions': recipe.get('instructions', []),
                'tags': recipe.get('tags', [])
            }
            
            # Process ingredients
            for ing in recipe.get('ingredients', []):
                if isinstance(ing, dict) and 'text' in ing:
                    processed_recipe['ingredients'].append(ing)
                elif isinstance(ing, str):
                    processed_recipe['ingredients'].append({'text': ing})
            
            # Process instructions if they're in a different format
            if isinstance(processed_recipe['instructions'], str):
                # Split by newlines or numbers
                instructions = re.split(r'\n|^\d+\.\s*', processed_recipe['instructions'])
                processed_recipe['instructions'] = [
                    instr.strip() for instr in instructions if instr.strip()
                ]
            
            # Add additional metadata if available
            if 'description' in recipe:
                processed_recipe['description'] = recipe['description']
            
            if 'nutr_per_ingredient' in recipe:
                processed_recipe['nutr_per_ingredient'] = recipe['nutr_per_ingredient']
            
            # Add to processed list
            processed_recipes.append(processed_recipe)
        
        print(f"Processed {len(processed_recipes)} recipes")
        
        # Save if output path is provided
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(processed_recipes, f, indent=2)
            print(f"Processed recipes saved to {output_path}")
        
        return processed_recipes
    
    def generate_triples_dataset(self, diet_data_path=None, output_path="diet_food_triples.json"):
        """
        Generate (diet, food, label) triples dataset for training diet model
        
        Args:
            diet_data_path: Path to diet data JSON (optional)
            output_path: Path to save triples dataset
            
        Returns:
            List of (diet, food, label) triples
        """
        # Load diet data if provided, otherwise use default
        if diet_data_path and os.path.exists(diet_data_path):
            with open(diet_data_path, 'r') as f:
                diet_data = json.load(f)
        else:
            # Default diet data (based on Model.ipynb)
            diet_data = { 
                # Collected from https://becomeanutritionist.org/blog/types-of-diets/
                "Balanced Diet": {
                    "recommended": [],
                    "avoids": []
                },
                "Keto Diet": {
                    "recommended": ["Meat", "Fatty fish", "Eggs", "Cheese", "Butter", "Nuts", "Oils", "Low-carb vegetables"],
                    "avoids": ["Grains", "Sugar", "High-carb fruits", "Legumes"]
                },
                "Paleo Diet": {
                    "recommended": ["Meat", "Fish", "Fruits", "Vegetables", "Nuts", "Seeds"],
                    "avoids": ["Dairy", "Grains", "Legumes", "Processed foods", "Sugar"]
                },
                "Vegetarian Diet": {
                    "recommended": ["Vegetables", "Fruits", "Grains", "Dairy", "Eggs"],
                    "avoids": ["Meat"]
                },
                "Vegan Diet": {
                    "recommended": ["Fruits", "Vegetables", "Grains", "Legumes", "Nuts", "Seeds"],
                    "avoids": ["All animal-derived products"]
                },
                "Mediterranean Diet": {
                    "recommended": ["Olive oil", "Fish", "Fruits", "Vegetables", "Legumes", "Nuts", "Whole grains"],
                    "avoids": ["Red meat", "Refined grains", "Processed foods"]
                },
                "Intermittent Fasting Diet": {
                    "recommended": [],
                    "avoids": []
                },
                "Low-Carb Diet": {
                    "recommended": ["Meat", "Fish", "Eggs", "Vegetables", "Nuts", "Seeds"],
                    "avoids": ["Grains", "Starchy vegetables", "Sugars"]
                },
                "DASH Diet": {
                    "recommended": ["Fruits", "Vegetables", "Whole grains", "Lean protein", "Low-fat dairy"],
                    "avoids": ["High sodium foods", "Sugary drinks", "Red meat"]
                },
                "MIND Diet": {
                    "recommended": ["Leafy greens", "Berries", "Nuts", "Olive oil", "Whole grains", "Fish"],
                    "avoids": ["Red meat", "Butter", "Cheese", "Pastries", "Fried food"]
                },
                "Flexitarian Diet": {
                    "recommended": ["Vegetables", "Fruits", "Whole grains", "Legumes", "Some meat"],
                    "avoids": ["Highly processed foods"]
                },
                "Raw Food Diet": {
                    "recommended": ["Fruits", "Vegetables", "Nuts", "Seeds", "Raw grains", "Legumes"],
                    "avoids": ["Cooked or processed foods", "Refined sugars", "Oils"]
                },
                "Carnivore Diet": {
                    "recommended": ["Meat", "Fish", "Eggs", "Animal fats"],
                    "avoids": ["Fruits", "Vegetables", "Grains", "Legumes"]
                },
                "Whole30 Diet": {
                    "recommended": ["Meat", "Seafood", "Vegetables", "Fruits", "Some fats"],
                    "avoids": ["Sugar", "Grains", "Dairy", "Alcohol", "Legumes", "Processed foods"]
                },
                "Zone Diet": {
                    "recommended": ["Lean proteins", "Healthy fats", "Low-glycemic carbs"],
                    "avoids": ["Refined carbs", "Sugary snacks", "Processed foods"]
                }
            }
        
        # Generate triples
        triples = []
        
        for diet, food_dict in diet_data.items():
            for food in food_dict.get("recommended", []):
                triples.append((diet.lower(), food.lower(), 1))
            for food in food_dict.get("avoids", []):
                triples.append((diet.lower(), food.lower(), -1))
        
        # Special cases handling
        foods = [
            "Meat", "Fatty fish", "fish", "Eggs", "Cheese", "Butter", "Nuts", "Oils", 
            "Low-carb vegetables", "Grains", "whole grains", "Sugar", "High-carb fruits", 
            "Legumes", "Lean proteins", "Healthy fats", "Low-glycemic carbs"
        ]
        
        for food in foods:
            triples.append(("Balanced Diet".lower(), food.lower(), 1))
            triples.append(("Intermittent Fasting Diet".lower(), food.lower(), 1))
        
        print(f"Generated {len(triples)} (diet, food, label) triples")
        
        # Save triples
        with open(output_path, 'w') as f:
            json.dump(triples, f, indent=2)
        
        print(f"Triples dataset saved to {output_path}")
        return triples
    
    def load_model_data(self, model_path):
        """
        Load saved model data
        
        Args:
            model_path: Path to saved model
            
        Returns:
            Loaded model data
        """
        if not os.path.exists(model_path):
            print(f"Model file {model_path} not found")
            return None
        
        model_data = torch.load(model_path, map_location=torch.device('cpu'))
        print(f"Loaded model data from {model_path}")
        return model_data

def setup_resources(config):
    """
    Set up resources needed for the recipe adaptation system
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if setup was successful, False otherwise
    """
    try:
        # Create directory structure
        os.makedirs("data", exist_ok=True)
        os.makedirs("models", exist_ok=True)
        os.makedirs("output", exist_ok=True)
        
        # Initialize data loader
        loader = RecipeDataLoader()
        
        # Download example recipe data if not available
        if not os.path.exists(config.get("recipe_data_path", "data/recipes.json")):
            print("Recipe data not found. Downloading example data...")
            # Simplified example - in practice, download from a real source
            example_recipes = [
                {
                    "title": "Classic Spaghetti Bolognese",
                    "ingredients": [
                        {"text": "ground beef, 1 pound"},
                        {"text": "onion, 1 medium, chopped"},
                        {"text": "garlic, 3 cloves, minced"},
                        {"text": "carrots, 2 medium, diced"},
                        {"text": "celery stalks, 2, diced"},
                        {"text": "tomato paste, 2 tablespoons"},
                        {"text": "canned crushed tomatoes, 28 ounces"},
                        {"text": "dried oregano, 1 teaspoon"},
                        {"text": "dried basil, 1 teaspoon"},
                        {"text": "salt, 1 teaspoon"},
                        {"text": "black pepper, 1/2 teaspoon"},
                        {"text": "spaghetti, 1 pound"},
                        {"text": "parmesan cheese, grated, for serving"}
                    ],
                    "instructions": [
                        "Heat oil in a large pot over medium heat.",
                        "Add onion, garlic, carrots, and celery. Cook until softened.",
                        "Add ground beef and cook until browned.",
                        "Stir in tomato paste, then add crushed tomatoes and herbs.",
                        "Simmer for 30 minutes.",
                        "Cook spaghetti according to package directions.",
                        "Serve sauce over pasta with grated parmesan."
                    ]
                },
                {
                    "title": "Garden Vegetable Soup",
                    "ingredients": [
                        {"text": "olive oil, 2 tablespoons"},
                        {"text": "onion, 1 large, chopped"},
                        {"text": "carrots, 2 medium, diced"},
                        {"text": "celery stalks, 2, diced"},
                        {"text": "zucchini, 1 medium, diced"},
                        {"text": "green beans, 1 cup, trimmed and cut"},
                        {"text": "garlic, 3 cloves, minced"},
                        {"text": "vegetable broth, 6 cups"},
                        {"text": "diced tomatoes, 14.5 ounce can"},
                        {"text": "italian seasoning, 1 teaspoon"},
                        {"text": "salt, to taste"},
                        {"text": "black pepper, to taste"},
                        {"text": "spinach, 2 cups, fresh"}
                    ],
                    "instructions": [
                        "Heat oil in a large pot over medium heat.",
                        "Add onion, carrots, and celery. Cook for 5 minutes.",
                        "Add zucchini, green beans, and garlic. Cook for 2 minutes.",
                        "Add broth, tomatoes, and seasonings. Bring to a boil.",
                        "Reduce heat and simmer for 15 minutes.",
                        "Add spinach and cook until wilted.",
                        "Adjust seasonings and serve hot."
                    ]
                }
            ]
            
            with open(config.get("recipe_data_path", "data/recipes.json"), 'w') as f:
                json.dump(example_recipes, f, indent=2)
            
            print(f"Example recipe data saved to {config.get('recipe_data_path', 'data/recipes.json')}")
        
        # Generate triples dataset for diet model training if not available
        if not os.path.exists(config.get("triples_data_path", "data/diet_food_triples.json")):
            print("Diet-food triples data not found. Generating default dataset...")
            loader.generate_triples_dataset(output_path=config.get("triples_data_path", "data/diet_food_triples.json"))
        
        return True
    
    except Exception as e:
        print(f"Error setting up resources: {e}")
        return False

if __name__ == "__main__":
    # Example usage
    config = {
        "recipe_data_path": "data/recipes.json",
        "triples_data_path": "data/diet_food_triples.json"
    }
    
    success = setup_resources(config)
    print(f"Resource setup {'successful' if success else 'failed'}")