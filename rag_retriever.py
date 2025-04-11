import json
import numpy as np
import torch
import faiss
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

# Download NLTK resources if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class RecipeKnowledgeRetriever:
    """
    Retrieval-Augmented Generation component for recipe knowledge
    """
    def __init__(self, recipe_data_path, nutrient_data_path=None, embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the recipe knowledge retriever
        
        Args:
            recipe_data_path: Path to JSON file with recipe data
            nutrient_data_path: Path to nutrient data file (optional)
            embedding_model: Model to use for embedding recipes
        """
        self.recipe_data_path = recipe_data_path
        self.nutrient_data_path = nutrient_data_path
        
        # Load recipe data
        print(f"Loading recipe data from {recipe_data_path}")
        with open(recipe_data_path, 'r') as f:
            self.recipes = json.load(f)
        
        # Load nutrient data if available
        self.nutrient_data = None
        if nutrient_data_path:
            print(f"Loading nutrient data from {nutrient_data_path}")
            with open(nutrient_data_path, 'r') as f:
                self.nutrient_data = json.load(f)
        
        # Initialize embedding model
        print(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Initialize TF-IDF vectorizer for keyword search
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        
        # Initialize FAISS index for semantic search
        self.faiss_index = None
        self.recipe_ids = []
        
        # Build search indices
        self._build_indices()
    
    def _build_indices(self):
        """Build search indices for recipes"""
        print("Building search indices...")
        
        # Prepare text corpus for TF-IDF
        corpus = []
        self.recipe_ids = []
        
        for recipe in self.recipes:
            # Create a text representation of the recipe
            recipe_text = f"{recipe['title']} "
            if 'description' in recipe and recipe['description']:
                recipe_text += f"{recipe['description']} "
            
            # Add ingredients
            ingredients_text = " ".join([ing['text'] for ing in recipe['ingredients']])
            recipe_text += f"{ingredients_text} "
            
            # Add instructions if available
            # if 'instructions' in recipe:
            #     instructions_text = " ".join(recipe['instructions'])
            #     recipe_text += instructions_text
            
            # Add instructions if available
            if 'instructions' in recipe:
                if isinstance(recipe['instructions'], list):
                    # Handle both cases: list of strings or list of dictionaries
                    if recipe['instructions'] and isinstance(recipe['instructions'][0], dict):
                        # If instructions are dictionaries, extract the text field
                        if 'text' in recipe['instructions'][0]:
                            instructions_text = " ".join([step['text'] for step in recipe['instructions']])
                        else:
                            # Use the first field available as a fallback
                            first_key = next(iter(recipe['instructions'][0]))
                            instructions_text = " ".join([step[first_key] for step in recipe['instructions']])
                    else:
                        # Instructions are strings
                        instructions_text = " ".join(recipe['instructions'])
                else:
                    # If instructions is a single string
                    instructions_text = str(recipe['instructions'])
                
                recipe_text += instructions_text

            corpus.append(recipe_text)
            self.recipe_ids.append(recipe['id'] if 'id' in recipe else len(self.recipe_ids))
        
        # Build TF-IDF index
        print("Building TF-IDF index...")
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000, 
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus)
        
        # Build FAISS index for semantic search
        print("Building FAISS index for semantic search...")
        # Generate recipe embeddings
        recipe_embeddings = []
        for i, recipe_text in enumerate(corpus):
            # Print progress every 1000 recipes
            if i % 1000 == 0 and i > 0:
                print(f"Embedded {i}/{len(corpus)} recipes")
            
            embedding = self.embedding_model.encode(recipe_text)
            recipe_embeddings.append(embedding)
        
        # Convert to numpy array
        embeddings_array = np.array(recipe_embeddings).astype('float32')
        
        # Build FAISS index
        dimension = embeddings_array.shape[1]
        self.faiss_index = faiss.IndexFlatL2(dimension)
        self.faiss_index.add(embeddings_array)
        
        print(f"Built indices for {len(self.recipes)} recipes")
    
    def keyword_search(self, query, top_k=5):
        """
        Search recipes by keywords using TF-IDF
        
        Args:
            query: Search query string
            top_k: Number of results to return
            
        Returns:
            List of top-k matching recipes
        """
        query_vec = self.tfidf_vectorizer.transform([query])
        
        # Calculate cosine similarity between query and recipes
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # Get indices of top-k similar recipes
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        # Return top-k recipes
        results = []
        for idx in top_indices:
            recipe_id = self.recipe_ids[idx]
            recipe = next((r for r in self.recipes if r.get('id', -1) == recipe_id), None)
            if recipe:
                results.append({
                    'recipe': recipe,
                    'score': similarities[idx]
                })
        
        return results
    
    def semantic_search(self, query, top_k=5):
        """
        Search recipes semantically using embeddings
        
        Args:
            query: Search query string
            top_k: Number of results to return
            
        Returns:
            List of top-k semantically similar recipes
        """
        # Encode query
        query_embedding = self.embedding_model.encode(query)
        query_embedding = np.array([query_embedding]).astype('float32')
        
        # Search in FAISS index
        distances, indices = self.faiss_index.search(query_embedding, top_k)
        
        # Return results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0 and idx < len(self.recipe_ids):
                recipe_id = self.recipe_ids[idx]
                recipe = next((r for r in self.recipes if r.get('id', -1) == recipe_id), None)
                if recipe:
                    results.append({
                        'recipe': recipe,
                        'score': 1.0 / (1.0 + distances[0][i])  # Convert distance to similarity score
                    })
        
        return results
    
    def retrieve_ingredient_knowledge(self, ingredient_name, top_k=5):
        """
        Retrieve knowledge about a specific ingredient
        
        Args:
            ingredient_name: Name of the ingredient
            top_k: Number of results to return
            
        Returns:
            Knowledge about the ingredient from various recipes
        """
        # Search for recipes containing this ingredient
        query = f"ingredient {ingredient_name}"
        recipes = self.keyword_search(query, top_k=top_k)
        
        # Extract knowledge about this ingredient
        knowledge = {
            'ingredient': ingredient_name,
            'nutrient_info': self._get_nutrient_info(ingredient_name),
            'usage_examples': [],
            'common_pairings': []
        }
        
        # Extract usage examples and common pairings
        ingredient_occurrences = []
        common_pairings = {}
        
        for result in recipes:
            recipe = result['recipe']
            
            # Check if this recipe uses the ingredient
            for ing in recipe['ingredients']:
                if ingredient_name.lower() in ing['text'].lower():
                    # Add usage example
                    knowledge['usage_examples'].append({
                        'recipe_title': recipe['title'],
                        'usage_context': ing['text']
                    })
                    ingredient_occurrences.append(ing['text'])
                    
                    # Find common pairings
                    for other_ing in recipe['ingredients']:
                        if ingredient_name.lower() not in other_ing['text'].lower():
                            other_name = other_ing['text']
                            if other_name in common_pairings:
                                common_pairings[other_name] += 1
                            else:
                                common_pairings[other_name] = 1
        
        # Sort common pairings by frequency
        sorted_pairings = sorted(common_pairings.items(), key=lambda x: x[1], reverse=True)
        knowledge['common_pairings'] = [pair[0] for pair in sorted_pairings[:10]]  # Top 10 pairings
        
        return knowledge
    
    def _get_nutrient_info(self, ingredient_name):
        """Get nutritional information for an ingredient if available"""
        if not self.nutrient_data:
            return None
        
        # Search for the ingredient in nutrient data
        for item in self.nutrient_data:
            if ingredient_name.lower() in item['name'].lower():
                return item['nutrients']
        
        return None
    
    def retrieve_dietary_information(self, diet_name):
        """
        Retrieve information about a specific diet
        
        Args:
            diet_name: Name of the diet (e.g., "keto", "vegan")
            
        Returns:
            Information about the diet
        """
        # Search for recipes and information about this diet
        query = f"{diet_name} diet recipes"
        recipes = self.semantic_search(query, top_k=5)
        
        # Try keyword search if semantic search doesn't yield good results
        if not recipes or recipes[0]['score'] < 0.5:
            recipes = self.keyword_search(query, top_k=5)
        
        # Collect information about the diet
        diet_info = {
            'name': diet_name,
            'common_ingredients': {},
            'avoided_ingredients': {},
            'example_recipes': []
        }
        
        # Extract common and avoided ingredients
        for result in recipes:
            recipe = result['recipe']
            diet_info['example_recipes'].append({
                'title': recipe['title'],
                'ingredients': [ing['text'] for ing in recipe['ingredients']]
            })
            
            # Count ingredient frequencies
            for ing in recipe['ingredients']:
                ing_text = ing['text']
                if ing_text in diet_info['common_ingredients']:
                    diet_info['common_ingredients'][ing_text] += 1
                else:
                    diet_info['common_ingredients'][ing_text] = 1
        
        # Sort ingredients by frequency
        diet_info['common_ingredients'] = dict(
            sorted(diet_info['common_ingredients'].items(), 
                  key=lambda x: x[1], 
                  reverse=True)[:20]  # Top 20 common ingredients
        )
        
        return diet_info
    
    def retrieve_adaptation_knowledge(self, recipe, diet_constraints):
        """
        Retrieve knowledge needed to adapt a recipe to dietary constraints
        
        Args:
            recipe: Recipe to adapt
            diet_constraints: Dietary constraints to adapt to
            
        Returns:
            Knowledge to guide recipe adaptation
        """
        # Construct query combining recipe and diet information
        diet_str = " ".join(diet_constraints)
        query = f"adapt {recipe['title']} recipe for {diet_str} diet"
        
        # Retrieve similar adaptation examples
        examples = self.semantic_search(query, top_k=3)
        
        # Extract ingredients that might need substitution
        ingredients_to_check = [ing['text'] for ing in recipe['ingredients']]
        
        # Gather knowledge for adaptation
        adaptation_knowledge = {
            'similar_adaptations': [],
            'ingredient_substitutions': {},
            'general_advice': []
        }
        
        # Extract similar adaptations
        for result in examples:
            adaptation_knowledge['similar_adaptations'].append({
                'title': result['recipe']['title'],
                'ingredients': [ing['text'] for ing in result['recipe']['ingredients']]
            })
        
        # Get information for each potentially problematic ingredient
        for ingredient in ingredients_to_check:
            ing_knowledge = self.retrieve_ingredient_knowledge(ingredient)
            adaptation_knowledge['ingredient_substitutions'][ingredient] = ing_knowledge
        
        # Add general advice based on the diet
        for diet in diet_constraints:
            diet_info = self.retrieve_dietary_information(diet)
            adaptation_knowledge['general_advice'].append({
                'diet': diet,
                'common_ingredients': list(diet_info['common_ingredients'].keys())[:5],
                'example_recipe': diet_info['example_recipes'][0] if diet_info['example_recipes'] else None
            })
        
        return adaptation_knowledge