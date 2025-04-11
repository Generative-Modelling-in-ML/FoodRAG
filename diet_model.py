import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

class DietFoodContrastiveModel(nn.Module):
    """
    Contrastive learning model that projects diet types and foods into a unified embedding space
    """
    def __init__(self, in_dim=768, proj_dim=128):
        super().__init__()
        
        self.proj_d = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )
        
        self.proj_n = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )

    def forward(self, phi_d, phi_n):
        """
        Project diet and food embeddings to the same space
        
        Args:
            phi_d: Diet embeddings from BERT
            phi_n: Food embeddings from BERT
            
        Returns:
            z_d, z_n: Projected embeddings in the same space
        """
        z_d = self.proj_d(phi_d)
        z_n = self.proj_n(phi_n)
        return z_d, z_n


class DietaryConstraintEmbedding:
    """
    A model that predicts compatibility between diets and foods
    """
    def __init__(self, model_path=None):
        # Initialize BERT for embeddings
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.bert_model = AutoModel.from_pretrained("bert-base-uncased")
        
        # Initialize contrastive model
        self.model = DietFoodContrastiveModel(in_dim=768, proj_dim=128)
        
        # Load pre-trained weights if available
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        
        # Set model to evaluation mode
        self.model.eval()
        self.bert_model.eval()
        
        # Diet types supported by the model
        self.diet_types = [
            "keto diet", "paleo diet", "vegetarian diet", "vegan diet", 
            "mediterranean diet", "intermittent fasting diet", "low-carb diet", 
            "dash diet", "mind diet", "flexitarian diet", "raw food diet", 
            "carnivore diet", "whole30 diet", "zone diet", "balanced diet"
        ]
    
    def get_bert_embedding(self, text):
        """Get BERT embedding for text input"""
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt")
            outputs = self.bert_model(**inputs)
            # Use the CLS token embedding
            embedding = outputs.last_hidden_state[:, 0, :]
        return embedding
    
    def check_food_compatibility(self, diet_name, food_name):
        """
        Check if a food is compatible with a diet
        
        Args:
            diet_name: The name of the diet (e.g., "keto diet")
            food_name: The name of the food to check
            
        Returns:
            compatibility_score: Value between -1 and 1 where higher values indicate compatibility
            is_compatible: Boolean indicating compatibility
        """
        # Standardize input
        diet_name = diet_name.lower()
        food_name = food_name.lower()
        
        # Get BERT embeddings
        diet_embed = self.get_bert_embedding(diet_name)
        food_embed = self.get_bert_embedding(food_name)
        
        # Project embeddings
        with torch.no_grad():
            z_diet, z_food = self.model(diet_embed, food_embed)
        
        # Calculate cosine similarity
        compatibility_score = F.cosine_similarity(z_diet, z_food).item()
        
        # Use threshold to determine compatibility
        # Values above 0.8 typically indicate "recommended" foods
        is_compatible = compatibility_score > 0.8
        
        return compatibility_score, is_compatible
    
    def batch_check_compatibility(self, diet_name, food_list):
        """Check compatibility for multiple foods with a diet"""
        results = {}
        
        for food in food_list:
            score, is_compatible = self.check_food_compatibility(diet_name, food)
            results[food] = {
                "score": score,
                "is_compatible": is_compatible
            }
        
        return results
    
    def find_compatible_substitutes(self, diet_name, food_name, candidate_foods):
        """
        Find compatible substitutes for a food given a diet
        
        Args:
            diet_name: The diet to check compatibility against
            food_name: The food to be substituted
            candidate_foods: List of potential substitute foods
            
        Returns:
            List of compatible foods sorted by compatibility score
        """
        compatibility_results = self.batch_check_compatibility(diet_name, candidate_foods)
        
        # Filter for compatible foods
        compatible_foods = [(food, data["score"]) for food, data in compatibility_results.items() 
                            if data["is_compatible"]]
        
        # Sort by compatibility score (high to low)
        compatible_foods.sort(key=lambda x: x[1], reverse=True)
        
        return compatible_foods


def train_diet_model(triples, bert_model, tokenizer, epochs=10000, lr=1e-6):
    """
    Train the Diet-Food Contrastive Model
    
    Args:
        triples: List of (diet, food, label) triples
        bert_model: Pre-trained BERT model
        tokenizer: BERT tokenizer
        epochs: Number of training epochs
        lr: Learning rate
        
    Returns:
        Trained DietFoodContrastiveModel
    """
    # Initialize model
    model = DietFoodContrastiveModel()
    
    # Get BERT embeddings for all diets and foods
    batch_size = len(triples)
    dim = 768
    phi_d = torch.empty(batch_size, dim)
    phi_n = torch.empty(batch_size, dim)
    y = torch.empty(batch_size, 1)
    
    for i, triple in enumerate(triples):
        diet, food, label = triple
        
        with torch.no_grad():
            # Get diet embedding
            diet_embed = tokenizer(diet, return_tensors="pt")
            outputs = bert_model(**diet_embed)
            phi_d[i, :] = outputs.last_hidden_state[0, 0]  # CLS token
            
            # Get food embedding
            food_embed = tokenizer(food, return_tensors="pt")
            outputs = bert_model(**food_embed)
            phi_n[i, :] = outputs.last_hidden_state[0, 0]  # CLS token
            
        y[i] = label
    
    # Convert label tensor
    y = y.squeeze().long()
    
    # Contrastive loss function
    def cosine_contrastive_loss(phi_r, phi_v, y, alpha=0.5):
        # Cosine similarity
        cos_sim = F.cosine_similarity(phi_r, phi_v, dim=-1)
        
        # Positive loss: 1 - cosine similarity
        pos_loss = 1 - cos_sim
        
        # Negative loss: max(0, cosine similarity - alpha)
        neg_loss = F.relu(cos_sim - alpha)
        
        # Combine losses based on labels
        loss = torch.where(y == 1, pos_loss, neg_loss)
        
        return loss.mean()
    
    # Training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        
        # Forward pass
        z_d, z_n = model(phi_d, phi_n)
        
        # Calculate loss
        loss = cosine_contrastive_loss(z_d, z_n, y, alpha=0.5)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 1000 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    
    return model