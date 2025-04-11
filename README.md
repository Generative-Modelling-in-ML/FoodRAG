# FoodRAG

## How to Run:
* Clone the repo
* Include the .json file with the recipes (recipes_with_nutritional_info.json -- downloaded from recipe1m+)
* Run
  ```
  !pip install -r requirements.txt
  
  ```

* Example: Run
  ```
  !python /content/main.py --recipe_title "Chocolate Cake" --diet_constraints gluten-free vegan --diet_model "diet_food_contrastive_model.pth" --recipe_data "recipes_with_nutritional_info.json" --generate_image

  ```
