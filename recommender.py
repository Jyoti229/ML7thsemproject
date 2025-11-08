# recommender.py
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def fooditems_to_dataframe(food_items):
    """
    Convert list of FoodItem objects into a pandas DataFrame with numeric features.
    Assumes each item has attributes: name, calories, protein, carbs, fats, category, portion, dietary_flags
    """
    rows = []
    for f in food_items:
        # Normalize category into columns later if needed
        rows.append({
            'name': f.name,
            'calories': float(f.calories),
            'protein': float(f.protein),
            'carbs': float(f.carbs),
            'fats': float(f.fats),
            'category': f.category,
            'portion': f.portion,
            'dietary_flags': ','.join(f.dietary_flags)
        })
    df = pd.DataFrame(rows)
    # For now features are nutrition vector; later you can add TF-IDF of ingredients or tags
    return df

class SimpleRecommender:
    def __init__(self, food_items):
        """
        food_items: list of FoodItem objects from DietPlanner
        """
        self.items_df = fooditems_to_dataframe(food_items).reset_index(drop=True)
        # Build feature matrix from nutrition columns (protein, carbs, fats, calories)
        # We scale calories down to avoid dominance (simple normalization)
        feat = self.items_df[['protein','carbs','fats','calories']].astype(float).values
        # normalize per-column (min-max) to keep features comparable
        mins = feat.min(axis=0)
        maxs = feat.max(axis=0)
        denom = (maxs - mins)
        denom[denom == 0] = 1.0
        self.feature_matrix = (feat - mins) / denom
        self._mins = mins
        self._maxs = maxs

    def _user_vector_from_preferences(self, liked_names=None, goal=None):
        # If liked items exist, average their vectors; otherwise derive from goal: e.g., for 'lose' prefer higher protein per cal
        if liked_names:
            mask = self.items_df['name'].isin(liked_names)
            if mask.sum() > 0:
                vecs = self.feature_matrix[mask.values]
                return vecs.mean(axis=0)
        # fallback: create vector based on goal
        # goal: 'lose' -> emphasize protein and lower calories; 'gain' -> allow higher calories; 'maintain' balanced
        if goal == 'lose':
            # more protein, moderate carbs, low fats, low calories
            proto = np.array([0.9, 0.4, 0.2, 0.1])
        elif goal == 'gain':
            proto = np.array([0.6, 0.7, 0.5, 0.9])
        else:  # maintain / default
            proto = np.array([0.7, 0.6, 0.4, 0.5])
        return proto

    def recommend_candidates(self, liked_food_names=None, goal=None, top_k=30, dietary_restrictions=None, allergies=None):
        """
        Returns top_k candidate rows (pandas DataFrame) ordered by similarity to user vector,
        also filtered by dietary restrictions and allergies.
        """
        user_vec = self._user_vector_from_preferences(liked_food_names, goal)
        sims = cosine_similarity([user_vec], self.feature_matrix).flatten()
        rank_idx = sims.argsort()[::-1]
        candidates = self.items_df.iloc[rank_idx].copy()
        # Filter by dietary restrictions: each restriction must be present in dietary_flags (simple approach)
        if dietary_restrictions:
            def keep_row_flags(flags):
                flags_l = flags.lower()
                for r in dietary_restrictions:
                    if r.lower() not in flags_l:
                        return False
                return True
            candidates = candidates[candidates['dietary_flags'].apply(keep_row_flags)]
        # Filter allergies by checking name (simple)
        if allergies:
            for a in allergies:
                candidates = candidates[~candidates['name'].str.contains(a, case=False)]
        return candidates.head(top_k)

    def assemble_meal_greedy(self, candidates_df, calorie_target, tol=0.2):
        """
        Greedy assembly: pick items with better protein_per_calorie until reach target within tolerance.
        Returns list of chosen item rows as dicts.
        """
        cand = candidates_df.copy()
        cand['protein_per_cal'] = cand['protein'] / (cand['calories'] + 1e-6)
        cand = cand.sort_values('protein_per_cal', ascending=False).reset_index(drop=True)
        chosen = []
        total_cal = 0.0
        lower = calorie_target * (1 - tol)
        upper = calorie_target * (1 + tol)
        for _, row in cand.iterrows():
            if total_cal + row['calories'] <= upper:
                chosen.append(row.to_dict())
                total_cal += row['calories']
            if total_cal >= lower:
                break
        # if nothing chosen (e.g., target is small), choose smallest calorie item
        if not chosen and not cand.empty:
            chosen.append(cand.iloc[[0]].to_dict('records')[0])
        return chosen
