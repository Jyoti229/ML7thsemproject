# diet_planner_ml.py
import tkinter as tk
from tkinter import ttk, messagebox
from typing import List, Dict
import json
import random
from datetime import datetime
import csv
import joblib
import numpy as np
from recommender import SimpleRecommender

# ----- FoodItem, UserProfile, DietPlanner classes (modified to integrate recommender & predictor) -----

class FoodItem:
    def __init__(self, name: str, calories: int, protein: float, carbs: float, 
                 fats: float, category: str, portion: str, dietary_flags: List[str]):
        self.name = name
        self.calories = calories
        self.protein = protein
        self.carbs = carbs
        self.fats = fats
        self.category = category
        self.portion = portion
        self.dietary_flags = dietary_flags

class UserProfile:
    def __init__(self):
        self.weight = 0
        self.height = 0
        self.age = 0
        self.gender = ""
        self.activity_level = ""
        self.goal = ""
        self.dietary_restrictions = []
        self.allergies = []
        self.meals_per_day = 3
        self.meal_history = []

class DietPlanner:
    def __init__(self):
        self.food_database = self._initialize_food_database()
        self.user_profile = UserProfile()
        # Attempt to load calorie prediction model
        try:
            self.calorie_model = joblib.load('calorie_model.pkl')
            print("Loaded calorie_model.pkl")
        except Exception as e:
            print("No calorie_model.pkl found, falling back to Harris-Benedict. Error:", e)
            self.calorie_model = None
        # Recommender built from the food database
        from recommender import SimpleRecommender
        self.recommender = SimpleRecommender(self.food_database)

    def _initialize_food_database(self) -> List[FoodItem]:
        # (Use same database as original code -- shortened for brevity here, you can paste the full list)
        foods = [
            FoodItem("Chicken Breast (skinless)", 165, 31, 0, 3.6, "protein", "100g", ["lean-protein", "low-fat", "low-carb"]),
            FoodItem("Turkey Breast", 135, 30, 0, 2.1, "protein", "100g", ["lean-protein", "low-fat", "low-carb"]),
            FoodItem("Egg Whites", 52, 11, 0.7, 0.2, "protein", "100g", ["vegetarian", "lean-protein"]),
            FoodItem("Tuna (canned in water)", 116, 26, 0, 1.3, "protein", "100g", ["pescatarian", "lean-protein", "omega-3"]),
            FoodItem("Firm Tofu", 144, 15.6, 3.5, 8.7, "protein", "100g", ["vegan", "vegetarian", "gluten-free", "low-carb"]),
            FoodItem("Brown Rice", 112, 2.6, 23.5, 0.9, "carbs", "100g cooked", ["vegan", "gluten-free", "whole-grain"]),
            FoodItem("Quinoa", 120, 4.4, 21.3, 1.9, "carbs", "100g cooked", ["vegan", "gluten-free", "complete-protein"]),
            FoodItem("Broccoli", 55, 3.7, 11.2, 0.6, "vegetable", "100g", ["vegan", "gluten-free", "cruciferous"]),
            FoodItem("Spinach (raw)", 23, 2.9, 3.6, 0.4, "vegetable", "100g", ["vegan", "gluten-free", "low-carb", "leafy-green"]),
            FoodItem("Avocado", 160, 2, 8.5, 14.7, "fats", "100g", ["vegan", "gluten-free", "healthy-fats"]),
            FoodItem("Almonds", 579, 21.2, 21.7, 49.9, "fats", "100g", ["vegan", "gluten-free", "vitamin-e"]),
            FoodItem("Greek Yogurt (2%)", 73, 9.9, 3.6, 1.9, "protein", "100g", ["vegetarian", "probiotic"]),
            FoodItem("Apple", 52, 0.3, 13.8, 0.2, "fruit", "100g", ["vegan", "gluten-free", "fiber-rich"]),
            FoodItem("Banana", 89, 1.1, 22.8, 0.3, "fruit", "100g", ["vegan", "gluten-free", "potassium"]),
            FoodItem("Oatmeal", 68, 2.4, 12, 1.4, "carbs", "100g cooked", ["vegan", "fiber-rich"]),
            FoodItem("Olive Oil", 884, 0, 0, 100, "fats", "100g", ["vegan", "gluten-free", "monounsaturated"]),
            FoodItem("Salmon (Atlantic)", 208, 22, 0, 13, "protein", "100g", ["pescatarian", "omega-3", "healthy-fats"]),
            FoodItem("Chickpeas", 164, 8.9, 27.4, 2.6, "protein", "100g cooked", ["vegan", "vegetarian", "gluten-free", "fiber-rich"]),
            FoodItem("Sweet Potato", 86, 1.6, 20.1, 0.1, "carbs", "100g baked", ["vegan", "gluten-free", "vitamin-a"]),
            FoodItem("Kale (raw)", 49, 4.3, 8.8, 0.9, "vegetable", "100g", ["vegan", "gluten-free", "low-carb", "leafy-green"]),
            # Add rest of your items here (you had many more in original)
        ]
        return foods

    def predict_calories_ml(self):
        """Use trained ML model if available; otherwise fall back to Harris-Benedict calculation."""
        user = self.user_profile
        # Try ML model
        if self.calorie_model is not None:
            # map goal
            goal_map = {"lose":0, "maintain":1, "gain":2}
            goal_num = goal_map.get(user.goal, 1)
            # activity mapping: try to parse numbers if stored as strings like 'sedentary' -> use default values
            activity_map = {
                "sedentary":1.2, "light":1.375, "moderate":1.55, "very active":1.725, "very_active":1.725, "extra active":1.9, "extra_active":1.9
            }
            activity_val = activity_map.get(user.activity_level.lower(), 1.5)
            features = np.array([[user.age, user.weight, user.height, activity_val, goal_num]])
            try:
                pred = self.calorie_model.predict(features)[0]
                return float(round(pred))
            except Exception as e:
                print("Calorie model predict failed, falling back. Error:", e)

        # Fallback: Harris-Benedict as before
        if user.gender.lower() == "male":
            bmr = 88.362 + (13.397 * user.weight) + (4.799 * user.height) - (5.677 * user.age)
        else:
            bmr = 447.593 + (9.247 * user.weight) + (3.098 * user.height) - (4.330 * user.age)

        activity_multipliers = {
            "sedentary": 1.2,
            "light": 1.375,
            "moderate": 1.55,
            "very_active": 1.725,
            "extra_active": 1.9
        }
        tdee = bmr * activity_multipliers.get(self.user_profile.activity_level, 1.2)
        goal_adjustments = {"lose": -500, "maintain": 0, "gain": 500}
        daily_calories = tdee + goal_adjustments.get(self.user_profile.goal, 0)
        return float(round(daily_calories))

    def calculate_daily_needs(self):
        """Return dict with calories, protein(g), carbs(g), fats(g) using ML-predicted calories."""
        daily_calories = self.predict_calories_ml()
        # Keep macro splits same as before (you can make this ML later)
        if "vegan" in self.user_profile.dietary_restrictions:
            protein_ratio, carbs_ratio, fats_ratio = 0.25, 0.55, 0.20
        elif self.user_profile.goal == "lose":
            protein_ratio, carbs_ratio, fats_ratio = 0.40, 0.35, 0.25
        elif self.user_profile.goal == "gain":
            protein_ratio, carbs_ratio, fats_ratio = 0.30, 0.50, 0.20
        else:
            protein_ratio, carbs_ratio, fats_ratio = 0.30, 0.40, 0.30

        return {
            "calories": round(daily_calories),
            "protein": round(daily_calories * protein_ratio / 4),
            "carbs": round(daily_calories * carbs_ratio / 4),
            "fats": round(daily_calories * fats_ratio / 9)
        }

    def filter_foods_by_restrictions(self, foods: List[FoodItem]) -> List[FoodItem]:
        filtered_foods = foods.copy()
        for restriction in self.user_profile.dietary_restrictions:
            filtered_foods = [
                food for food in filtered_foods 
                if restriction.lower() in [flag.lower() for flag in food.dietary_flags]
            ]
        for allergen in self.user_profile.allergies:
            filtered_foods = [
                food for food in filtered_foods 
                if allergen.lower() not in food.name.lower()
            ]
        return filtered_foods

    def generate_meal_plan(self) -> List[List[FoodItem]]:
        """ML-driven meal plan: for each meal, get candidates from recommender and assemble greedily."""
        daily_needs = self.calculate_daily_needs()
        meal_plan = []
        calories_per_meal = max(150, daily_needs["calories"] / max(1, self.user_profile.meals_per_day))  # avoid too low

        # build dietary restrictions list and allergies
        diet_restr = self.user_profile.dietary_restrictions
        allergies = self.user_profile.allergies

        # Prepare simple liked history from previous meals (names)
        liked_names = []
        for h in self.user_profile.meal_history[-10:]:
            try:
                for meal in h['summary']['meals']:
                    for f in meal['foods']:
                        liked_names.append(f['name'])
            except:
                pass

        for meal_idx in range(self.user_profile.meals_per_day):
            # Filter foods and create candidates
            available_foods = self.filter_foods_by_restrictions(self.food_database)
            # Use recommender to get candidate DataFrame
            candidates_df = self.recommender.recommend_candidates(
                liked_food_names=liked_names,
                goal=self.user_profile.goal,
                top_k=60,
                dietary_restrictions=diet_restr,
                allergies=allergies
            )
            # assemble meal using greedy method targeting calories_per_meal
            assembled = self.recommender.assemble_meal_greedy(candidates_df, calorie_target=calories_per_meal, tol=0.25)
            # Convert assembled dicts back to FoodItem (portion may be present)
            meal_items = []
            for row in assembled:
                # Find the FoodItem by name in original database
                name = row.get('name')
                found = next((f for f in self.food_database if f.name == name), None)
                if found:
                    meal_items.append(found)
                else:
                    # create a simple FoodItem fallback
                    meal_items.append(FoodItem(row.get('name','Unknown'), int(row.get('calories',0)),
                                               float(row.get('protein',0)), float(row.get('carbs',0)),
                                               float(row.get('fats',0)), row.get('category','misc'),
                                               row.get('portion','100g'), row.get('dietary_flags','').split(',')))
            # If recommender returned empty, fallback to random (keeps system robust)
            if not meal_items:
                # original random logic fallback
                protein_foods = [food for food in available_foods if food.category == "protein"]
                if protein_foods:
                    meal_items.append(random.choice(protein_foods))
                carb_foods = [food for food in available_foods if food.category == "carbs"]
                if carb_foods:
                    meal_items.append(random.choice(carb_foods))
                vegetable_foods = [food for food in available_foods if food.category == "vegetable"]
                for _ in range(2):
                    if vegetable_foods:
                        meal_items.append(random.choice(vegetable_foods))
            meal_plan.append(meal_items)

        # Save history and return
        self.save_meal_plan_to_history(meal_plan)
        return meal_plan

    def save_meal_plan_to_history(self, meal_plan):
        summary = self.get_meal_plan_summary(meal_plan)
        history_entry = {"date": datetime.now().strftime("%Y-%m-%d"), "summary": summary}
        self.user_profile.meal_history.append(history_entry)
        try:
            with open('meal_history.csv','a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([history_entry["date"], json.dumps(summary)])
        except Exception as e:
            print("Failed to write history:", e)

    def get_meal_plan_summary(self, meal_plan: List[List[FoodItem]]):
        total_calories = total_protein = total_carbs = total_fats = 0
        meal_details = []
        for i, meal in enumerate(meal_plan, 1):
            mc = sum(food.calories for food in meal)
            mp = sum(food.protein for food in meal)
            mcar = sum(food.carbs for food in meal)
            mf = sum(food.fats for food in meal)
            total_calories += mc
            total_protein += mp
            total_carbs += mcar
            total_fats += mf
            meal_details.append({
                "meal_number": i,
                "foods": [{"name": food.name, "portion": food.portion} for food in meal],
                "nutrition": {"calories": round(mc), "protein": round(mp), "carbs": round(mcar), "fats": round(mf)}
            })
        return {"total_nutrition": {"calories": round(total_calories), "protein": round(total_protein), "carbs": round(total_carbs), "fats": round(total_fats)},
                "meals": meal_details}

# ----- GUI Class (mostly same as your original but uses updated DietPlanner) -----

class DietPlannerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Diet Planner (ML Integrated)")
        self.planner = DietPlanner()
        # Notebook and tabs
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=5)
        self.profile_tab = ttk.Frame(self.notebook)
        self.meal_plan_tab = ttk.Frame(self.notebook)
        self.history_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.profile_tab, text='Profile')
        self.notebook.add(self.meal_plan_tab, text='Meal Plan')
        self.notebook.add(self.history_tab, text='History')
        self._setup_profile_tab()
        self._setup_meal_plan_tab()
        self._setup_history_tab()

    def _setup_profile_tab(self):
        info_frame = ttk.LabelFrame(self.profile_tab, text="Personal Information")
        info_frame.pack(fill='x', padx=10, pady=5)
        # weight
        ttk.Label(info_frame, text="Weight (kg):").grid(row=0, column=0, padx=5, pady=5)
        self.weight_var = tk.StringVar()
        ttk.Entry(info_frame, textvariable=self.weight_var).grid(row=0, column=1, padx=5, pady=5)
        # height
        ttk.Label(info_frame, text="Height (cm):").grid(row=1, column=0, padx=5, pady=5)
        self.height_var = tk.StringVar()
        ttk.Entry(info_frame, textvariable=self.height_var).grid(row=1, column=1, padx=5, pady=5)
        # age
        ttk.Label(info_frame, text="Age:").grid(row=2, column=0, padx=5, pady=5)
        self.age_var = tk.StringVar()
        ttk.Entry(info_frame, textvariable=self.age_var).grid(row=2, column=1, padx=5, pady=5)
        # gender
        ttk.Label(info_frame, text="Gender:").grid(row=3, column=0, padx=5, pady=5)
        self.gender_var = tk.StringVar()
        gender_combo = ttk.Combobox(info_frame, textvariable=self.gender_var)
        gender_combo['values'] = ('Male','Female')
        gender_combo.grid(row=3, column=1, padx=5, pady=5)
        # activity
        ttk.Label(info_frame, text="Activity Level:").grid(row=4, column=0, padx=5, pady=5)
        self.activity_var = tk.StringVar()
        activity_combo = ttk.Combobox(info_frame, textvariable=self.activity_var)
        activity_combo['values'] = ('Sedentary','Light','Moderate','Very Active','Extra Active')
        activity_combo.grid(row=4, column=1, padx=5, pady=5)
        # goal
        ttk.Label(info_frame, text="Goal:").grid(row=5, column=0, padx=5, pady=5)
        self.goal_var = tk.StringVar()
        goal_combo = ttk.Combobox(info_frame, textvariable=self.goal_var)
        goal_combo['values'] = ('Lose','Maintain','Gain')
        goal_combo.grid(row=5, column=1, padx=5, pady=5)
        # meals per day
        ttk.Label(info_frame, text="Meals per day:").grid(row=6, column=0, padx=5, pady=5)
        self.meals_var = tk.StringVar(value="3")
        meals_spin = ttk.Spinbox(info_frame, from_=2, to=6, textvariable=self.meals_var)
        meals_spin.grid(row=6, column=1, padx=5, pady=5)
        # dietary restrictions
        restrictions_frame = ttk.LabelFrame(self.profile_tab, text="Dietary Restrictions")
        restrictions_frame.pack(fill='x', padx=10, pady=5)
        self.vegan_var = tk.BooleanVar()
        ttk.Checkbutton(restrictions_frame, text="Vegan", variable=self.vegan_var).grid(row=0, column=0, padx=5, pady=5)
        self.vegetarian_var = tk.BooleanVar()
        ttk.Checkbutton(restrictions_frame, text="Vegetarian", variable=self.vegetarian_var).grid(row=0, column=1, padx=5, pady=5)
        self.gluten_free_var = tk.BooleanVar()
        ttk.Checkbutton(restrictions_frame, text="Gluten-Free", variable=self.gluten_free_var).grid(row=0, column=2, padx=5, pady=5)
        # allergies
        allergies_frame = ttk.LabelFrame(self.profile_tab, text="Allergies")
        allergies_frame.pack(fill='x', padx=10, pady=5)
        ttk.Label(allergies_frame, text="List allergies (comma-separated):").grid(row=0, column=0, padx=5, pady=5)
        self.allergies_var = tk.StringVar()
        ttk.Entry(allergies_frame, textvariable=self.allergies_var).grid(row=0, column=1, padx=5, pady=5)
        # save button
        save_button = ttk.Button(self.profile_tab, text="Save Profile", command=self.save_profile)
        save_button.pack(pady=10)

    def _setup_meal_plan_tab(self):
        generate_button = ttk.Button(self.meal_plan_tab, text="Generate Meal Plan", command=self.generate_meal_plan)
        generate_button.pack(pady=10)
        self.meal_plan_text = tk.Text(self.meal_plan_tab, height=20, width=60)
        self.meal_plan_text.pack(padx=10, pady=5, fill='both', expand=True)
        export_button = ttk.Button(self.meal_plan_tab, text="Export Meal Plan", command=self.export_meal_plan)
        export_button.pack(pady=10)

    def _setup_history_tab(self):
        self.history_text = tk.Text(self.history_tab, height=20, width=60)
        self.history_text.pack(padx=10, pady=5, fill='both', expand=True)
        refresh_button = ttk.Button(self.history_tab, text="Refresh History", command=self.load_history)
        refresh_button.pack(pady=10)

    def save_profile(self):
        try:
            self.planner.user_profile.weight = float(self.weight_var.get())
            self.planner.user_profile.height = float(self.height_var.get())
            self.planner.user_profile.age = int(self.age_var.get())
            self.planner.user_profile.gender = self.gender_var.get()
            self.planner.user_profile.activity_level = self.activity_var.get().lower()
            self.planner.user_profile.goal = self.goal_var.get().lower()
            self.planner.user_profile.meals_per_day = int(self.meals_var.get())
            restrictions = []
            if self.vegan_var.get(): restrictions.append("vegan")
            if self.vegetarian_var.get(): restrictions.append("vegetarian")
            if self.gluten_free_var.get(): restrictions.append("gluten-free")
            self.planner.user_profile.dietary_restrictions = restrictions
            allergies = [a.strip() for a in self.allergies_var.get().split(',') if a.strip()]
            self.planner.user_profile.allergies = allergies
            messagebox.showinfo("Success", "Profile saved successfully!")
        except ValueError as e:
            messagebox.showerror("Error", "Please enter valid numeric values for weight, height, and age.")

    def generate_meal_plan(self):
        if not self.planner.user_profile.weight:
            messagebox.showwarning("Warning", "Please save your profile first!")
            return
        meal_plan = self.planner.generate_meal_plan()
        summary = self.planner.get_meal_plan_summary(meal_plan)
        self.meal_plan_text.delete(1.0, tk.END)
        self.meal_plan_text.insert(tk.END, "Your Meal Plan (ML-powered)\n\n")
        totals = summary["total_nutrition"]
        self.meal_plan_text.insert(tk.END, f"Daily Totals:\nCalories: {totals['calories']} kcal\nProtein: {totals['protein']}g\nCarbs: {totals['carbs']}g\nFats: {totals['fats']}g\n\n")
        for meal in summary["meals"]:
            self.meal_plan_text.insert(tk.END, f"\nMeal {meal['meal_number']}:\n")
            for food in meal["foods"]:
                self.meal_plan_text.insert(tk.END, f"- {food['name']} ({food['portion']})\n")
            nutrition = meal["nutrition"]
            self.meal_plan_text.insert(tk.END, f"\nMeal Nutrition:\nCalories: {nutrition['calories']} kcal\nProtein: {nutrition['protein']}g\nCarbs: {nutrition['carbs']}g\nFats: {nutrition['fats']}g\n")
            self.meal_plan_text.insert(tk.END, "\n" + "-"*40 + "\n")

    def export_meal_plan(self):
        try:
            content = self.meal_plan_text.get(1.0, tk.END)
            with open('meal_plan.txt', 'w') as f:
                f.write(content)
            messagebox.showinfo("Success", "Meal plan exported to meal_plan.txt")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export meal plan: {str(e)}")

    def load_history(self):
        try:
            self.history_text.delete(1.0, tk.END)
            with open('meal_history.csv', 'r') as file:
                reader = csv.reader(file)
                for row in reader:
                    date, summary = row
                    self.history_text.insert(tk.END, f"Date: {date}\n")
                    summary_dict = json.loads(summary)
                    self.history_text.insert(tk.END, f"Calories: {summary_dict['total_nutrition']['calories']} kcal\n")
                    self.history_text.insert(tk.END, "-"*40 + "\n")
        except FileNotFoundError:
            self.history_text.insert(tk.END, "No history available yet.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load history: {str(e)}")

def main():
    root = tk.Tk()
    app = DietPlannerGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
