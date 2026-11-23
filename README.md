# econ3916-lab19-random-forests

# Tree-Based Models: Random Forests

## Objective
I wanted to see whether a random forest predicts California house values better than a single tree and linear models, what its feature importances do and don't tell me, and whether I can trust AI-written code by checking it against my own numbers.

## Methodology
- I split California Housing (8 features) into 16,512 training rows and 4,128 test rows.
- I fitted four models on the same training rows: a single decision tree with no depth limit, OLS, Ridge (in a pipeline behind StandardScaler), and a random forest of 100 trees with out-of-bag scoring. I scored each on RMSE and R² for both the training and test rows.
- I compared two measures of feature importance for the forest: impurity-based (MDI), which comes from training, and permutation importance, which shuffles one column at a time on the test rows and measures how far test R² falls.
- I tuned the forest's max_depth and max_features with 3-fold cross-validation using GridSearchCV, then scored the winning forest on the test rows.
- I wrote my own prompt asking an AI for a function that computes permutation importance and draws a bar chart of the top features, ran it through a check against my Step 3 numbers, and rewrote the prompt until it matched.

## Key Findings
- Test R² was 0.6228 for the single tree, 0.5758 for Ridge, and 0.8046 for the forest. The single tree had a train R² of 1.0 with 15,860 leaves for 16,512 training rows, so it basically memorized the training data. Ridge did no better than OLS, since with 16,512 rows and only 8 features there was almost no variance for the penalty to remove.
- The forest beating Ridge by about 0.23 with the same features suggests house values depend on the features in nonlinear ways and through interactions, like location, that a linear model can't capture.
- Latitude scored 0.0890 by MDI but 0.4427 by permutation, ranking 3rd by MDI and 2nd by permutation. MDI is computed on training rows and splits credit between features that work together, like latitude and longitude, while permutation shows how much the forest relies on a feature for new data.
- Tuning raised the forest's test R² to 0.8094. Within the grid, max_depth mattered more, but compared with my first forest the tuned one only changed max_features, from all 8 to 4.
- I went 2 rounds with the AI. In round 1 it picked a random seed of 0 on its own, so every importance was off from my numbers (up to 6% on the forest and 22% on another model). After I specified the seed and settings in round 2, its numbers matched mine exactly on every check. The lesson for me was that AI code can look right and still give different numbers, so it only counts once I've checked it against a known answer.
