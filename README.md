# NBA Sports Betting Using Machine Learning 🏀
<img src="https://github.com/kyleskom/NBA-Machine-Learning-Sports-Betting/blob/master/Screenshots/output.png" width="1010" height="292" />

A machine learning AI used to predict the winners and under/overs of NBA games. Takes all team data from the 2007-08 season to current season, matched with odds of those games, using a neural network to predict winning bets for today's games. Achieves ~69% accuracy on money lines and ~55% on under/overs. Outputs expected value for teams money lines to provide better insight. The fraction of your bankroll to bet based on the Kelly Criterion is also outputted. Note that a popular, less risky approach is to bet 50% of the stake recommended by the Kelly Criterion.

## Recent Updates

### Model Architecture Improvements
- Modernized neural network architecture with:
  - Batch Normalization for better training stability
  - LeakyReLU activation for improved gradient flow
  - Dropout layers for regularization
  - L2 regularization to prevent overfitting
  - Learning rate scheduling for better convergence

### Feature Engineering Enhancements
- Added comprehensive feature engineering:
  - Relative metrics between teams (differences and ratios)
  - Rolling averages for key statistics (3, 5, and 10-game windows)
  - Streak features (winning/losing streaks)
  - Head-to-head historical performance
  - Rest day impact analysis

### Training and Evaluation
- Implemented proper time-based validation splits
- Added comprehensive evaluation metrics:
  - Accuracy, Precision, Recall, F1-score
  - ROC-AUC for binary classification
  - Confusion matrices
  - Per-class performance metrics
  - Support statistics

## Project Structure

```
src/
├── Train-Models/
│   ├── model_config.py     # Model hyperparameters and architecture configs
│   ├── model_builder.py    # Model building utilities
│   ├── NN_Model_ML.py     # Money Line prediction model
│   └── NN_Model_UO.py     # Under/Over prediction model
├── Process-Data/
│   ├── feature_engineering.py  # Feature engineering functions
│   ├── Create_Games.py        # Game data processing
│   └── Add_Days_Rest.py       # Rest day calculations
└── Utils/
    └── evaluation.py          # Model evaluation utilities
```

## Packages Used

Use Python 3.11. In particular the packages/libraries used are...

* Tensorflow - Machine learning library
* XGBoost - Gradient boosting framework
* Numpy - Package for scientific computing in Python
* Pandas - Data manipulation and analysis
* Colorama - Color text output
* Tqdm - Progress bars
* Requests - Http library
* Scikit_learn - Machine learning library

## Usage

<img src="https://github.com/kyleskom/NBA-Machine-Learning-Sports-Betting/blob/master/Screenshots/Expected_value.png" width="1010" height="424" />

Make sure all packages above are installed.

```bash
$ git clone https://github.com/kyleskom/NBA-Machine-Learning-Sports-Betting.git
$ cd NBA-Machine-Learning-Sports-Betting
$ pip3 install -r requirements.txt
$ python3 main.py -xgb -odds=fanduel
```

Odds data will be automatically fetched from sbrodds if the -odds option is provided with a sportsbook.  Options include: fanduel, draftkings, betmgm, pointsbet, caesars, wynn, bet_rivers_ny

If `-odds` is not given, enter the under/over and odds for today's games manually after starting the script.

Optionally, you can add '-kc' as a command line argument to see the recommended fraction of your bankroll to wager based on the model's edge

## Flask Web App
<img src="https://github.com/kyleskom/NBA-Machine-Learning-Sports-Betting/blob/master/Screenshots/Flask-App.png" width="922" height="580" />

This repo also includes a small Flask application to help view the data from this tool in the browser.  To run it:
```
cd Flask
flask --debug run
```

## Getting new data and training models
```
# Create dataset with the latest data for 2023-24 season
cd src/Process-Data
python -m Get_Data
python -m Get_Odds_Data
python -m Create_Games

# Train models
cd ../Train-Models
python -m XGBoost_Model_ML
python -m XGBoost_Model_UO
```

## Configuration

### Model Configuration
The project now uses a centralized configuration system in `model_config.py`. Key configurations include:

```python
# Example configuration for Money Line model
ML_CONFIG = {
    'dense_layers': [
        {'units': 512, 'dropout': 0.3},
        {'units': 256, 'dropout': 0.2},
        {'units': 128, 'dropout': 0.1}
    ],
    'l2_reg': 0.001,
    'learning_rate': 0.001
}
```

### Feature Engineering
Configure feature generation in `FEATURE_CONFIG`:

```python
FEATURE_CONFIG = {
    'rolling_windows': [3, 5, 10],
    'relative_metrics': True,
    'streak_features': True,
    'head_to_head': True
}
```

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.
