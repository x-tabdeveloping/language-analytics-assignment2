# language-analytics-assignment2
Assignment 2 for Language Analytics: Classifying fake news with logistic regression and ANNs based on bag-of-words representations.

The data is available on [Kaggle](https://www.kaggle.com/datasets/jillanisofttech/fake-or-real-news).
The downloaded csv file should be placed in a `data` directory:
```
- data/
    fake_or_real_news.csv
```

## Usage

Install requirements:

```bash
pip install -r requirements.txt
```

Run logistic regression benchmark:

```bash
python3 src/logistic_regression.py
```

Run Neural Network benchmark:

```bash
python3 src/neural_network.py
```

Both files save classification reports to the `out/` folder in the form of txt files
and serialize models as pipelines, including the vectorizer with `joblib`.

To load the fitted models in a separate script for inference:

```python
import joblib

classifier = joblib.load("models/logistic_regression.joblib")
classifier.predict(["Write your text here"])
```

> Additionally the scripts will produce csv files with the CO2 emissions of the substasks in the code (`emissions/`).
> This is necessary for Assignment 5, and is not directly relevant to this assignment.

> Note: The `emissions/emissions.csv` file should be ignored. This is due to the fact, that codecarbon can't track process and task emissions at the same time.
