# language-analytics-assignment2
Assignment 2 for Language Analytics: Classifying fake news with logistic regression and ANNs.

## Usage

Install requirements:

```bash
pip install -r requirements.txt
```

Run logistic regression benchmark:

```bash
python3 logistic_regression.py
```

Run Neural Network benchmark:

```bash
python3 neural_network.py
```

Both files save reports to the `out/` folder in the form of txt files
and serialize models as pipelines, including the vectorizer with `joblib`.

To load the fitted models in a separate script for inference:

```python
import joblib

classifier = joblib.load("models/logistic_regression.joblib")
classifier.predict(["Write your text here"])
```
