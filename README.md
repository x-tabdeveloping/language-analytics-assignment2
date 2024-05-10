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

## Results

We can see that the performance of the neural network surpasses the logistic regression classifier by about two percentage points and that it performs generally better on unseen data than LR.
This is expected as the MLP is able to better represent the data (in the hidden layer) before passing it forward to a logistic regression head.
The difference in accuracy, however, is relatively negligable compared to the difference in CO2 emissions and training time, and in a real world setting these might be more important than sheer performance.
Both classifiers performed very reliably, and I would certainly deem them to be usable in a production setting with both of themm boasting F1 scores above 0.9.

## Potential Limitations
No experimentation with the hyperparameters was done in the code, and default parameters were used in almost all scikit-learn estimators.

### Vectorization
TF-IDF weighting was used on bag-of-words representations and stop words were removed as part of the preprocessing.
Reducing the dimensionality of the language representations would probably have benefitted the performance of both classifiers.
This could have been done either by a matrix decomposition step (NMF or SVD), which would account for polysemy and homonymy in the feature space, or simply by reducing the vocabulary.
The feature space could have been reduced by filtering out overly frequent or infrequent terms or by lemmatizing the texts in preprocessing.
These options might result in better model fit and potentially even close the gap between LR and MLP classifiers.

### Model Hyperparameters
Artificial Neural Networks are typically sensitive to the number and size of hidden layers, along with the activation functions in between them.
I have not experimented with these aspects, but thorough and systematic evaluations might have given us a few extra percentage points in performance.
Grid search or Bayesian hyperparameter optimization could be used to find optimal values if maximal performance was neededd.

### Cross-validation
Evaluating over multiple folds (K-folds CV) might have given us a more realistic expectation for what the model might perform like in a real world setting.
If we did not shuffle the dataset before we might even have a reasonable estimate of how domain shift would affect the model in a production setting.
K-fold CV would also englighten us about the uncertainty around the models' performance.


