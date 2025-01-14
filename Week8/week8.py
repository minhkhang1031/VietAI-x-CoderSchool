import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectKBest, chi2
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from sympy.physics.vector.printing import params

data = pd.read_excel("job_dataset.ods", engine="odf", dtype="str")
data = data.dropna(axis=0).reset_index(drop=True)

def split_location(location):
    result = location.split(",")
    if len(result) > 1:
        return result[-1].strip(" ")
    else:
        return location

data["location"] = data["location"].apply(split_location)

#split data

x = data.drop("career_level", axis=1)
y = data["career_level"]

#resampling = SMOTE(random_state = 12)
#x,y = resampling.fit_resample(x,y)
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state=12, stratify=y)

#x_train,y_train = resampling.fit_resample(x_train,y_train)

# vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,1)) # (1,1) => unigram. (min, max) => (1,2): unigram and bigram
# processed_data = vectorizer.fit_transform(x_train[["title"]])

param = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [10, 20, None],
    'model__min_samples_split': [2, 5]
}


prepocess = ColumnTransformer(transformers=[
    ("title", TfidfVectorizer(stop_words="english", ngram_range=(1,1)), "title"),
    ("location", OneHotEncoder(handle_unknown="ignore"), ["location"]),
    ("description", TfidfVectorizer(stop_words="english", ngram_range=(1,2), min_df = 0.01, max_df = 0.99), "description"),
    ("function", OneHotEncoder(), ["function"]),
    ("industry", TfidfVectorizer(stop_words="english", ngram_range=(1,1)), "industry")
])

model = Pipeline(steps=[
    ("preprocessor", prepocess),
    ("feature_selector", SelectKBest(chi2, k=400)),
    ("model", RandomForestClassifier(random_state=12))
])

grid_search = GridSearchCV(estimator=model, param_grid=param, cv=3, scoring="accuracy", verbose=1)
grid_search.fit(x_train, y_train)

y_pred = grid_search.best_estimator_.predict(x_test)
print(classification_report(y_test, y_pred))

print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Score:", grid_search.best_score_)