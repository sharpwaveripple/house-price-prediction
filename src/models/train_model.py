import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from skopt import BayesSearchCV
from skopt.space import Categorical, Integer, Real


def optimize(
    X_train, y_train, X_test, y_test, estimator, search_spaces, n_iter, cv, random_state
):
    opt = BayesSearchCV(
        estimator=estimator,
        search_spaces=search_spaces,
        n_iter=n_iter,
        cv=cv,
        verbose=0,
        n_jobs=4,
        random_state=1,
    )
    opt.fit(X_train, y_train)

    train_score = opt.best_score_ * 100
    test_score = opt.score(X_test, y_test) * 100

    print("Best estimator:\n%s" % opt.best_estimator_)
    print("Validation score: %.1f%%" % train_score)
    print("Test score: %.1f%%" % test_score)

    return opt


# load train and test data
df = pd.read_csv("../../data/processed/train.csv")


X = df[[x for x in df.columns if x != "SalePrice"]]
y = df["SalePrice"]

random_state = 3
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.4,
    random_state=random_state,
)

search_spaces = {
    "learning_rate": Real(0.01, 0.1, "log-uniform"),
    "n_estimators": Integer(100, 1000),
}

estimator = LGBMRegressor(
    objective="regression_l1", random_state=random_state,
)

opt = BayesSearchCV(
    estimator=estimator,
    search_spaces=search_spaces,
    n_iter=100,
    cv=25,
    verbose=0,
    n_jobs=4,
    random_state=random_state,
)
opt.fit(X_train, y_train)

train_score = opt.best_score_ * 100
test_score = opt.score(X_test, y_test) * 100

print("Best estimator:\n%s" % opt.best_estimator_)
print("Validation score: %.1f%%" % train_score)
print("Test score: %.1f%%" % test_score)

score = cross_val_score(opt, X_train, y_train, cv=10)
print(score)
