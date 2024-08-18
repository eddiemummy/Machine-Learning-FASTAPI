from enum import Enum
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB


class Algorithms(Enum):
    LOGISTIC_REGRESSION = "LOGISTIC_REGRESSION"
    KNN = "KNN"
    SVM = "SVM"
    RANDOM_FOREST = "RANDOM_FOREST"
    DECISION_TREE = "DECISION_TREE"
    NAIVE_BAYES = "NAIVE_BAYES"
    RANDOM_FOREST_REG = "RANDOM_FOREST_REG"
    KNN_REG = "KNN_REG"
    SVM_REG = "SVM_REG"
    LINEAR_REGRESSION = "LINEAR_REGRESSION"
    LASSO_REGRESSION = "LASSO_REGRESSION"
    DECISION_TREE_REG = "DECISION_TREE_REG"
    RIDGE_REGRESSION = "RIDGE"


classification_algorithms = {
    Algorithms.LOGISTIC_REGRESSION: LogisticRegression(),
    Algorithms.KNN: KNeighborsClassifier(),
    Algorithms.SVM: SVC(),
    Algorithms.RANDOM_FOREST: RandomForestClassifier(),
    Algorithms.DECISION_TREE: DecisionTreeClassifier(),
    Algorithms.NAIVE_BAYES: GaussianNB(),
}

regression_algorithms = {
    Algorithms.RANDOM_FOREST_REG: RandomForestRegressor(),
    Algorithms.KNN_REG: KNeighborsRegressor(),
    Algorithms.SVM_REG: SVR(),
    Algorithms.LINEAR_REGRESSION: LinearRegression(),
    Algorithms.LASSO_REGRESSION: Lasso(),
    Algorithms.DECISION_TREE_REG: DecisionTreeRegressor(),
    Algorithms.RIDGE_REGRESSION: Ridge(),
}


class AlgorithmType(Enum):
    REGRESSION = "REGRESSION"
    CLASSIFICATION = "CLASSIFICATION"

