import argparse
import numpy as np
# Polyaxon
from polyaxon_client.tracking import Experiment

from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier



def load_data():
    data = np.loadtxt("/data/1580fengji.csv",delimiter=",")
    return data[:,:-1],data[:,-1]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--n_estimators',
        type=int,
        default=3)
    parser.add_argument(
        '--max_features',
        type=int,
        default=3
    )
    parser.add_argument(
        '--min_samples_leaf',
        type=int,
        default=80
    )
    args = parser.parse_args()

    # Polyaxon
    experiment = Experiment()

    (X, y) = load_data()
    # Polyaxon
    experiment.log_data_ref(data=X, data_name='dataset_X')
    experiment.log_data_ref(data=y, data_name='dataset_y')

    classifier = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_features=args.max_features,
        min_samples_leaf=args.min_samples_leaf,
    )
    model = classifier.fit(X,y)
    joblib.dump(model,"/outputs/random_forest.md")

