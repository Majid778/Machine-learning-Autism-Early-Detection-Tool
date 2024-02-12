import sys
import pandas as pd
import numpy as np
from joblib import load

def predict_classes(otu_file, feature_file, model_file):
    # Load OTU data
    data = pd.read_csv(otu_file, sep="\t")
    
    # Load feature selection
    if feature_file.endswith('.txt'):
        selected_feature_names = np.loadtxt(feature_file, dtype=str)
        X = data[selected_feature_names].values  # Use `.values` to match the training conditions
    else:
        print("Feature file must be a .txt file")
        sys.exit(1)

    # Load the trained model
    model = load(model_file)

    # Predict the classes
    predictions = model.predict(X)

    return predictions

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python predict_classes.py otu_file.tsv features_file.txt model_file.joblib")
        sys.exit(1)

    otu_file = sys.argv[1]
    feature_file = sys.argv[2]
    model_file = sys.argv[3]

    # Run predictions
    predicted_classes = predict_classes(otu_file, feature_file, model_file)
    
    # Output the predictions
    for i, pred in enumerate(predicted_classes, 1):
        print(f"Sample {i}: Class {pred}")
