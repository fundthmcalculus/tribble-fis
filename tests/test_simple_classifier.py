import pandas as pd
from tribblefis.gauss_data import GaussianMembership, Rule, SimpleGaussianClassifierModel, AnomalyParameters
from tribblefis.gauss_math import simple_gaussian_predict

def test_simple_gaussian_classifier():
    # 1. Define membership functions for two features
    input_mfs = {
        "feature1": [
            GaussianMembership(mu=0.0, sigma=1.0),  # index 0: Low
            GaussianMembership(mu=5.0, sigma=1.0),  # index 1: High
        ],
        "feature2": [
            GaussianMembership(mu=0.0, sigma=1.0),  # index 0: Low
            GaussianMembership(mu=10.0, sigma=1.0), # index 1: High
        ]
    }

    # 2. Define rules
    # Rule 1: IF feature1 is Low AND feature2 is Low THEN Class 0
    # Rule 2: IF feature1 is High AND feature2 is High THEN Class 1
    rules = [
        Rule(antecedents={"feature1": 0, "feature2": 0}, consequent=0),
        Rule(antecedents={"feature1": 1, "feature2": 1}, consequent=1),
    ]

    # 3. Create the model
    model = SimpleGaussianClassifierModel(
        input_mfs=input_mfs,
        rules=rules,
        anomaly_params=AnomalyParameters(include_anomaly=True, threshold=0.1, label="anomaly")
    )

    # 4. Create test data
    data = {
        "feature1": [0.1, 4.9, 100.0], # Low, High, Way out (Anomaly)
        "feature2": [0.2, 10.1, 100.0] # Low, High, Way out (Anomaly)
    }
    df = pd.DataFrame(data)

    # 5. Run prediction
    predictions = simple_gaussian_predict(df, model)

    print(f"Inputs:\n{df}")
    print(f"Predictions: {predictions}")

    # Expected: [0, 1, 'anomaly']
    # Note: simple_gaussian_predict returns np.array, which may coerce types to string if mixed
    assert str(predictions[0]) == "0"
    assert str(predictions[1]) == "1"
    assert str(predictions[2]) == "anomaly"
    print("Test passed!")

if __name__ == "__main__":
    test_simple_gaussian_classifier()
