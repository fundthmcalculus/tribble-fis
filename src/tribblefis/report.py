import numpy as np
from tribblefis.gauss_data import GaussianMixtureModel


def print_membership_details(model: GaussianMixtureModel):
    # Compute the total number of membership functions by input variable
    per_var_membership_fcns = {
        feature_name: sum(len(label_model.memberships) for label_model in feature_model.label_models.values())
        for feature_name, feature_model in model.feature_models.items()
    }

    print(f"Total membership functions: {sum(per_var_membership_fcns.values())}")
    print(f"Total possible AND-rules: {np.prod(np.array(list(per_var_membership_fcns.values())), dtype=float)}")

    # Get all unique labels from the model structure
    all_labels: set[int] = set()
    for feature_model in model.feature_models.values():
        all_labels.update(feature_model.label_models.keys())

    # Print total possible rules for each label
    for label in sorted(all_labels):
        per_var_membership_fcns_for_label = {
            feature_name: len(feature_model.label_models[label].memberships) if label in feature_model.label_models else 0
            for feature_name, feature_model in model.feature_models.items()
        }
        print(
            f"Total possible rules for label={label}: {np.prod(np.array(list(per_var_membership_fcns_for_label.values())), dtype=float)}"
        )
    print("Membership functions:", per_var_membership_fcns)


def print_gaussian_memberships(model):
    print("\nMembership Functions Dictionary:")
    print("=" * 80)
    for feature_name, feature_model in model.feature_models.items():
        print(f"\n{feature_name}:")
        for label, label_model in feature_model.label_models.items():
            print(f"  Label {label}: {len(label_model.memberships)} membership functions")
            for i, mf in enumerate(label_model.memberships):
                print(f"    MF {i + 1}: {mf}")
    print("=" * 80)
