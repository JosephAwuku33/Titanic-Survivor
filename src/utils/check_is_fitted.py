"""
Module for checking if transformers in a pipeline are fitted.

This script demonstrates various methods to inspect the fitted status
of transformers within sklearn pipelines.
"""

import joblib
from sklearn.utils.validation import check_is_fitted

# Load the trained pipeline
pipeline = joblib.load("models/titanic_pipeline.pkl")

# ============================================================================
# METHOD 1: Check the entire pipeline
# ============================================================================
print("=" * 70)
print("METHOD 1: Check if entire pipeline is fitted")
print("=" * 70)

try:
    check_is_fitted(pipeline)
    print("✓ Pipeline is fitted\n")
except AttributeError as e:
    print(f"✗ Pipeline is not fitted: {e}\n")


# ============================================================================
# METHOD 2: Check individual steps in the pipeline
# ============================================================================
print("=" * 70)
print("METHOD 2: Check individual steps in the pipeline")
print("=" * 70)

for step_name, step in pipeline.named_steps.items():
    try:
        check_is_fitted(step)
        print(f"✓ Step '{step_name}' is fitted")
    except AttributeError as e:
        print(f"✗ Step '{step_name}' is not fitted: {e}")

print()


# ============================================================================
# METHOD 3: Check transformers within ColumnTransformer
# ============================================================================
print("=" * 70)
print("METHOD 3: Check transformers within ColumnTransformer")
print("=" * 70)

preprocessor = pipeline.named_steps["preprocessing"]

# Access individual transformers
for name, transformer, columns in preprocessor.transformers_:
    print(f"\nTransformer: '{name}' (columns: {columns})")

    try:
        check_is_fitted(transformer)
        print("✓ Transformer is fitted")
    except AttributeError as e:
        print(f"✗ Transformer is not fitted: {e}")

    # For Pipeline transformers, check each step
    if hasattr(transformer, "named_steps"):
        for sub_step_name, sub_step in transformer.named_steps.items():
            try:
                check_is_fitted(sub_step)
                print(f"    ✓ Sub-step '{sub_step_name}' is fitted")
            except AttributeError as e:
                print(f"    ✗ Sub-step '{sub_step_name}' is not fitted")

print()


# ============================================================================
# METHOD 4: Check fitted attributes directly
# ============================================================================
# print("=" * 70)
# print("METHOD 4: Check fitted attributes directly")
# print("=" * 70)

# preprocessor = pipeline.named_steps['preprocessing']

# for name, transformer, columns in preprocessor.transformers_:
#     print(f"\nTransformer: '{name}'")

#     if hasattr(transformer, 'named_steps'):
#         for sub_step_name, sub_step in transformer.named_steps.items():
#             print(f"  Sub-step: '{sub_step_name}'")

#             # Get attributes that end with underscore (fitted attributes)
#             fitted_attrs = [attr for attr in dir(sub_step)
#                            if attr.endswith('_') and not attr.startswith('_')]

#             if fitted_attrs:
#                 print(f"    Fitted attributes: {fitted_attrs}")
#             else:
#                 print(f"    No fitted attributes found")

# print()


# ============================================================================
# METHOD 5: Display detailed information about each transformer
# ============================================================================
# print("=" * 70)
# print("METHOD 5: Detailed transformer information")
# print("=" * 70)

# preprocessor = pipeline.named_steps['preprocessing']

# for name, transformer, columns in preprocessor.transformers_:
#     print(f"\nTransformer: '{name}'")
#     print(f"  Columns: {columns}")
#     print(f"  Type: {type(transformer).__name__}")

#     if hasattr(transformer, 'named_steps'):
#         print(f"  Sub-steps:")
#         for sub_step_name, sub_step in transformer.named_steps.items():
#             print(f"    - {sub_step_name}: {type(sub_step).__name__}")

#             # Check for common fitted attributes
#             if hasattr(sub_step, 'mean_'):
#                 print(f"      Has mean_: {hasattr(sub_step, 'mean_')}")
#             if hasattr(sub_step, 'scale_'):
#                 print(f"      Has scale_: {hasattr(sub_step, 'scale_')}")
#             if hasattr(sub_step, 'statistics_'):
#                 print(f"      Has statistics_: {hasattr(sub_step, 'statistics_')}")
#             if hasattr(sub_step, 'categories_'):
#                 print(f"      Has categories_: {hasattr(sub_step, 'categories_')}")
#             if hasattr(sub_step, 'n_features_in_'):
#                 print(f"      Has n_features_in_: {hasattr(sub_step, 'n_features_in_')}")

# print()


# ============================================================================
# METHOD 6: Custom helper function for pipeline inspection
# ============================================================================
# print("=" * 70)
# print("METHOD 6: Using a custom helper function")
# print("=" * 70)


# def inspect_pipeline_fitted_status(pipeline):
#     """
#     Inspect and print the fitted status of all transformers in a pipeline.

#     Parameters
#     ----------
#     pipeline : Pipeline
#         The sklearn pipeline to inspect
#     """
#     print(f"Pipeline: {type(pipeline).__name__}")

#     for step_name, step in pipeline.named_steps.items():
#         print(f"\n  Step: '{step_name}'")

#         try:
#             check_is_fitted(step)
#             is_fitted = True
#         except Exception:
#             is_fitted = False

#         print(f"    Fitted: {is_fitted}")

#         # If it's a ColumnTransformer, inspect its transformers
#         if hasattr(step, 'transformers_'):
#             print(f"    Transformers:")
#             for name, transformer, columns in step.transformers_:
#                 try:
#                     check_is_fitted(transformer)
#                     t_fitted = True
#                 except Exception:
#                     t_fitted = False

#                 print(f"      - '{name}': {t_fitted}")


# inspect_pipeline_fitted_status(pipeline)
