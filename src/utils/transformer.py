"""
Module for Family features Transformer

This transformer creates family-related features from passenger data.
It properly follows the sklearn transformer API with correct fitted attributes.
"""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
import pandas as pd
import numpy as np


class FamilyFeatures(BaseEstimator, TransformerMixin):
    """
    Transformer for creating family-related features from passenger data.

    This transformer creates two new features:
    - family_size: Total family members (SibSp + Parch + 1)
    - isAlone: Binary indicator of traveling alone (family_size == 1)

    The transformer properly stores fitted attributes (ending with underscore)
    so that sklearn recognizes it as fitted after calling fit().

    Parameters
    ----------
    sibsp_col : str, default="SibSp"
        Name of the column containing number of siblings/spouses
    parch_col : str, default="Parch"
        Name of the column containing number of parents/children

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during fit.

    feature_names_in_ : ndarray of shape (n_features_in_,)
        Names of features seen during fit.

    Examples
    --------
    >>> import pandas as pd
    >>> from src.utils.transformer import FamilyFeatures
    >>> X = pd.DataFrame({
    ...     'SibSp': [0, 1, 2],
    ...     'Parch': [0, 1, 0]
    ... })
    >>> transformer = FamilyFeatures()
    >>> transformer.fit(X)
    FamilyFeatures()
    >>> X_transformed = transformer.transform(X)
    >>> X_transformed
       family_size  isAlone
    0            1        1
    1            3        0
    2            3        0
    """

    def __init__(self, sibsp_col="SibSp", parch_col="Parch"):
        """
        Initialize the FamilyFeatures transformer.

        Parameters
        ----------
        sibsp_col : str, default="SibSp"
            Name of the column containing number of siblings/spouses
        parch_col : str, default="Parch"
            Name of the column containing number of parents/children
        """
        self.sibsp_col = sibsp_col
        self.parch_col = parch_col
        self.n_features_in_ = None
        self.feature_names_in_ = None

    def fit(self, X, y=None):
        """
        Fit the transformer by learning from the training data.

        This method validates that the required columns exist in the input
        and stores metadata about the features for later validation.

        Parameters
        ----------
        X : pd.DataFrame
            Training input features. Must contain columns specified by
            sibsp_col and parch_col parameters.
        y : None
            Ignored, present for API consistency with scikit-learn.

        Returns
        -------
        self : FamilyFeatures
            Returns self for method chaining.

        Raises
        ------
        TypeError
            If input is not a pandas DataFrame.
        ValueError
            If required columns are not found in the input.
        """
        # Input validation: Check if X is a DataFrame
        if not isinstance(X, pd.DataFrame):
            print(f"This is X: {X} and this is its shape {X.shape}")
            raise TypeError("This transformer requires a pandas DataFrame. xaha")

        # Validate that required columns exist
        if self.sibsp_col not in X.columns:
            raise ValueError(
                f"Column '{self.sibsp_col}' not found in input. "
                f"Available columns: {list(X.columns)}"
            )
        if self.parch_col not in X.columns:
            raise ValueError(
                f"Column '{self.parch_col}' not found in input. "
                f"Available columns: {list(X.columns)}"
            )

        # IMPORTANT: Store fitted attributes (ending with underscore)
        # This tells sklearn that the transformer is fitted
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.array(X.columns)

        return self

    def transform(self, X):
        """
        Transform the input data by creating family-related features.

        This method uses the learned information from fit() to transform
        new data. It will raise an error if fit() hasn't been called.

        Parameters
        ----------
        X : pd.DataFrame
            Input features containing SibSp and Parch columns.
            Must have the same columns as the data used in fit().

        Returns
        -------
        pd.DataFrame
            DataFrame with two columns:
            - family_size: Total number of family members
            - isAlone: Binary indicator (1 if traveling alone, 0 otherwise)

        Raises
        ------
        TypeError
            If input is not a pandas DataFrame.
        NotFittedError
            If fit() has not been called before transform().
        """
        # Check if transformer has been fitted
        # This looks for attributes ending with underscore
        check_is_fitted(self, ["n_features_in_", "feature_names_in_"])

        # Input validation: Check if X is a DataFrame
        if not isinstance(X, pd.DataFrame):
            raise TypeError("This transformer requires a pandas DataFrame. haha")

        # Make a copy to avoid modifying the original data
        X_copy = X.copy()

        # Calculate features
        family_size = X_copy[self.sibsp_col] + X_copy[self.parch_col] + 1
        is_alone = (family_size == 1).astype(int)

        # Return only the new features as a DataFrame
        # IMPORTANT: When used in ColumnTransformer, only return your columns
        result = pd.DataFrame(
            {
                "family_size": family_size,
                "isAlone": is_alone,
            },
            index=X_copy.index,
        )

        return result