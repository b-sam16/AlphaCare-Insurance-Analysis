import pandas as pd
import scipy.stats as stats

class HypothesisTester:
    def __init__(self, data: pd.DataFrame):
        """
        Initializes the HypothesisTester with a given pandas DataFrame.
        :param data: A pandas DataFrame containing the dataset to analyze.
        """
        self.data = data

    def test_risk_by_group(self, group_col: str, value_col: str, alpha=0.05):
        """
        Performs hypothesis testing (ANOVA or t-test) to evaluate if there are significant differences
        in the value column across different groups in the group column.
        - For more than 2 groups, ANOVA is used.
        - For 2 groups, an independent t-test is used.
        
         """
        try:
            # Identify unique groups in the specified group column
            groups = self.data[group_col].unique()

            # Create a list of data for each group, dropping any NaN values in the value column
            group_data = [self.data[self.data[group_col] == g][value_col].dropna() for g in groups]

            # If there are more than two groups, use ANOVA; otherwise, use an independent t-test
            if len(groups) > 2:
                # ANOVA for multiple groups
                stat, p_value = stats.f_oneway(*group_data)
                test_type = "ANOVA"
            else:
                # Independent t-test for two groups
                stat, p_value = stats.ttest_ind(*group_data, equal_var=False)
                test_type = "t-test"

            # Print the results of the test
            print(f"Test: {test_type}")
            print(f"Statistic: {stat}")
            print(f"P-value: {p_value}")
            # Display a message based on the p-value
            print(f"Message: {self._get_hypothesis_message(p_value, alpha)}")

        except Exception as e:
            # Handle any exceptions that occur during hypothesis testing
            print(f"Error during hypothesis testing: {e}")
    
    def _get_hypothesis_message(self, p_value, alpha=0.05):
        """
        Returns the message indicating whether to accept or reject the null hypothesis
        based on the p-value and the given alpha level.

        """
        if p_value < alpha:
            return "Reject Null Hypothesis: There are significant differences."
        else:
            return "Accept Null Hypothesis: There are no significant differences."
