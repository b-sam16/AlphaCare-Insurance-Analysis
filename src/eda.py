import pandas as pd

class EDAHandler:
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)

    def descriptive_statistics(self):
        # Select numerical columns
        numerical_columns = self.data.select_dtypes(include=['number']).columns
        numerical_stats = self.data[numerical_columns].describe()

        # Select categorical columns
        categorical_columns = self.data.select_dtypes(include=['object']).columns
        categorical_stats = self.data[categorical_columns].describe()

        return numerical_stats, categorical_stats

    def data_structure(self):
        # Review the dtype of each column to ensure proper formatting
        return self.data.dtypes
    
    def missing_values(self):
        # Check for missing values across the dataset
        missing_data = self.data.isnull().sum()
        missing_data = missing_data[missing_data > 0]  # Only show columns with missing data
        return missing_data
# Example usage
#eda = EDAHandler('data/Insurance_data.csv')
