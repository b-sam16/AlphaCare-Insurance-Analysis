import sys
sys.path.append('./src')
import pytest
import pandas as pd
import os
from eda import EDAHandler  # Assuming your class is in 'eda.py'

# Create a fixture to save the test data
@pytest.fixture
def sample_data():
    # Define the data
    data = {
        'Age': [23, 45, 25, 35, None],
        'Gender': ['M', 'F', 'M', 'M', 'F'],
        'Income': [50000, 60000, 55000, None, 70000],
        'TransactionMonth': ['2022-01-01', '2022-02-01', '2022-03-01', '2022-04-01', '2022-05-01']
    }
    df = pd.DataFrame(data)
    
    # Ensure the folder exists
    os.makedirs('data/test_data', exist_ok=True)
    
    # Save the data to the test_data folder
    file_path = 'data/test_data/test_data.csv'
    df.to_csv(file_path, index=False)
    
    return file_path

# Test initialization and loading data
def test_initialization(sample_data):
    eda = EDAHandler(sample_data)
    assert isinstance(eda.data, pd.DataFrame)
    assert len(eda.data) == 5  # Ensure 5 rows are loaded
    assert 'Age' in eda.data.columns  # Ensure 'Age' column exists

# Test descriptive statistics calculation
def test_descriptive_statistics(sample_data):
    eda = EDAHandler(sample_data)
    numerical_stats, categorical_stats = eda.descriptive_statistics()

    # Check if descriptive stats are calculated for numerical columns
    assert 'Age' in numerical_stats.columns
    assert 'Income' in numerical_stats.columns
    assert 'count' in numerical_stats.index  # Check if count is included

    # Check if categorical stats are calculated for non-numerical columns
    assert 'Gender' in categorical_stats.columns
    assert 'count' in categorical_stats.index  # Check if count is included

# Test missing values handling
def test_missing_values(sample_data):
    eda = EDAHandler(sample_data)
    missing_data = eda.missing_values()
    assert missing_data['Age'] == 1  # One missing value in 'Age' column
    assert missing_data['Income'] == 1  # One missing value in 'Income' column

# Test data structure
def test_data_structure(sample_data):
    eda = EDAHandler(sample_data)
    data_types = eda.data_structure()
    assert data_types['Age'] == 'float64'
    assert data_types['Gender'] == 'object'
    assert data_types['TransactionMonth'] == 'object'  # Initially string

# Test clean_data functionality
def test_clean_data(sample_data):
    eda = EDAHandler(sample_data)
    eda.clean_data()

    # Check that the missing values are cleaned
    missing_data = eda.missing_values()
    assert 'Age' not in missing_data  # Should have no missing values after imputation
    assert 'Income' not in missing_data  # Should have no missing values after imputation

    # Check that 'TransactionMonth' is now a datetime type
    assert pd.api.types.is_datetime64_any_dtype(eda.data['TransactionMonth'])

# Test handling outliers using IQR
def test_handle_outliers_iqr(sample_data):
    eda = EDAHandler(sample_data)
    
    # Manually introduce some outliers for testing purposes
    eda.data.loc[0, 'Age'] = 100  # Outlier
    eda.data.loc[4, 'Income'] = 1e6  # Outlier
    
    # Handle outliers
    cleaned_data = eda.handle_outliers_iqr()

    # Check that the outliers are removed
    assert cleaned_data['Age'].max() <= 75  # Should be less than the outlier
    assert cleaned_data['Income'].max() <= 75000  # Should be less than the outlier

# Test clean_data edge case where no missing values
def test_clean_data_no_missing(sample_data):
    # Remove missing data to simulate no missing values
    data = {
        'Age': [23, 45, 25, 35],
        'Gender': ['M', 'F', 'M', 'M'],
        'Income': [50000, 60000, 55000, 70000],
        'TransactionMonth': ['2022-01-01', '2022-02-01', '2022-03-01', '2022-04-01']
    }
    df = pd.DataFrame(data)
    os.makedirs('data/test_data', exist_ok=True)
    df.to_csv('data/test_data/test_data_no_missing.csv', index=False)
    
    eda = EDAHandler('data/test_data/test_data_no_missing.csv')
    eda.clean_data()
    
    # Assert that no columns are dropped or imputed
    assert eda.missing_values().empty  # Should be no missing values
    assert 'Age' in eda.data.columns  # 'Age' should still be present
    assert 'Income' in eda.data.columns  # 'Income' should still be present
