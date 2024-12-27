# src/visualization.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class Visualizer:
    def __init__(self, data):
        """
        Initialize the Visualizer class with a DataFrame.
        
        Parameters:
        - data (pd.DataFrame): The dataset to visualize.
        """
        self.data = data
        self.fig_size = (8, 4)  # Default smaller figure size for faster loading
    
    def plot_numerical_distribution(self):
        """
        Plot histograms for all numerical columns in the dataset.
        """
        numerical_cols = self.data.select_dtypes(include=['int64', 'float64']).columns
        
        for col in numerical_cols:
            plt.figure(figsize=self.fig_size)
            sns.histplot(self.data[col], kde=True, bins=30, color='skyblue')
            plt.title(f'Distribution of {col}')
            plt.xlabel(col)
            plt.ylabel('Frequency')
            plt.show()
    
    def plot_categorical_distribution(self):
        """
        Plot bar charts for all categorical columns in the dataset.
        """
        categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            plt.figure(figsize=self.fig_size)
            value_counts = self.data[col].value_counts().head(10)  # Show top 10 categories
            sns.barplot(x=value_counts.index, y=value_counts.values, palette='viridis')
            plt.title(f'Distribution of {col}')
            plt.xlabel(col)
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.show()


    def plot_totalpremium_vs_totalclaims_by_postalcode(self):
        """
        Scatter plot of TotalPremium vs TotalClaims grouped by PostalCode.
        """
        if {'TotalPremium', 'TotalClaims', 'PostalCode'}.issubset(self.data.columns):
            plt.figure(figsize=(10, 6))
            sns.scatterplot(
                data=self.data,
                x='TotalPremium',
                y='TotalClaims',
                hue='PostalCode',
                palette='viridis',
                alpha=0.6
            )
            plt.title('TotalPremium vs TotalClaims by PostalCode')
            plt.xlabel('Total Premium')
            plt.ylabel('Total Claims')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.show()
        else:
            raise ValueError("Columns 'TotalPremium', 'TotalClaims', and 'PostalCode' must exist in the dataset.")

    def plot_correlation_matrix(self):
        """
                Plot a correlation matrix for numerical variables including TotalPremium and TotalClaims.
        """
        numerical_columns = self.data[['TotalPremium', 'TotalClaims']].select_dtypes(include='number')
        correlation_matrix = numerical_columns.corr()
        
        plt.figure(figsize=(6, 5))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Matrix: TotalPremium & TotalClaims')
        plt.show()


    def plot_premium_trend(self):
        """Compare Total Premium trends across regions over time."""
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=self.data, x='TransactionMonth', y='TotalPremium', hue='Province', marker='o')
        plt.title('Trends in Total Premium Over Time by Region')
        plt.xlabel('Transaction Month')
        plt.ylabel('Total Premium')
        plt.legend(title='Province', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()
    
    def plot_cover_type_trend(self):
        """Compare Insurance Cover Type trends across regions over time."""
        # Aggregating CoverType if necessary (e.g., counts)
        cover_type_trend = (
            self.data.groupby(['TransactionMonth', 'Province'])['CoverType']
            .count()
            .reset_index()
        )
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=cover_type_trend, x='TransactionMonth', y='CoverType', hue='Province', marker='o')
        plt.title('Trends in Cover Type Over Time by Region')
        plt.xlabel('Transaction Month')
        plt.ylabel('Cover Type Count')
        plt.legend(title='Province', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()
    
    def plot_auto_make_trend(self):
        """Compare Auto Make trends across regions over time."""
        # Aggregating make if necessary (e.g., counts)
        auto_make_trend = (
            self.data.groupby(['TransactionMonth', 'Province'])['make']
            .count()
            .reset_index()
        )
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=auto_make_trend, x='TransactionMonth', y='make', hue='Province', marker='o')
        plt.title('Trends in Auto Make Over Time by Region')
        plt.xlabel('Transaction Month')
        plt.ylabel('Auto Make Count')
        plt.legend(title='Province', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

    def plot_outlier_detection(self):
        """
        Plot box plots for numerical columns to detect outliers.
        """
        numerical_columns = self.data.select_dtypes(include=['number']).columns
        
        fig, axes = plt.subplots(nrows=len(numerical_columns), ncols=1, figsize=(8, len(numerical_columns) * 3))
        for ax, col in zip(axes, numerical_columns):
            sns.boxplot(x=self.data[col], ax=ax, color='skyblue')
            ax.set_title(f'Outlier Detection for {col}')
        plt.tight_layout()
        plt.show()