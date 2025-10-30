# src/bias_debugger/visuals.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.figure import Figure

sns.set_theme(style="whitegrid")

def plot_group_outcomes(
    y_pred: pd.Series,
    sensitive_features: pd.Series,
    title: str = "Rate of Favorable Outcomes by Group"
) -> Figure:
    """
    Creates a bar plot showing the % of favorable outcomes (y_pred=1)
    for each sensitive group. This visualizes Disparate Impact.
    """
    df = pd.DataFrame({
        'Group': sensitive_features,
        'Favorable_Outcome': y_pred
    })
    
    rates = df.groupby('Group')['Favorable_Outcome'].mean().reset_index()

    fig, ax = plt.subplots()
    sns.barplot(
        data=rates,
        x='Group',
        y='Favorable_Outcome',
        ax=ax,
        palette="viridis"
    )
    ax.set_title(title)
    ax.set_ylabel("Favorable Outcome Rate")
    ax.set_ylim(0, 1)
    
    # Add labels to bars
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2%}',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center',
                    xytext=(0, 9),
                    textcoords='offset points')
    
    return fig

def plot_tpr_by_group(
    y_true: pd.Series,
    y_pred: pd.Series,
    sensitive_features: pd.Series,
    title: str = "True Positive Rate (TPR) by Group"
) -> Figure:
    """
    Creates a bar plot showing the TPR for each sensitive group.
    This visualizes Equal Opportunity Difference.
    """
    df = pd.DataFrame({
        'Group': sensitive_features,
        'y_true': y_true,
        'y_pred': y_pred
    })
    
    # Filter for only actual positives (y_true == 1)
    positives_df = df[df['y_true'] == 1]
    
    # Calculate TPR (mean of y_pred *for actual positives*)
    tpr_rates = positives_df.groupby('Group')['y_pred'].mean().reset_index()
    tpr_rates.columns = ['Group', 'True_Positive_Rate']

    fig, ax = plt.subplots()
    sns.barplot(
        data=tpr_rates,
        x='Group',
        y='True_Positive_Rate',
        ax=ax,
        palette="plasma"
    )
    ax.set_title(title)
    ax.set_ylabel("True Positive Rate")
    ax.set_ylim(0, 1)
    
    # Add labels to bars
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2%}',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center',
                    xytext=(0, 9),
                    textcoords='offset points')
    
    return fig