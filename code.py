
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def read_and_clean_data(file_path):
    """Reads and cleans data from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: The cleaned DataFrame, or None if an error occurs.
    """

    try:
        df = pd.read_csv(file_path)
        df.rename(columns={"User Rating": "User_Rating"}, inplace=True)
        df.loc[df.Author == 'J. K. Rowling', 'Author'] = 'J.K. Rowling'
        df['name_len'] = df['Name'].apply(lambda x: len(x) - x.count(" "))
        # ... other cleaning steps
        return df
    except FileNotFoundError:
        print("Error: File not found!")
        return None


def analyze_top_authors(cleaned_df):
    """Analyzes and plots the top authors in the data.

    Args:
        cleaned_df (pd.DataFrame): The cleaned DataFrame.
    """

    n_best = 20  # Number of top authors to consider

    # Get top n_best authors by appearance count
    top_authors = cleaned_df.Author.value_counts().nlargest(n_best)

    # Remove rows with duplicate book names
    no_dup = cleaned_df.drop_duplicates('Name')

    # Create a figure with 3 subplots
    fig, ax = plt.subplots(1, 3, figsize=(11, 10), sharey=True)

    # Color palette for lines and bars
    color = sns.color_palette("hls", n_best)

    # ---------- Subplot 1: Appearances ----------
    ax[0].hlines(y=top_authors.index, xmin=0, xmax=top_authors.values, color=color, linestyles='dashed')
    ax[0].plot(top_authors.values, top_authors.index, 'go', markersize=9)
    ax[0].set_xlabel('Number of Appearances')
    ax[0].set_xticks(np.arange(top_authors.values.max() + 1))  # Adjust for max count
    ax[0].set_yticklabels(top_authors.index, fontweight='semibold')
    ax[0].set_title('Appearences')

    # ---------- Subplot 2: Unique Books ----------
    book_count = []
    for name, col in zip(top_authors.index, color):
        book_count.append(len(no_dup[no_dup.Author == name]['Name']))

    ax[1].hlines(y=top_authors.index, xmin=0, xmax=book_count, color=color, linestyles='dashed')
    ax[1].plot(book_count, top_authors.index, 'go', markersize=9)
    ax[1].set_xlabel('Number of Unique Books')
    ax[1].set_xticks(np.arange(max(book_count) + 1))  # Adjust for max count
    ax[1].set_title('Unique Books')

    # ---------- Subplot 3: Total Reviews ----------
    total_reviews = []
    for name in top_authors.index:  # Assuming you have access to the 'Reviews' column
        total_reviews.append(no_dup[no_dup.Author == name]['Reviews'].sum() / 1000)

    ax[2].barh(y=top_authors.index, width=total_reviews, color=color, edgecolor='black', height=0.7)
    for name, val in zip(top_authors.index, total_reviews):
        ax[2].text(val + 2, name, f"{val:.1f}k")  # Display reviews with one decimal place and 'k' for thousands
    ax[2].set_xlabel("Total Reviews (in 1,000's)")
    ax[2].set_title('Total Reviews')

    # Improve readability (optional)
    plt.tight_layout()

    plt.show()


# Replace with the correct file path
file_path = "C:/Users/super/Downloads/bestsellers with categories.csv"
cleaned_df = read_and_clean_data(file_path)

if cleaned_df is not None:
    analyze_top_authors(cleaned_df)
