# Andrew Bae
# ITP-216
# Final Project
# Analyzes video game sales from 1980 to 2020 and predicts a game's genre based on its regional sales

# Source: https://www.kaggle.com/datasets/gregorut/videogamesales?resource=download

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

def main():
    print("**USER INPUT SECTION**")

    # Loads in file and checks first few rows
    df_vg = pd.read_csv('vgsales.csv')

    # Drops empty rows
    df_vg = df_vg.dropna()

    # Filters out non-float Year rows
    num_year_filter = df_vg['Year'].map(type) == float
    df_vg = df_vg[num_year_filter]

    # Converts remaining rows' Year to int
    df_vg['Year'] = df_vg['Year'].astype(int)

    # User input through the command line
    genre = input("Enter a genre: ") # Action, Adventure, Sports, Platform, Racing, Role-Playing, Puzzle, Shooter, Fighting, Simulation, Music, Misc
    platform = input("Enter a platform: ") # NES, SNES, GB, N64, GBA, GC, DS, Wii, 3DS, WiiU, PS, PS2, PSP, PS3, PSV, PS4, XB, X360, XOne, SAT, SCD, WS, NG, TG16, 3DO, GG, PCFX
    min_year = int(input("Enter a minimum year: "))
    max_year = int(input("Enter a maximum year: "))

    # Creates df subset based on user's entered criteria; .lower() ensure case-insensitive
    df_search_filter = (df_vg['Genre'].str.lower() == genre.lower()) & (df_vg['Platform'].str.lower() == platform.lower()) & (df_vg['Year'] >= min_year) & (df_vg['Year'] <= max_year)
    vg_subset = df_vg[df_search_filter]

    # Aggregates subset by sales per year
    vg_subset = vg_subset.groupby('Year')
    yearly_sales = vg_subset['Global_Sales'].sum()

    # Plots data in a bar chart
    fig, ax = plt.subplots()
    ax.bar(yearly_sales.index, yearly_sales.values)

    # Labels the axes and title
    ax.set_xlabel('Year')
    ax.set_ylabel('Global Sales (in Millions)')
    plot_title = 'Total ' + genre + ' Video Game Sales on ' + platform + ' by Release Year'
    ax.set_title(plot_title)

    # Adjusts ticks to only show whole years
    ax.set_xticks(yearly_sales.index)

    # Displays the plot in a png
    print("Generating plot of sales...")
    plt.savefig('annual_sales.png')

    print()

    print("**MACHINE LEARNING SECTION**")

    # Trains a model to predict a game's genre based on regional and global sales
    # Establishes feature vector and target
    X = df_vg[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']]
    y = df_vg['Genre']

    # Splits the data into testing and training sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)

    # Transforms the data using StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = pd.DataFrame(scaler.transform(X_train), columns=X.columns, index=X_train.index)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)

    # Creates a model and trains it
    model_knn = KNeighborsClassifier(n_neighbors=3)
    model_knn.fit(X_train, y_train)

    # Predicts on a single sample to demonstrate
    X_new = pd.DataFrame([[6.16, 3.4, 1.2, 0.76, 11.52]], columns=X.columns)
    X_new = pd.DataFrame(scaler.transform(X_new), columns=X.columns)
    sample_prediction = model_knn.predict(X_new)
    # It predicts 'Misc' but these numbers are from a 'Platform' game, so the model is incorrect on this prediction
    print("Prediction for sample entry:", sample_prediction)

    # Predicts the testing subset based on the model and evaluates the accuracy
    y_pred = model_knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Model accuracy:", accuracy)

    # Creates the confusion matrix and a ConfusionMatrixDisplay object
    print("Generating confusion matrix...")
    conf_matrix = confusion_matrix(y_test, y_pred)
    matrix_display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=model_knn.classes_)

    # Plots the confusion matrix
    fig, axes = plt.subplots()
    matrix_display.plot(ax=axes)

    # Fixes overlapping labels on the x-axis by angling them
    plt.setp(axes.get_xticklabels(), rotation=45, ha='right')

    # Displays the plot in a png
    axes.set_title('KNN Video Game Genre Confusion Matrix')
    fig.savefig('confusion_matrix.png')

if __name__ == '__main__':
    main()
