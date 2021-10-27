import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def understanding_data():
    # iris = datasets.load_iris()
    wines = datasets.load_wine()
    cols = wines.feature_names
    st.write('cols: ', cols, type(cols), len(cols))
    st.write('data: ', wines.data, type(wines.data), wines.data.shape)
    st.write('targets: ', wines.target, type(wines.target), wines.target.shape)
    # pd.Series(wines.target)
    df = pd.DataFrame(data=wines.data, columns=cols)
    df["target"] = wines.target

    st.dataframe(df.describe())
    st.dataframe(df)

    print(df.columns)

def run_lda_vs_pca():
    wines = datasets.load_wine()
    X = wines.data
    y = wines.target
    target_names = wines.target_names
    print('loading....')
    st.write('X: ', X, type(X), len(X))
    st.write('y: ', y, type(y), len(y))
    st.write('target_names: ', target_names, type(target_names), len(target_names))

    pca = PCA(n_components=2)
    X_r = pca.fit(X).transform(X)

    lda = LinearDiscriminantAnalysis(n_components=2)
    X_r2 = lda.fit(X, y).transform(X)

    # Percentage of variance explained for each components
    print(
        "explained variance ratio (first two components): %s"
        % str(pca.explained_variance_ratio_)
    )

    fig, ax = plt.subplots()
    colors = ["navy", "turquoise", "darkorange"]
    lw = 2

    for color, i, target_name in zip(colors, [0, 1, 2], target_names):
        ax.scatter(
            X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=0.8, lw=lw, label=target_name
        )
    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.title("PCA of IRIS dataset")

    plt.figure()
    for color, i, target_name in zip(colors, [0, 1, 2], target_names):
        ax.scatter(
            X_r2[y == i, 0], X_r2[y == i, 1], alpha=0.8, color=color, label=target_name
        )
    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.title("LDA of IRIS dataset")

    plt.show()

if __name__ == '__main__':
    run_lda_vs_pca()
