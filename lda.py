import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
# from sklearn.metrics import accuracy_score #same as classifier.score

def understanding_data(dataset):
    cols = dataset.feature_names
    st.write('cols: ', cols, type(cols), len(cols))
    st.write('data: ', dataset.data, type(dataset.data), dataset.data.shape)
    st.write('targets: ', dataset.target, type(dataset.target), dataset.target.shape)
    # pd.Series(wines.target)
    df = pd.DataFrame(data=dataset.data, columns=cols)
    df["target"] = dataset.target
    st.write("Dataframe: data + target")
    st.dataframe(df.describe())
    st.dataframe(df)
    # st.write('Columns: ', df.columns.tolist())

########################################################################################################################
def run_lda(dataset, title):
    X = dataset.data
    y = dataset.target
    target_names = dataset.target_names
    st.write('target_names: ', target_names, type(target_names), len(target_names))
    st.write("Counter y: ", Counter(y).items())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    sc = StandardScaler()
    sc_X_train = sc.fit_transform(X_train)
    sc_X_test = sc.transform(X_test)

    lda = LinearDiscriminantAnalysis(n_components=n_comp)
    lda_X_train = lda.fit(sc_X_train, y_train).transform(sc_X_train)
    lda_X_test = lda.transform(sc_X_test)

    # Percentage of variance explained for each components
    st.write(
        "explained variance ratio (first two components): %s"
        % str(lda.explained_variance_ratio_)
        )

    # Generating colors
    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0., 1., len(target_names)))

    # Generating indexes for the colors
    colors_indexes = list(range(0, len(target_names)))
    lw = 2

    fig = plt.figure(figsize=fig_size)
    for color, i, target_name in zip(colors, colors_indexes, target_names):
        plt.scatter(
            lda_X_train[y_train == i, 0], lda_X_train[y_train == i, 1], color=color, edgecolors='black', alpha=0.75,
            s=gs_dot, lw=lw, label=target_name
        )
    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.title("LDA of " + title + " training dataset")
    st.pyplot(fig)

    fig = plt.figure(figsize=fig_size)
    for color, i, target_name in zip(colors, colors_indexes, target_names):
        plt.scatter(
            lda_X_test[y_test == i, 0], lda_X_test[y_test == i, 1], color=color, edgecolors='black', alpha=0.75,
            s=gs_dot, lw=lw, label=target_name
            )
    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.title("LDA of " + title + " test dataset")
    st.pyplot(fig)

    # st.write("================== LDA ========================")
    # lda_run = LinearDiscriminantAnalysis()
    # lda_run.fit(X_train, y_train)
    # score_lda = lda.score(X_test, y_test)
    # st.write('score: ', format(score_lda, ".2%"))
    # st.write("===============================================")

    return lda_X_train, lda_X_test, y_train, y_test

########################################################################################################################
def run_pca(dataset, title):
    X = dataset.data
    y = dataset.target
    target_names = dataset.target_names
    st.write('target_names: ', target_names, type(target_names), len(target_names))
    st.write("Counter y: ", Counter(y).items())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    sc = StandardScaler()
    sc_X_train = sc.fit_transform(X_train)
    sc_X_test = sc.transform(X_test)

    pca = PCA(n_components=n_comp)
    pca_X_train = pca.fit_transform(sc_X_train)
    pca_X_test = pca.transform(sc_X_test)

    # Percentage of variance explained for each components
    st.write(
        "explained variance ratio (first two components): %s"
        % str(pca.explained_variance_ratio_)
    )

    # Generating colors
    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0., 1., len(target_names)))

    # Generating indexes for the colors
    colors_indexes = list(range(0, len(target_names)))
    lw = 2

    fig = plt.figure(figsize=fig_size)
    for color, i, target_name in zip(colors, colors_indexes, target_names):
        plt.scatter(
            pca_X_train[y_train == i, 0], pca_X_train[y_train == i, 1], color=color,  edgecolors='black', alpha=0.75,
            s=gs_dot, lw=lw, label=target_name
        )
    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.title("PCA of " + title + " training dataset")
    st.pyplot(fig)

    fig = plt.figure(figsize=fig_size)
    for color, i, target_name in zip(colors, colors_indexes, target_names):
        plt.scatter(
            pca_X_test[y_test == i, 0], pca_X_test[y_test == i, 1], color=color,  edgecolors='black', alpha=0.75,
            s=gs_dot, lw=lw, label=target_name
            )
    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.title("PCA of " + title + " test dataset")
    st.pyplot(fig)

    # st.write("================== PCA ========================")
    # pca_run = PCA()
    # pca_run.fit(X_train, y_train)
    # score_pca = pca_run.score(X_test, y_test)
    # st.write('PCA score: ', format(score_pca, ".2%"))
    # st.write("===============================================")

    return pca_X_train, pca_X_test, y_train, y_test

########################################################################################################################
def run_ica(dataset, title):
    X = dataset.data
    y = dataset.target
    target_names = dataset.target_names
    st.write('target_names: ', target_names, type(target_names), len(target_names))
    st.write("Counter y: ", Counter(y).items())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    sc = StandardScaler()
    sc_X_train = sc.fit_transform(X_train)
    sc_X_test = sc.transform(X_test)

    ica = FastICA(n_components=n_comp)
    ica_X_train = ica.fit_transform(sc_X_train)
    ica_X_test = ica.transform(sc_X_test)

    # Generating colors
    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0., 1., len(target_names)))

    # Generating indexes for the colors
    colors_indexes = list(range(0, len(target_names)))
    lw = 2

    fig = plt.figure(figsize=fig_size)
    for color, i, target_name in zip(colors, colors_indexes, target_names):
        plt.scatter(
            ica_X_train[y_train == i, 0], ica_X_train[y_train == i, 1], color=color,  edgecolors='black', alpha=0.75,
            s=gs_dot, lw=lw, label=target_name
        )
    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.title("ICA of " + title + " training dataset")
    st.pyplot(fig)

    fig = plt.figure(figsize=fig_size)
    for color, i, target_name in zip(colors, colors_indexes, target_names):
        plt.scatter(
            ica_X_test[y_test == i, 0], ica_X_test[y_test == i, 1], color=color,  edgecolors='black', alpha=0.75,
            s=gs_dot, lw=lw, label=target_name
            )
    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.title("ICA of " + title + " test dataset")
    st.pyplot(fig)

    # st.write("================== ICA ========================")
    # ica_run = ICA()
    # ica_run.fit(X_train, y_train)
    # st.write("===============================================")

    return ica_X_train, ica_X_test, y_train, y_test

########################################################################################################################
def predict_rfr(X_train, y_train, X_test, y_test):
    classifier = RandomForestClassifier(max_depth=2, random_state=42)
    classifier.fit(X_train, y_train)

    score = classifier.score(X_test, y_test)
    st.write('score: ', format(score, ".2%"))
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    # st.write('prediction: ', format(y_pred, ".2%"))
    # st.write('prediction: ', list(map('{:.2f}%'.format, y_pred)))

    cm = confusion_matrix(y_test, y_pred)
    st.write('confusion matrix: ', cm)
    # st.write('Accuracy:', format(accuracy_score(y_test, y_pred), ".2%")) # same as classifier.score

########################################################################################################################
def analysis(dataset, title):
    understanding_data(dataset)
    pca_X_train, pca_X_test, y_train, y_test = run_pca(dataset, title)
    st.write("################# PCA ########################")
    predict_rfr(pca_X_train, y_train, pca_X_test, y_test)

    ica_X_train, ica_X_test, y_train, y_test = run_ica(dataset, title)
    st.write("################# ICA ########################")
    predict_rfr(ica_X_train, y_train, ica_X_test, y_test)

    lda_X_train, lda_X_test, y_train, y_test = run_lda(dataset, title)
    st.write("################# LDA ########################")
    predict_rfr(lda_X_train, y_train, lda_X_test, y_test)
    st.write("##############################################")

########################################################################################################################
if __name__ == '__main__':
    # defining parameters
    n_comp = 2
    fig_size = (5, 3)
    gs_dot = 20

    # Analysis runned in different datasets
    dataset = datasets.load_wine()
    title = 'Wine'
    analysis(dataset, title)

    dataset = datasets.load_iris()
    title = 'IRIS'
    analysis(dataset, title)

    dataset = datasets.load_digits()
    title = 'Digits'
    analysis(dataset, title)
