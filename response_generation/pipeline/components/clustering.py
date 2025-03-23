from abc import ABC
from typing import List

import nltk
import numpy as np
import pandas as pd
from bertopic import BERTopic
from hdbscan import HDBSCAN
from nltk import word_tokenize  # tokenizing
from nltk.stem.wordnet import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from umap import UMAP

_DEFAULT_SENTENCE_TRANSFORMERS_MODEL = "all-MiniLM-L6-v2"

stop_words = list(nltk.corpus.stopwords.words("english"))


def text_preprocessing(text: str):
    """Preprocess the given text.

    Args:
        text: The text to preprocess.

    Returns:
        The lemmatized text with stopwords removed.
    """
    le = WordNetLemmatizer()
    word_tokens = word_tokenize(text)
    tokens = [
        le.lemmatize(w)
        for w in word_tokens
        if w not in stop_words and len(w) > 3
    ]
    cleaned_text = " ".join(tokens)
    return cleaned_text


class Clustering(ABC):
    def __init__(self):
        self._clustering_model = None

    def cluster(self, texts: List[str]) -> pd.DataFrame:
        """Cluster the given texts.

        Args:
            texts (List[str]): The texts to cluster.

        Returns:
            pd.DataFrame: A DataFrame with the cluster labels.
        """


class LSAClustering(Clustering):
    def __init__(self):
        super().__init__()

    def cluster(self, texts: List[str]) -> pd.DataFrame:
        """Cluster the given texts using LSA.

        Args:
            texts (List[str]): The texts to cluster.

        Returns:
            pd.DataFrame: A DataFrame with the cluster labels.
        """
        processed_texts = [text_preprocessing(text) for text in texts]
        vect = TfidfVectorizer(stop_words=stop_words, max_features=1000)
        vect_text = vect.fit_transform(processed_texts)

        lsa_model = TruncatedSVD(
            n_components=int(len(texts) / 2),
            algorithm="randomized",
            n_iter=10,
            random_state=42,
        )
        lsa_matrix = lsa_model.fit_transform(vect_text)

        topics = []
        probabilities = []

        for doc_stats in lsa_matrix:
            topic = np.argmax(doc_stats)
            topics.append(topic)
            probabilities.append(doc_stats[topic])

        return pd.DataFrame(
            {"Document": texts, "Topic": topics, "Probability": probabilities}
        )


class BERTopicClustering(Clustering):
    def __init__(
        self,
        n_neighbors: int = 50,
        n_components: int = 2,
        min_dist: float = 0.01,
        min_cluster_size: int = 2,
        min_samples: int = 1,
    ):
        """Initialize the BERTopic clustering model.

        Args:
            n_neighbors (int, optional): The number of neighbors to consider. Defaults to 50.
            n_components (int, optional): The number of components. Defaults to 2.
            min_dist (float, optional): The minimum distance. Defaults to 0.01.
            min_cluster_size (int, optional): The minimum cluster size. Defaults to 2.
            min_samples (int, optional): The minimum number of samples. Defaults to 1.
        """
        vectorizer_model = CountVectorizer(
            ngram_range=(1, 2), stop_words="english"
        )
        embedding_model = SentenceTransformer(
            _DEFAULT_SENTENCE_TRANSFORMERS_MODEL
        )
        umap_model = UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,  # 3
            min_dist=min_dist,
            random_state=42,
        )
        hdbscan_model = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            gen_min_span_tree=True,
            prediction_data=True,
        )

        self._clustering_model = BERTopic(
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            embedding_model=embedding_model,
            vectorizer_model=vectorizer_model,
            language="english",
            calculate_probabilities=True,
            verbose=True,
        )

    def cluster(self, texts: List[str]) -> pd.DataFrame:
        """Cluster the given texts using BERTopic.

        Args:
            texts (List[str]): The texts to cluster.

        Returns:
            pd.DataFrame: A DataFrame with the cluster labels.
        """
        _, _ = self._clustering_model.fit_transform(texts)
        return self._clustering_model.get_document_info(texts)


if __name__ == "__main__":
    # Example of how to use the clustering components
    information_nuggets = pd.read_csv(
        "data/input_passages/5_relevant-cast-snippets.csv"
    )

    bertopic = BERTopicClustering()
    lsa = LSAClustering()

    bertopic_topic_clusters_count = []
    lsa_topic_clusters_count = []
    queries_with_no_nuggets = []
    number_of_nuggets = []

    for i, info_nugget in information_nuggets.iterrows():
        query_query_id = info_nugget["query_id"] + "-" + info_nugget["query"]
        query_information_nuggets = list(info_nugget["information_nuggets"])
        number_of_nuggets.append(len(query_information_nuggets))
        if len(query_information_nuggets) >= 4:
            bertopic_freq = bertopic.cluster(query_information_nuggets)
            bertopic_topic_clusters_count.append(len(bertopic_freq))
            lsa_freq = lsa.cluster(query_information_nuggets)
            lsa_topic_clusters_count.append(len(lsa_freq))
        else:
            queries_with_no_nuggets.append(query_query_id)
            print("--- Not enough nuggets ---")

    print(
        "Average number of topics clustered with BERTopic: "
        + str(
            sum(bertopic_topic_clusters_count)
            / len(bertopic_topic_clusters_count)
        )
    )
    print(
        "Average number of topics clustered with LSA: "
        + str(sum(lsa_topic_clusters_count) / len(lsa_topic_clusters_count))
    )
    print(
        "Number of queries with not enough nuggets: "
        + str(len(queries_with_no_nuggets))
    )
    print("Number of nuggets: " + str(len(number_of_nuggets)))
