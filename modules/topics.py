from typing import List
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

def extract_topics(chunks: List[str], n_topics: int = 4, n_words: int = 8) -> List[List[str]]:
    vec = CountVectorizer(stop_words="english", max_df=0.95, min_df=2)
    X = vec.fit_transform(chunks)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X)
    vocab = vec.get_feature_names_out()
    topics: List[List[str]] = []
    for comp in lda.components_:
        top_idx = comp.argsort()[-n_words:]
        words = [vocab[i] for i in top_idx]
        topics.append(words)
    return topics
