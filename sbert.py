from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt

from comments_summarize_ollama_bash import generate_comments_list

def clustering_comments(comments:list, model_name:str, num_clusters:int):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(comments, convert_to_tensor=True)
    clustering_model = KMeans(n_clusters=num_clusters)

    clustering_model.fit(embeddings) # Ajustar o modelo de clustering nos embeddings
    comment_cluster_list = clustering_model.labels_ # Atribuir cada comentário a um cluster

    return embeddings, comment_cluster_list


def generate_plot_fig(embeddings, comment_cluster_list, num_clusters:int, model_name:str):
    tsne = TSNE(n_components=2, random_state=0) # Redução de dimensionalidade com t-SNE para visualização
    embeddings_2d = tsne.fit_transform(embeddings.cpu().detach().numpy())

    # Plotar os clusters
    plt.figure(figsize=(10, 8))
    for i in range(num_clusters):
        cluster_points = embeddings_2d[comment_cluster_list == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i}')

    plt.legend()
    plt.savefig(f"sbert/{model_name}_{num_clusters}_clusters.png")

def generate_cluster_description_list(comments, comment_cluster_list, num_clusters, num_terms):
    cluster_texts = [[] for _ in range(num_clusters)]
    for i, label in enumerate(comment_cluster_list):
        cluster_texts[label].append(comments[i])

    # Criar TF-IDF para cada cluster e identificar termos importantes
    for idx, texts in enumerate(cluster_texts):
        tfidf = TfidfVectorizer(max_features=num_terms)  # Limite para as 10 palavras mais relevantes
        tfidf_matrix = tfidf.fit_transform(texts)
        termos_importantes = tfidf.get_feature_names_out()
        print(f"Cluster {idx} - Termos principais: {termos_importantes}")

    for idx, texts in enumerate(cluster_texts):
        # Transformar textos para TF-IDF antes de aplicar NMF
        tfidf = TfidfVectorizer(max_features=1000)
        tfidf_matrix = tfidf.fit_transform(texts)

        # Aplicar NMF para descobrir tópicos
        nmf_model = NMF(n_components=1, random_state=0)
        nmf_model.fit(tfidf_matrix)

        # Obter as palavras principais do tópico
        for topic_idx, topic in enumerate(nmf_model.components_):
            termos_importantes = [tfidf.get_feature_names_out()[i] for i in topic.argsort()[-num_terms:]]
            print(f"Cluster {idx} - Termos principais: {termos_importantes}")

if __name__ == '__main__':
    comments = generate_comments_list(['Likes','Dislikes'])[:5000]

    model = 'paraphrase-MiniLM-L12-v2' #'all-MiniLM-L6-v2'
    num_clusters = 10
    num_interest_terms = 10

    embeddings, comment_cluster_list = clustering_comments(comments, model, num_clusters)
    generate_plot_fig(embeddings, comment_cluster_list, num_clusters, model)
    generate_cluster_description_list(comments, comment_cluster_list, num_clusters, num_interest_terms)
