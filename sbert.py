from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import numpy as np

from comments_summarize_ollama_bash import generate_comments_list

def clustering_comments(comments:list, model_name:str, num_clusters:int):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(comments, convert_to_tensor=True)
    clustering_model = KMeans(n_clusters=num_clusters)

    clustering_model.fit(embeddings) # Ajustar o modelo de clustering nos embeddings
    comment_cluster_list = clustering_model.labels_ # Atribuir cada comentário a um cluster

    return embeddings, comment_cluster_list, clustering_model


def generate_cluster_description(comments, comment_cluster_list, num_clusters, num_terms):
    cluster_texts = [[] for _ in range(num_clusters)] # Agrupar os comentários por cluster
    for i, label in enumerate(comment_cluster_list):
        cluster_texts[label].append(comments[i])
    
    cluster_topics = {}  # Dicionário para armazenar os tópicos de cada cluster
    for idx, texts in enumerate(cluster_texts): # Analisar cada cluster e identificar os tópicos com NMF
        # Transformar os textos em uma matriz TF-IDF
        tfidf = TfidfVectorizer(max_features=1000)
        tfidf_matrix = tfidf.fit_transform(texts)

        # Aplicar NMF para descobrir tópicos
        nmf_model = NMF(n_components=1, random_state=0, max_iter=1000)
        nmf_model.fit(tfidf_matrix)

        # Obter as palavras principais do tópico
        topic = nmf_model.components_[0]  # Como n_components=1, pegamos o único tópico gerado
        termos_importantes = [tfidf.get_feature_names_out()[i] for i in topic.argsort()[-num_terms:]]
        
        # Armazenar os termos importantes no dicionário
        cluster_topics[idx] = termos_importantes

    return cluster_topics

def generate_plot_fig(embeddings, comment_cluster_list, cluster_description_dict, num_clusters:int, model_name:str):
    tsne = TSNE(n_components=2, random_state=0) # Redução de dimensionalidade com t-SNE para visualização
    embeddings_2d = tsne.fit_transform(embeddings.cpu().detach().numpy())

    # Plotar os clusters
    plt.figure(figsize=(12, 10))
    for i in range(num_clusters):
        cluster_points = embeddings_2d[comment_cluster_list == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'{cluster_description_dict[i]}')

    plt.legend(bbox_to_anchor=(0.5, 1.15), loc='upper center', ncol=1)
    plt.savefig(f"sbert/{model_name}_{num_clusters}_clusters.png")

def generate_cluster_metrics(clustering_model, embeddings, num_clusters):
    inertia = clustering_model.inertia_
    print(f'inertia: {inertia}')
    silhouette_avg = silhouette_score(embeddings, clustering_model.labels_)
    print(f'silhouette_avg: {silhouette_avg}')
    silhouette_vals = silhouette_samples(embeddings, clustering_model.labels_)
    silhouette_avg_clusters = {}
    for i in range(num_clusters):
        cluster_silhouette = silhouette_vals[clustering_model.labels_ == i]
        silhouette_avg_clusters[i] = np.mean(cluster_silhouette)
    print(f'silhouette_avg_clusters: {silhouette_avg_clusters}')
    db_index = davies_bouldin_score(embeddings, clustering_model.labels_)
    print(f'db_index: {db_index}')
    ch_index = calinski_harabasz_score(embeddings, clustering_model.labels_)
    print(f'ch_index: {ch_index}')

if __name__ == '__main__':
    comments = generate_comments_list(['Likes','Dislikes'])#[:5000]

    model = ['all-MiniLM-L6-v2','paraphrase-MiniLM-L12-v2'][0]
    num_clusters = 10
    num_interest_terms = 10

    embeddings, comment_cluster_list, clustering_model = clustering_comments(comments, model, num_clusters)
    cluster_description_dict = generate_cluster_description(comments, comment_cluster_list, num_clusters, num_interest_terms)
    generate_plot_fig(embeddings, comment_cluster_list, cluster_description_dict, num_clusters, model)
    generate_cluster_metrics(clustering_model, embeddings, num_clusters)