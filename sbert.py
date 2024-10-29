from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd

from comments_summarize_ollama_bash import generate_comments_list

def main(comments, model, num_clusters):
    embeddings, comment_cluster_list, clustering_model = clustering_comments(comments, model, num_clusters)
    cluster_description_dict = generate_cluster_description(comments, comment_cluster_list, num_clusters)
    generate_plot_fig(embeddings, comment_cluster_list, cluster_description_dict, num_clusters, model)
    metrics = generate_cluster_metrics(clustering_model, embeddings, num_clusters)
    model_info = [model,num_clusters]
    return model_info + metrics

def clustering_comments(comments:list, model_name:str, num_clusters:int):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(comments, convert_to_tensor=True)
    clustering_model = KMeans(n_clusters=num_clusters)

    clustering_model.fit(embeddings) # Ajustar o modelo de clustering nos embeddings
    comment_cluster_list = clustering_model.labels_ # Atribuir cada comentário a um cluster

    return embeddings, comment_cluster_list, clustering_model


def generate_cluster_description(comments, comment_cluster_list, num_clusters):
    cluster_texts = [[] for _ in range(num_clusters)] # Agrupar os comentários por cluster
    for i, label in enumerate(comment_cluster_list):
        cluster_texts[label].append(comments[i])
    
    cluster_topics = {}  # Dicionário para armazenar os tópicos de cada cluster
    for idx, texts in enumerate(cluster_texts): # Analisar cada cluster e identificar os tópicos com NMF
        # Transformar os textos em uma matriz TF-IDF
        tfidf = TfidfVectorizer(max_features=1000)
        tfidf_matrix = tfidf.fit_transform(texts)

        # Aplicar NMF para descobrir tópicos
        nmf_model = NMF(n_components=1, random_state=0, max_iter=10_000)
        nmf_model.fit(tfidf_matrix)

        # Obter as palavras principais do tópico
        topic = nmf_model.components_[0]  # Como n_components=1, pegamos o único tópico gerado
        num_interest_terms = 10
        termos_importantes = [tfidf.get_feature_names_out()[i] for i in topic.argsort()[-num_interest_terms:]]
        
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
    os.makedirs(f"sbert/{model}", exist_ok=True)
    plt.savefig(f"sbert/{model}/kmeans_{model_name}_{num_clusters}_clusters.png")

def generate_cluster_metrics(clustering_model, embeddings, num_clusters):
    inertia = clustering_model.inertia_
    silhouette_avg = silhouette_score(embeddings, clustering_model.labels_)
    silhouette_vals = silhouette_samples(embeddings, clustering_model.labels_)
    silhouette_avg_clusters = {}
    for i in range(num_clusters):
        cluster_silhouette = silhouette_vals[clustering_model.labels_ == i]
        silhouette_avg_clusters[i] = np.mean(cluster_silhouette)
    db_index = davies_bouldin_score(embeddings, clustering_model.labels_)
    ch_index = calinski_harabasz_score(embeddings, clustering_model.labels_)
    return [inertia,silhouette_avg,silhouette_avg,db_index,ch_index,silhouette_avg_clusters]

if __name__ == '__main__':
    comments = generate_comments_list(['Likes','Dislikes'])[:5000]
    models = [
    'all-MiniLM-L6-v2',            # Muito leve
    'paraphrase-MiniLM-L12-v2',    # Leve, mas mais robusto que L6
    'msmarco-distilbert-base-v4',  # Leve e otimizado para similaridade
    'all-distilroberta-v1',        # Leve, mas maior precisão semântica
    'all-mpnet-base-v2',           # Intermediário, alta precisão
    'paraphrase-mpnet-base-v2',    # Intermediário, ótima para parafraseamento
    'stsb-roberta-large'           # Mais pesado, alta precisão semântica
    ]
    nums_clusters = [10, 25, 50, 75, 100]
    
    infos = []
    for model in models:
        for num_clusters in nums_clusters:
            info = main(comments, model, num_clusters)
            infos.append(info)
    
    df = pd.DataFrame(infos, columns=['model','num_clusters','inertia','silhouette_avg','silhouette_avg','db_index','ch_index','silhouette_avg_clusters'])
    print(df)
    pd.to_pickle(df,'sbert/models_metrics.csv')