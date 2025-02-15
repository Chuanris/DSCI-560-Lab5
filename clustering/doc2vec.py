import os
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import mysql.connector
from mysql.connector import Error
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import joblib


# -----------------------------------------------------------------------------
# Load environment variables and configure database connection
# -----------------------------------------------------------------------------
load_dotenv()

DB_CONFIG = {
    'host': os.getenv("DB_HOST"),
    'user': os.getenv("DB_USER"),
    'password': os.getenv("DB_PASSWORD"),
    'database': os.getenv("DB_DATABASE")
}

def connect_db():
    """Connect to the MySQL database."""
    try:
        connection = mysql.connector.connect(
            host=DB_CONFIG['host'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password'],
            database=DB_CONFIG['database']
        )
        return connection
    except Error as e:
        print("Error while connecting to database", e)
        raise

# -----------------------------------------------------------------------------
# Step 1: Fetch posts from the cleaned_posts table
# -----------------------------------------------------------------------------
def get_posts():
    """Retrieve posts from the cleaned_posts table."""
    connection = connect_db()
    cursor = connection.cursor(dictionary=True)
    query = "SELECT id, clean_title, clean_selftext, ocr_text FROM cleaned_posts"
    cursor.execute(query)
    posts = cursor.fetchall()
    cursor.close()
    connection.close()
    return posts

# -----------------------------------------------------------------------------
# Step 2: Prepare the text for embedding (combine title, selftext, and OCR text)
# -----------------------------------------------------------------------------
def prepare_documents(posts):
    """
    Combine the 'clean_title', 'clean_selftext' and 'ocr_text' (if present)
    into one string per post.
    """
    docs = []
    for post in posts:
        text = f"{post['clean_title']} {post['clean_selftext']}"
        if post['ocr_text'] and post['ocr_text'].strip().lower() != "none":
            text += " " + post['ocr_text']
        docs.append(text)
    return docs

# -----------------------------------------------------------------------------
# Step 3: Compute document embeddings using Doc2Vec
# -----------------------------------------------------------------------------
def compute_doc2vec_embeddings(docs):
    """
    Train a Doc2Vec model on the provided documents and return an embedding
    for each document.
    """
    tagged_docs = [TaggedDocument(doc.split(), [i]) for i, doc in enumerate(docs)]
    model = Doc2Vec(vector_size=100, window=5, min_count=1, workers=4, epochs=40)
    model.build_vocab(tagged_docs)
    model.train(tagged_docs, total_examples=model.corpus_count, epochs=model.epochs)
    
    # Retrieve the vector for each document (using the internal document vectors)
    embeddings = [model.dv[i] for i in range(len(docs))]
    return embeddings, model

# -----------------------------------------------------------------------------
# Step 4: Cluster the document embeddings using KMeans
# -----------------------------------------------------------------------------
def cluster_embeddings(embeddings, num_clusters=5):
    """
    Cluster the embeddings using KMeans.
    Returns cluster labels and the KMeans model.
    """
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    return labels, kmeans  # Return the KMeans model


# -----------------------------------------------------------------------------
# Step 5a: Identify the message closest to each cluster centroid
# -----------------------------------------------------------------------------
def find_closest_messages(embeddings, labels, centroids):
    """
    For each cluster, find the document whose embedding is closest to the centroid.
    Returns a dictionary mapping cluster index to the document index.
    """
    closest = {}
    for cluster in np.unique(labels):
        # Get indices for documents in this cluster
        cluster_indices = [i for i, label in enumerate(labels) if label == cluster]
        cluster_vectors = np.array([embeddings[i] for i in cluster_indices])
        centroid = centroids[cluster]
        # Compute Euclidean distances from the centroid
        distances = np.linalg.norm(cluster_vectors - centroid, axis=1)
        # Get the index (in the full embeddings list) of the closest document
        min_index = cluster_indices[np.argmin(distances)]
        closest[cluster] = min_index
    return closest

# -----------------------------------------------------------------------------
# Step 5b: Extract keywords/topics from messages in each cluster using TF-IDF
# -----------------------------------------------------------------------------
def extract_keywords_for_cluster(texts, top_n=5):
    """
    Extract keywords from a list of texts by computing TF-IDF scores.
    Returns a list of the top_n keywords.
    """
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(texts)
    # Average TF-IDF scores across documents in the cluster
    scores = np.array(tfidf_matrix.mean(axis=0)).flatten()
    # Get indices of top scoring terms
    indices = scores.argsort()[-top_n:][::-1]
    feature_names = np.array(vectorizer.get_feature_names_out())
    keywords = feature_names[indices]
    return list(keywords)

def get_cluster_keywords(docs, labels, num_clusters=5):
    """
    For each cluster, extract keywords from all documents within that cluster.
    Returns a dictionary mapping cluster index to its list of keywords.
    """
    cluster_keywords = {}
    for cluster in range(num_clusters):
        cluster_docs = [docs[i] for i, label in enumerate(labels) if label == cluster]
        if cluster_docs:
            keywords = extract_keywords_for_cluster(cluster_docs, top_n=5)
        else:
            keywords = []
        cluster_keywords[cluster] = keywords
    return cluster_keywords


# -----------------------------------------------------------------------------
# Step 6: Visualize the clusters using PCA for dimensionality reduction
# -----------------------------------------------------------------------------
def visualize_clusters(embeddings, labels, cluster_keywords, closest_messages, posts):
    """
    Reduce the embeddings to 2D using PCA and visualize clusters.
    The closest message of each cluster is marked and clusters are annotated
    with their keywords.
    """
    # Reduce to 2D for visualization
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels,
                          cmap='viridis', alpha=0.6, edgecolor='k')
    plt.colorbar(scatter, label="Cluster Label")
    
    unique_labels = np.unique(labels)
    for cluster in unique_labels:
        # Get points of current cluster and compute centroid in 2D
        indices = np.where(labels == cluster)[0]
        cluster_points = embeddings_2d[indices]
        centroid = np.mean(cluster_points, axis=0)
        
        # Annotate centroid with cluster number and its keywords
        keywords_text = ", ".join(cluster_keywords[cluster])
        plt.text(centroid[0], centroid[1], f"Cluster {cluster}\n{keywords_text}",
                 fontsize=9, ha='center', va='center',
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='black', boxstyle='round'))
        
        # Mark the closest message to the centroid
        closest_index = closest_messages[cluster]
        plt.scatter(embeddings_2d[closest_index, 0], embeddings_2d[closest_index, 1],
                    c='red', marker='x', s=100, linewidths=2)
    
    plt.title("Message Clusters Visualization (PCA)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.tight_layout()
    plt.savefig("cluster_visualization.png", dpi=300, bbox_inches='tight')
    plt.close()


# -----------------------------------------------------------------------------
# Step 7: Update the database with the cluster ID and keywords for each post
# (Assumes that the cleaned_posts table has columns "cluster_id" and "keywords")
# -----------------------------------------------------------------------------
def update_posts_with_cluster_info(posts, labels, cluster_keywords):
    """
    For each post, update the database with the assigned cluster label and
    the keywords for that cluster.
    """
    try:
        connection = connect_db()
        cursor = connection.cursor()
        for post, label in zip(posts, labels):
            # Convert list of keywords into a comma-separated string
            keywords_str = ", ".join(cluster_keywords[label])
            update_query = "UPDATE cleaned_posts SET cluster_id = %s, keywords = %s WHERE id = %s"
            cursor.execute(update_query, (int(label), keywords_str, post['id']))
        connection.commit()
        print("Database successfully updated with cluster info.")
    except Error as e:
        print("Error updating posts:", e)
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

# -----------------------------------------------------------------------------
# Main processing function
# -----------------------------------------------------------------------------
def main():
    # 1. Load posts from the database
    posts = get_posts()
    if not posts:
        print("No posts found in the database.")
        return

    # 2. Prepare document texts (combining title, selftext, OCR text)
    docs = prepare_documents(posts)

    # 3. Compute document embeddings using Doc2Vec
    embeddings, d2v_model = compute_doc2vec_embeddings(docs)

    # 4. Cluster the embeddings (adjust num_clusters as needed)
    num_clusters = 2
    labels, kmeans_model = cluster_embeddings(embeddings, num_clusters=num_clusters)
    centroids = kmeans_model.cluster_centers_

    silhouette = silhouette_score(embeddings, labels)
    print(f"🧩 Silhouette Score: {silhouette:.3f}")


    # 5a. Identify the message closest to each cluster centroid
    closest_messages = find_closest_messages(embeddings, labels, centroids)

    # 5b. Extract keywords/topics for each cluster
    cluster_keywords = get_cluster_keywords(docs, labels, num_clusters=num_clusters)

    # Display results for each cluster
    print("\nCluster Analysis Results:")
    for cluster, msg_index in closest_messages.items():
        print(f"\n--- Cluster {cluster} ---")
        print("Closest message ID:", posts[msg_index]['id'])
        print("Message text:", docs[msg_index])
        print("Cluster keywords:", cluster_keywords[cluster])
    
    # 6. Visualize the clusters and keywords
    visualize_clusters(embeddings, labels, cluster_keywords, closest_messages, posts)
    
    # 7. Update the posts in the database with the cluster info
    update_posts_with_cluster_info(posts, labels, cluster_keywords)

    
    d2v_model.save("doc2vec_model.bin")
    print("Doc2Vec model saved to doc2vec_model.bin")
    joblib.dump(kmeans_model, "kmeans_model.pkl")
    print("KMeans model saved to kmeans_model.pkl")


if __name__ == '__main__':
    main()