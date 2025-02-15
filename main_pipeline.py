import sys
import time
import subprocess
import matplotlib.pyplot as plt
from collections import Counter
import mysql.connector
import joblib
from gensim.models.doc2vec import Doc2Vec
import threading

def run_pipeline():
    """
    Run the full pipeline:
      1) data_fetching/data_collection.py (with 500 posts)
      2) data_processing/data_processing.py
      3) clustering/doc2vec.py
    Print messages before/after each step and handle errors.
    """
    print("\n=== Starting data update pipeline ===")

    # 1. Data Collection
    print("Fetching data using data_fetching/data_collection.py...")
    try:
        subprocess.check_call(["python", "data_fetching/data_collection.py", "5000"])
        print("Data fetching completed successfully.\n")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Data fetching failed: {e}")
        return  # Stop the pipeline here

    # 2. Data Processing
    print("Processing data using data_processing/data_processing.py...")
    try:
        subprocess.check_call(["python", "data_processing/data_processing.py"])
        print("Data processing completed successfully.\n")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Data processing failed: {e}")
        return  # Stop the pipeline here

    # 3. Clustering / Doc2Vec
    print("Updating database using clustering/doc2vec.py...")
    try:
        subprocess.check_call(["python", "clustering/doc2vec.py"])
        print("Database update completed successfully.\n")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Database update failed: {e}")
        return  # Stop the pipeline here

    print("=== Pipeline completed ===\n")


def display_cluster_messages(user_input):
    """
    1) Load the saved Doc2Vec model and KMeans model.
    2) Infer the vector for user_input.
    3) Predict which cluster it belongs to.
    4) Retrieve and display posts from that cluster.
    5) Display a bar chart of the user input's word frequency.
    """

    # -------------------------------------------------------------------------
    # 1) Load Doc2Vec and KMeans models
    # -------------------------------------------------------------------------
    try:
        d2v_model = Doc2Vec.load("doc2vec_model.bin")
        kmeans_model = joblib.load("kmeans_model.pkl")
    except Exception as e:
        print(f"[ERROR] Could not load models: {e}")
        return

    # -------------------------------------------------------------------------
    # 2) Infer vector for user_input
    # -------------------------------------------------------------------------
    tokens = user_input.split()
    inferred_vector = d2v_model.infer_vector(tokens)

    # -------------------------------------------------------------------------
    # 3) Predict cluster
    # -------------------------------------------------------------------------
    predicted_cluster = kmeans_model.predict([inferred_vector])[0]
    print(f"\n[INFO] The user's input belongs to cluster: {predicted_cluster}")

    # -------------------------------------------------------------------------
    # 4) Retrieve posts from that cluster
    # -------------------------------------------------------------------------
    try:
        connection = mysql.connector.connect(
            host="localhost",
            user="root",
            password="Ashley020501",
            database="reddit_data"
        )
        cursor = connection.cursor(dictionary=True)
        query = "SELECT id, clean_title, clean_selftext, ocr_text FROM cleaned_posts WHERE cluster_id = %s"
        cursor.execute(query, (int(predicted_cluster),))
        results = cursor.fetchall()

        if not results:
            print("[INFO] No messages found in this cluster.")
        else:
            print(f"[INFO] Found {len(results)} messages in cluster {predicted_cluster}:")
            for row in results[:5]:
                # Display only the first 5 posts for brevity
                print(f" - Post ID {row['id']}, Title: {row['clean_title'][:50]}...")
    except Exception as e:
        print(f"[ERROR] Could not query the database: {e}")
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'connection' in locals() and connection.is_connected():
            connection.close()

    # -------------------------------------------------------------------------
    # 5) Display bar chart for word frequency
    # -------------------------------------------------------------------------
    freq = Counter(tokens)
    if not freq:
        print("[INFO] No tokens in user input to display a chart.")
        return

    plt.figure(figsize=(6, 4))
    plt.bar(freq.keys(), freq.values(), color='blue')
    plt.title("Word Frequency in User Input")
    plt.xlabel("Words")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


def pipeline_scheduler(interval_seconds, stop_event):
    """
    Scheduler that runs run_pipeline() every interval_seconds seconds
    until stop_event is set.
    """
    while not stop_event.is_set():
        print("\n[Scheduler] Running pipeline update...")
        run_pipeline()
        print(f"[Scheduler] Pipeline completed. Next run in {interval_seconds} seconds.\n")
        # Wait for the specified interval, but check periodically if we need to stop.
        for _ in range(interval_seconds):
            if stop_event.is_set():
                break
            time.sleep(1)


def main():
    if len(sys.argv) < 2:
        print("Usage: python main_pipeline.py <interval_in_minutes>")
        sys.exit(1)

    # Convert interval from minutes to seconds
    try:
        interval_minutes = int(sys.argv[1])
        interval_seconds = interval_minutes * 60
    except ValueError:
        print("[ERROR] Interval must be an integer (minutes).")
        sys.exit(1)

    print(f"[INFO] Starting main script with an update interval of {interval_minutes} minute(s).")

    # Create a threading event to signal when the scheduler should stop.
    stop_event = threading.Event()

    # Start the pipeline scheduler in a separate thread.
    scheduler_thread = threading.Thread(target=pipeline_scheduler, args=(interval_seconds, stop_event), daemon=True)
    scheduler_thread.start()

    try:
        while True:
            user_input = input("Enter your keywords or message (type 'exit' to quit): ")
            if user_input.lower() == "exit":
                print("Exiting the script. Goodbye!")
                stop_event.set()  # Signal the scheduler thread to stop.
                scheduler_thread.join()
                break
            elif user_input.strip():
                display_cluster_messages(user_input)
            else:
                print("[INFO] No input provided.")
    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt detected. Exiting.")
        stop_event.set()
        scheduler_thread.join()


if __name__ == "__main__":
    main()
