import mysql.connector
from PIL import Image
import io
import matplotlib.pyplot as plt
import os
import dotenv

# Load Environment Variables
dotenv.load_dotenv()

# Load Database Configuration from Environment Variables
DB_CONFIG = {
    'host': os.getenv("DB_HOST"),
    'user': os.getenv("DB_USER"),
    'password': os.getenv("DB_PASSWORD"),
    'database': os.getenv("DB_DATABASE")  # Ensure database name is included
}

def connect_db():
    """Connect to MySQL database and ensure the correct database is selected."""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute(f"USE {DB_CONFIG['database']}")  # Select database explicitly
        return conn
    except mysql.connector.Error as e:
        print(f"[Error] Connecting to database: {e}")
        return None

def fetch_images():
    """Fetch images stored in BLOB format from the MySQL database."""
    conn = connect_db()
    if conn is None:
        return []

    try:
        cursor = conn.cursor()
        cursor.execute("SELECT image FROM posts WHERE image IS NOT NULL LIMIT 5;")
        images = cursor.fetchall()
        cursor.close()
        conn.close()
        return images
    except mysql.connector.Error as e:
        print(f"[Error] Fetching images: {e}")
        return []

def display_images(images):
    """Display images retrieved from BLOB data."""
    if not images:
        print("[Warning] No images found in database.")
        return

    for i, image_data in enumerate(images):
        try:
            img = Image.open(io.BytesIO(image_data[0]))  # Convert BLOB to Image
            plt.figure(figsize=(6, 6))
            plt.imshow(img)
            plt.axis("off")
            plt.title(f"Image {i+1}")
            plt.show()
        except Exception as e:
            print(f"[Error] Unable to display image {i+1}: {e}")

# Fetch and Display Images
images = fetch_images()
display_images(images)