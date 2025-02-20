import re
import mysql.connector
import nltk
import spacy
import pytesseract
from PIL import Image
import io
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from datetime import datetime
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize NLP tools
nltk.download('stopwords')
spacy_model = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words('english'))

# Database Configuration
DB_CONFIG = {
    'host': os.getenv("DB_HOST"),
    'user': os.getenv("DB_USER"),
    'password': os.getenv("DB_PASSWORD"),
    'database': os.getenv("DB_DATABASE")
}

# Connect to MySQL
def connect_db():
    """Connect to MySQL database."""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        print(f"[Error] Connecting to database: {e}")
        return None

# Clean Text
def clean_text(text):
    """Remove HTML, special characters, and stopwords."""
    if not text:
        return ""
    
    # text = BeautifulSoup(text, "html.parser").get_text()  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip().lower()  # Normalize text
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

# Extract Text from Image Using OCR
def extract_text_from_image(image_blob):
    """Extract text from an image using OCR (Tesseract)."""
    try:
        image = Image.open(io.BytesIO(image_blob))  # Convert BLOB to Image
        extracted_text = pytesseract.image_to_string(image)  # Perform OCR
        return extracted_text.strip()
    except Exception as e:
        print(f"[Error] OCR Failed: {e}")
        return ""

# Mask Username
def mask_username(username):
    """Mask usernames to maintain privacy."""
    return "user_" + str(abs(hash(username)) % (10 ** 8))

# Fetch and Clean Posts (Including OCR for Images)
def fetch_and_clean_posts():
    """Fetch Reddit posts, clean them, and extract text from images using OCR."""
    conn = connect_db()
    cursor = conn.cursor()

    cursor.execute("SELECT id, title, selftext, author, created_utc, subreddit, image FROM posts")
    posts = cursor.fetchall()

    cleaned_posts = []
    for post in posts:
        post_id, title, selftext, author, created_utc, subreddit, image = post

        # Clean text
        clean_title = clean_text(title)
        clean_selftext = clean_text(selftext)

        # Extract text from images if available
        ocr_text = extract_text_from_image(image) if image else ""

        # Mask author
        masked_author = mask_username(author)

        # Convert timestamp correctly
        if isinstance(created_utc, datetime):
            formatted_time = created_utc.strftime('%Y-%m-%d %H:%M:%S')
        else:
            formatted_time = datetime.fromtimestamp(created_utc).strftime('%Y-%m-%d %H:%M:%S')

        cleaned_posts.append((post_id, clean_title, clean_selftext, ocr_text, masked_author, formatted_time, subreddit))

    return cleaned_posts

# Store Cleaned Posts into a New Table
def store_cleaned_posts(cleaned_posts):
    """Store cleaned posts into the cleaned_posts table."""
    conn = connect_db()
    cursor = conn.cursor()

    # Modify the cleaned_posts table to include OCR text
    create_table_query = """
    CREATE TABLE IF NOT EXISTS cleaned_posts (
        id VARCHAR(50) PRIMARY KEY,
        clean_title TEXT,
        clean_selftext TEXT,
        ocr_text TEXT,
        masked_author VARCHAR(255),
        created_utc DATETIME,
        subreddit VARCHAR(100)
    );
    """
    cursor.execute(create_table_query)
    conn.commit()

    # Insert cleaned data into cleaned_posts table
    insert_query = """
    INSERT INTO cleaned_posts (id, clean_title, clean_selftext, ocr_text, masked_author, created_utc, subreddit)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE 
        clean_title = VALUES(clean_title),
        clean_selftext = VALUES(clean_selftext),
        ocr_text = VALUES(ocr_text),
        masked_author = VALUES(masked_author),
        created_utc = VALUES(created_utc),
        subreddit = VALUES(subreddit);
    """

    cursor.executemany(insert_query, cleaned_posts)
    conn.commit()

    print(f"✅ Successfully stored {len(cleaned_posts)} cleaned posts in `cleaned_posts` table.")
    cursor.close()
    conn.close()

# Main Function
def main():
    print("🔄 Fetching and cleaning Reddit posts...")
    cleaned_posts = fetch_and_clean_posts()
    
    print("💾 Storing cleaned posts into `cleaned_posts` table...")
    store_cleaned_posts(cleaned_posts)

    print("🎯 Data cleaning and storage completed!")

if __name__ == "__main__":
    main()