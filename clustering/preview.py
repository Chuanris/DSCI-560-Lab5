import mysql.connector
import base64

# Database Configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'Ashley020501',
    'database': 'reddit_data'
}

def connect_db():
    """Connect to MySQL database."""
    return mysql.connector.connect(**DB_CONFIG)

def fetch_all_posts():
    """Fetch all columns from the posts table."""
    conn = connect_db()
    cursor = conn.cursor()

    # Fetch all columns from the posts table
    cursor.execute("SELECT id, title, selftext, author, created_utc, subreddit, image FROM posts LIMIT 5")
    rows = cursor.fetchall()

    for post_id, title, selftext, author, created_utc, subreddit, image_data in rows:
        print(f"🆔 ID: {post_id}")
        print(f"📄 Title: {title}")
        print(f"📝 Selftext: {selftext[:500] if selftext else 'None'}")  # Limit to 500 chars for readability
        print(f"👤 Author: {author}")
        print(f"⏰ Created UTC: {created_utc}")
        print(f"📂 Subreddit: {subreddit}")

        # Handle image data
        if image_data:
            print(f"🖼️ Image: {len(image_data)} bytes available.")
            # Save the image to inspect it visually
            with open(f"{post_id}_image.png", "wb") as img_file:
                img_file.write(image_data)
            print(f"💾 Image saved as {post_id}_image.png\n")
        else:
            print("⚠️ No image data found.\n")

        print("=" * 80)

    cursor.close()
    conn.close()


def fetch_all_cleaned_posts():
    """Fetch all columns from the cleaned_posts table, including cluster_id and keywords."""
    conn = connect_db()
    cursor = conn.cursor()

    # Fetch all columns from the cleaned_posts table, including the new columns
    cursor.execute("""
        SELECT id, clean_title, clean_selftext, ocr_text, masked_author, created_utc, subreddit, cluster_id, keywords 
        FROM cleaned_posts 
        ORDER BY created_utc DESC
        LIMIT 25
    """)
    rows = cursor.fetchall()

    for post_id, clean_title, clean_selftext, ocr_text, masked_author, created_utc, subreddit, cluster_id, keywords in rows:
        print(f"🆔 ID: {post_id}")
        print(f"📄 Clean Title: {clean_title[:100] if clean_title else 'None'}")  # Limit to 100 chars
        print(f"📝 Clean Selftext: {clean_selftext[:500] if clean_selftext else 'None'}")  # Limit to 500 chars
        print(f"🖼️ OCR Text: {ocr_text[:300] if ocr_text else 'None'}")  # Limit to 300 chars
        print(f"👤 Masked Author: {masked_author}")
        print(f"⏰ Created UTC: {created_utc}")
        print(f"📂 Subreddit: {subreddit}")
        print(f"🧩 Cluster ID: {cluster_id if cluster_id is not None else 'None'}")
        print(f"🔑 Keywords: {keywords if keywords else 'None'}")
        print("=" * 80)

    cursor.close()
    conn.close()




if __name__ == "__main__":
    # fetch_all_posts()
    fetch_all_cleaned_posts()
