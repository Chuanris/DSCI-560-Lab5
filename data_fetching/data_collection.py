import os
import sys
import time
import requests
import praw
import mysql.connector
from dotenv import load_dotenv
from prawcore.exceptions import TooManyRequests

#load .env
load_dotenv()

def get_reddit_instance():
    """Initialize Reddit API client"""
    try:
        reddit = praw.Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID"),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
            user_agent=os.getenv("REDDIT_USER_AGENT")
        )
        return reddit
    except Exception as e:
        print(f"[Error] Initializing Reddit instance: {e}")
        sys.exit(1)

def connect_db():
    """Connect to MySQL database"""
    try:
        conn = mysql.connector.connect(
            host=os.getenv("DB_HOST"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            database=os.getenv("DB_DATABASE")
        )
        return conn
    except Exception as e:
        print(f"[Error] Connecting to database: {e}")
        sys.exit(1)

# Fetch images in posts
def download_image(post):
    """Download image from post URL or Reddit preview/media metadata."""
    image_url = None
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif')

    if post.url.lower().endswith(image_extensions):
        image_url = post.url
    elif hasattr(post, 'preview'):
        image_url = post.preview['images'][0]['source']['url']
    elif hasattr(post, 'media_metadata'):
        for media in post.media_metadata.values():
            if 's' in media:
                image_url = media['s']['u']
                break

    if image_url:
        try:
            response = requests.get(image_url, timeout=10)
            if response.status_code == 200:
                return response.content
        except Exception as e:
            print(f"[Warning] Failed to download image from {image_url}: {e}")

    return None

# Insert query to store each post to MySQL database
def insert_post(conn, cursor, post):
    """Insert or update a Reddit post in MySQL, including image data."""
    image_data = download_image(post)

    try:
        insert_query = """
        INSERT INTO posts (id, title, selftext, author, created_utc, subreddit, image)
        VALUES (%s, %s, %s, %s, FROM_UNIXTIME(%s), %s, %s)
        ON DUPLICATE KEY UPDATE 
            title = VALUES(title),
            selftext = VALUES(selftext),
            author = VALUES(author),
            created_utc = VALUES(created_utc),
            subreddit = VALUES(subreddit),
            image = VALUES(image);
        """
        data = (
            post.id, post.title, post.selftext, str(post.author), post.created_utc,
            post.subreddit.display_name, image_data
        )
        
        cursor.execute(insert_query, data)
        conn.commit()

        print(f"[Success] Stored post {post.id}: {post.title[:50]}...")

        # Fetch and store comments for this post
        # insert_comments(conn, cursor, post)

    except mysql.connector.Error as err:
        print(f"[Error] Inserting post {post.id}: {err}")

# Store comments
# if the loading time is too long, can ignore this part
def insert_comments(conn, cursor, post):
    """Fetch and store all comments related to a Reddit post with rate limiting."""
    attempt = 0

    # Max 5 retrys
    while attempt < 5:  
        try:
            post.comments.replace_more(limit=None)
            comments = post.comments.list()

            for comment in comments:
                insert_query = """
                INSERT INTO comments (id, post_id, author, body, created_utc)
                VALUES (%s, %s, %s, %s, FROM_UNIXTIME(%s))
                ON DUPLICATE KEY UPDATE 
                    body = VALUES(body),
                    author = VALUES(author),
                    created_utc = VALUES(created_utc);
                """
                data = (
                    comment.id, post.id, str(comment.author), comment.body, comment.created_utc
                )
                cursor.execute(insert_query, data)

            conn.commit()
            print(f"[Success] Stored {len(comments)} comments for post {post.id}")
            return  # Exit retry loop if successful

        except TooManyRequests as e:
            wait_time = (2 ** attempt) * 30  # Exponential backoff (30s, 60s, 120s, ...)
            print(f"[Rate Limit] Too many requests. Waiting {wait_time} seconds before retrying...")
            time.sleep(wait_time)
            attempt += 1

        except Exception as e:
            print(f"[Error] Fetching comments for post {post.id}: {e}")
            return  # Exit if a non-429 error occurs

    print(f"[Failed] Skipped fetching comments for post {post.id} after multiple retries.")

# Fetch the post from specific subreddit (r/stocks)
def fetch_and_update_posts(num_posts):
    """Fetch and store Reddit posts in batches while handling API limits."""
    print("Initializing Reddit instance...")
    reddit = get_reddit_instance()
    
    print("Connecting to database...")
    conn = connect_db()
    cursor = conn.cursor()

    # r/stocks
    subreddit = reddit.subreddit("stocks")
    
    print(f"Fetching {num_posts} posts from r/stocks...")

    fetched_posts = 0
    fetched_comments = 0
  
    # Give a API limit per request
    batch_size = 1000  
    last_created_utc = None

    while fetched_posts < num_posts:
        current_limit = min(batch_size, num_posts - fetched_posts)
        print(f"Fetching next batch of {current_limit} posts...")

        try:
            if last_created_utc is None:
                posts = list(subreddit.new(limit=current_limit))
            else:
                posts = list(subreddit.new(limit=current_limit, params={'before': last_created_utc}))

        except TooManyRequests:
            print("[Rate Limit] Too many requests. Waiting 60 seconds before retrying...")
            time.sleep(60)
            continue  

        if not posts:
            print("No more posts available from the API.")
            break

        for post in posts:
            insert_post(conn, cursor, post)
            fetched_posts += 1

        conn.commit()
        last_created_utc = posts[-1].created_utc

        print(f"Fetched {fetched_posts} posts so far.")
        time.sleep(2)  # Prevent API rate limits

    cursor.execute("SELECT COUNT(*) FROM comments")
    fetched_comments = cursor.fetchone()[0]

    cursor.close()
    conn.close()
    print(f"Data collection completed. Total posts fetched: {fetched_posts}, Total comments fetched: {fetched_comments}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python filename.py <number_of_posts>")
        sys.exit(1)
    
    try:
        num_posts = int(sys.argv[1])
    except ValueError:
        print("Error: The number of posts must be an integer.")
        sys.exit(1)
    
    start_time = time.time()
    fetch_and_update_posts(num_posts)
    elapsed = time.time() - start_time
    print(f"Data collection completed in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    main()
