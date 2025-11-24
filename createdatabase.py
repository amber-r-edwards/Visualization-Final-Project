import sqlite3
import pandas as pd # type: ignore
import os

def create_database():
    """
    Create a SQLite database from the zinepub_metadata.csv file
    """
    # Database file path (in root directory)
    db_path = 'zine_database.db'
    
    # Remove existing database if it exists
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"Removed existing database: {db_path}")
    
    # Read the CSV file with proper encoding
    try:
        df = pd.read_csv('zinepub_metadata.csv', encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv('zinepub_metadata.csv', encoding='latin-1')
            print("Used latin-1 encoding to read CSV file")
        except UnicodeDecodeError:
            df = pd.read_csv('zinepub_metadata.csv', encoding='cp1252')
            print("Used cp1252 encoding to read CSV file")
    
    # Create connection to SQLite database
    conn = sqlite3.connect(db_path)
    
    try:
        # Create the Publications table with pub_id as primary key
        create_table_sql = """
        CREATE TABLE Publications (
            pub_id INTEGER PRIMARY KEY AUTOINCREMENT,
            publication_name TEXT,
            organization TEXT,
            contributors TEXT,
            location TEXT,
            region TEXT,
            first_year INTEGER,
            last_year INTEGER,
            issues_in_corpus INTEGER,
            processed TEXT,
            source_archive TEXT,
            archive_collection TEXT,
            notes TEXT
        )
        """
        
        conn.execute(create_table_sql)
        print("Created Publications table with pub_id as primary key")
        
        # Insert data from CSV into the database
        df.to_sql('Publications', conn, if_exists='append', index=False)
        print(f"Inserted {len(df)} records into Publications table")
        
        # Verify the data
        cursor = conn.execute("SELECT COUNT(*) FROM Publications")
        count = cursor.fetchone()[0]
        print(f"Database created successfully with {count} publications")
        
        # Show first few records
        cursor = conn.execute("SELECT pub_id, publication_name, location, first_year, last_year FROM Publications LIMIT 5")
        print("\nFirst 5 records:")
        print("pub_id | publication_name | location | first_year | last_year")
        print("-" * 70)
        for row in cursor.fetchall():
            print(f"{row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]}")
            
    except Exception as e:
        print(f"Error creating database: {e}")
        
    finally:
        conn.close()

if __name__ == "__main__":
    create_database()