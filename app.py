"""
Flask App with Plotly.js and D3.js Visualizations
==============================================

This app reads text reuse data drawn from X feminist zines from XXXX-XXXX.
It is meant as a research tool to understand knowledge sharing networks between local activists across the US.

How to run:
1. Install Flask: pip install -r requirements.txt
2. Run: python app.py
3. Open: http://localhost:5000
"""

from flask import Flask, render_template, jsonify # type: ignore
import pandas as pd # type: ignore
import csv
import os
import sqlite3

app = Flask(__name__)

#path to files
pub_metadata = 'zinepub_metadata.csv'
db_path = 'zine_database.db'

#home page route
@app.route('/')
def index():
    return render_template('index.html')

#page-level reuse map route
@app.route('/reuse-map')
def reuse_map():
    return render_template('reusemap.html')

#page-level reuse network route
@app.route('/reuse-network') 
def reuse_network():
    return render_template('reusenetwork.html')

# ADD THIS: geographic network route
@app.route('/geographic-network')
def geographic_network():
    return render_template('geographic-network.html')

#publication data route
@app.route('/publication-data')
def publication_data():
    return render_template('database.html')

#API endpoint for publications data
@app.route('/api/publications')
def api_publications():
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row  # This makes rows behave like dictionaries
        
        cursor = conn.execute("SELECT * FROM Publications ORDER BY pub_id")
        publications = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        
        return jsonify(publications)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
# API endpoint for reuse network data
@app.route('/api/reuse-data')
def api_reuse_data():
    try:
        reuse_data_path = 'reuse_results/text_reuse_ngrams_windowed_parallel_filtered.csv'
        
        # Check if file exists
        if not os.path.exists(reuse_data_path):
            return jsonify({'error': 'Reuse data file not found'}), 404
        
        # Read CSV and convert to JSON
        df = pd.read_csv(reuse_data_path)
        
        # Convert dates to string format for JSON serialization
        if 'source_date' in df.columns:
            df['source_date'] = pd.to_datetime(df['source_date']).dt.strftime('%Y-%m-%d')
        if 'target_date' in df.columns:
            df['target_date'] = pd.to_datetime(df['target_date']).dt.strftime('%Y-%m-%d')
        
        return jsonify(df.to_dict('records'))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ADD THIS: API endpoint for metadata
@app.route('/api/metadata')
def api_metadata():
    try:
        metadata_path = 'zinepage_metadata.csv'
        
        # Check if file exists
        if not os.path.exists(metadata_path):
            return jsonify({'error': 'Metadata file not found'}), 404
        
        # Read CSV and convert to JSON
        df = pd.read_csv(metadata_path)
        
        # Add debugging info
        print(f"Metadata loaded: {len(df)} rows, {len(df.columns)} columns")
        print(f"Columns: {list(df.columns)}")
        if len(df) > 0:
            print(f"Sample locations: {df['location'].dropna().unique()[:5] if 'location' in df.columns else 'No location column'}")
        
        return jsonify(df.to_dict('records'))
        
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Flask app...")
    app.run(debug=True, host='0.0.0.0', port=5001)