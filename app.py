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
    return render_template('reuse_map.html')

#page-level reuse network route
@app.route('/reuse-network') 
def reuse_network():
    return render_template('reuse_network.html')

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

if __name__ == '__main__':
    print("Starting Flask app...")
    app.run(debug=True, host='0.0.0.0', port=5001)