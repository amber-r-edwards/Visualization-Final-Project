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

from flask import Flask, render_template, jsonify
import pandas as pd
import csv
import os

app = Flask(__name__)

#path to CSV files
pub_metadata = 'zinepub_metadata.csv'

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
    return render_template('publication_data.html')

@app.route('/api/metadata')
def api_metadata():
    # Load just the publication metadata
    metadata = pd.read_csv('zinepub_metadata.csv')
    return metadata.to_json(orient='records')

if __name__ == '__main__':
    print("Starting Flask app...")
    app.run(debug=True, host='0.0.0.0', port=5001)