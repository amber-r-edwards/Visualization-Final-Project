# Feminist Zine Text Reuse Visualization

## Description
This project provides interactive visualizations for exploring text reuse patterns in feminist publications from 1970-1975, extracted from a larger corpus of zines (up to 1997) from Herstory Archive: Feminist Newspapers. The Flask web application presents multiple perspectives on the data through geographic mapping, network analysis, and a searchable database interface. The visualizations allow users to explore cross-publication connections, filter by publication type and time periods, and examine the geographic distribution of feminist publishing during this era.

## Methods
The project uses D3.js for interactive data visualizations, including:
- **Geographic Network Visualization**: Maps text reuse connections between cities where publications were based
- **Network Graph Visualization**: Shows page-level connections as an interactive node-link diagram  

The project also visualizes a SQL database of the included publications, including full metadata and tracking of my progress on OCR processing and notes. *The visualizations do not draw from the SQL database, rather individual CSV files - though eventually this may be consolidated.*
- **Publication Database**: Provides a searchable interface for exploring publication metadata


Data processing involves linking three CSV datasets (text reuse matches, page metadata, and publication metadata) using publication IDs for consistent cross-referencing and color coding. 

## Project Structure
```
VisualizationProject/
├── README.md
├── app.py                                                      # Script to deploy the Flask application to a development server
├── static/                                                     # Files served to the browser without processing
│   ├── data/
│   │   ├── text_reuse_ngrams_windowed_parallel_filtered.csv
│   │   ├── zinepage_metadata.csv
│   │   └── zinepub_metadata.csv
│   └── js/
│       └── publication-colors.js
├── templates/                                                  # HTML files for each webpage and the base 
│   ├── base.html
│   ├── database.html
│   ├── index.html
│   ├── reusemap.html
│   └── reusenetwork.html
├── createdatabase.py                                           # Script to create the SQL database (Publications)                                         
├── zine_database.db                                            # SQL database with Publication metadata, used for Publication Database page
├── requirements.txt                                            # Packages needed to duplicate project (does not include those for text analysis)
├── analysisscripts/                                            # N-Gram and Jaccard similarity and metadata scripts (for visibility)
└── txtfiles/ (ignored)                                         # Page-level txt files pulled into metadata in order to run N-gram and Jaccard Similarity                        
```

## Copyright Notice
The text files in the `txtfiles/` directory contain copyrighted material and are excluded from version control via `.gitignore`. These files are not included in the repository due to copyright restrictions.

## Requirements
- Python 3.7+
- Flask
- Modern web browser with JavaScript enabled

Python dependencies:
```
Flask==2.3.3
```

## Usage
### Deploying the Flask Application

1. **Clone the repository**:
   ```bash
   git clone [repository-url]
   cd VisualizationProject
   ```

2. **Install dependencies**:
   ```bash
   pip install Flask
   ```

3. **Ensure data files are present**:
   Verify that the following CSV files exist in `static/data/`:
   - `text_reuse_ngrams_windowed_parallel_filtered.csv`
   - `zinepage_metadata.csv`  
   - `zinepub_metadata.csv`

4. **Run the Flask application**:
   ```bash
   python app.py
   ```

5. **Access the application**:
   Open your web browser and navigate to `http://localhost:5000`

### Navigation
- **Home**: Overview and geographic distribution of publications drawn from full SQL database, default page opens to 1970-1975 reflected in visualizations.
- **Geographic Network**: Interactive map showing text reuse connections between cities
- **Text Reuse Network**: Page-level network visualization of reuse connections between pages.
- **Database**: Searchable interface for publication metadata

Each visualization includes interactive filters and controls for exploring different aspects of the data.
```