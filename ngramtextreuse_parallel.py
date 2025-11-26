"""
N-grams and Jaccard Similarity Text Reuse Analysis (PARALLEL VERSION - NO PANDAS)
=================================================

This script analyzes text reuse between feminist publications using n-grams 
and Jaccard similarity, with parallel processing for faster execution.
Uses only built-in Python modules (no pandas required).

"""

import csv
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Set, Dict, List, Tuple
from multiprocessing import Pool, cpu_count
from collections import defaultdict, Counter

# ============================================================================
# TEXT PREPROCESSING AND N-GRAM FUNCTIONS
# ============================================================================

def clean_text(text: str) -> str:
    """Clean and normalize text for n-gram analysis."""
    if text is None or not text:
        return ""
    
    text = str(text).lower().strip()
    
    # Remove common OCR artifacts and normalize
    text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation but keep word boundaries
    text = re.sub(r'\d+', '', text)  # Remove numbers (often page numbers/dates)
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    
    return text.strip()

def create_ngrams(text: str, n: int = 4) -> Set[str]:
    """Create word n-grams from text."""
    words = text.split()
    if len(words) < n:
        return set()
    
    ngrams = set()
    for i in range(len(words) - n + 1):
        ngram = ' '.join(words[i:i+n])
        ngrams.add(ngram)
    
    return ngrams

def create_shingles(text: str, k: int = 5) -> Set[str]:
    """Create character k-shingles from text."""
    if len(text) < k:
        return {text} if text else set()
    
    shingles = set()
    for i in range(len(text) - k + 1):
        shingle = text[i:i+k]
        shingles.add(shingle)
    
    return shingles

def jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    """Calculate Jaccard similarity between two sets."""
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    return intersection / union if union > 0 else 0.0

def find_shared_content(text1: str, text2: str, ngram_size: int = 4) -> Tuple[str, str, str]:
    """
    Find the longest shared n-gram sequence and extract context around it.
    Returns (shared_content, source_context, target_context)
    """
    clean1 = clean_text(text1)
    clean2 = clean_text(text2)
    
    words1 = clean1.split()
    words2 = clean2.split()
    
    if len(words1) < ngram_size or len(words2) < ngram_size:
        return "", "", ""
    
    # Find all shared n-grams and their positions
    ngrams1 = {}
    for i in range(len(words1) - ngram_size + 1):
        ngram = ' '.join(words1[i:i+ngram_size])
        if ngram not in ngrams1:
            ngrams1[ngram] = []
        ngrams1[ngram].append(i)
    
    shared_sequences = []
    for i in range(len(words2) - ngram_size + 1):
        ngram = ' '.join(words2[i:i+ngram_size])
        if ngram in ngrams1:
            for pos1 in ngrams1[ngram]:
                shared_sequences.append((ngram, pos1, i))
    
    if not shared_sequences:
        return "", "", ""
    
    # Find the longest contiguous shared sequence
    best_sequence = ""
    best_pos1 = 0
    best_pos2 = 0
    best_length = 0
    
    for ngram, pos1, pos2 in shared_sequences:
        # Try to extend this sequence
        current_length = ngram_size
        extend_pos1 = pos1 + ngram_size
        extend_pos2 = pos2 + ngram_size
        
        # Extend forward
        while (extend_pos1 < len(words1) and 
               extend_pos2 < len(words2) and 
               words1[extend_pos1] == words2[extend_pos2]):
            current_length += 1
            extend_pos1 += 1
            extend_pos2 += 1
        
        if current_length > best_length:
            best_length = current_length
            best_pos1 = pos1
            best_pos2 = pos2
            best_sequence = ' '.join(words1[pos1:pos1 + current_length])
    
    if not best_sequence:
        # Fall back to the first shared n-gram
        ngram, pos1, pos2 = shared_sequences[0]
        best_sequence = ngram
        best_pos1 = pos1
        best_pos2 = pos2
        best_length = ngram_size
    
    # Extract context (50 words before and after)
    context_size = 50
    
    # Source context
    source_start = max(0, best_pos1 - context_size)
    source_end = min(len(words1), best_pos1 + best_length + context_size)
    source_context = ' '.join(words1[source_start:source_end])
    
    # Target context
    target_start = max(0, best_pos2 - context_size)
    target_end = min(len(words2), best_pos2 + best_length + context_size)
    target_context = ' '.join(words2[target_start:target_end])
    
    return best_sequence, source_context, target_context

def calculate_text_similarity(text1: str, text2: str, ngram_size: int = 4, shingle_size: int = 5) -> Dict[str, any]:
    """Calculate similarity between two texts using n-grams and shingles, including shared content."""
    # Clean the texts
    clean1 = clean_text(text1)
    clean2 = clean_text(text2)
    
    # Skip if either text is too short
    if len(clean1.split()) < ngram_size or len(clean2.split()) < ngram_size:
        return {
            'ngram_similarity': 0.0,
            'shingle_similarity': 0.0,
            'combined_similarity': 0.0,
            'shared_content': '',
            'source_context': '',
            'target_context': ''
        }
    
    # Create n-grams and shingles
    ngrams1 = create_ngrams(clean1, n=ngram_size)
    ngrams2 = create_ngrams(clean2, n=ngram_size)
    shingles1 = create_shingles(clean1, k=shingle_size)
    shingles2 = create_shingles(clean2, k=shingle_size)
    
    # Calculate similarities
    ngram_sim = jaccard_similarity(ngrams1, ngrams2)
    shingle_sim = jaccard_similarity(shingles1, shingles2)
    
    # Combined similarity (weighted average - emphasize n-grams for semantic meaning)
    combined_sim = 0.4 * ngram_sim + 0.6 * shingle_sim
    
    # Find shared content and context
    shared_content, source_context, target_context = find_shared_content(text1, text2, ngram_size)
    
    return {
        'ngram_similarity': ngram_sim,
        'shingle_similarity': shingle_sim,
        'combined_similarity': combined_sim,
        'shared_content': shared_content,
        'source_context': source_context,
        'target_context': target_context
    }

def create_text_windows(text: str, window_size: int = 200, overlap: int = 50) -> List[Dict[str, any]]:
    """
    Split text into overlapping windows for more granular text reuse detection.
    
    Args:
        text: Input text to window
        window_size: Number of words per window
        overlap: Number of overlapping words between windows
        
    Returns:
        List of dictionaries with window information
    """
    words = clean_text(text).split()
    
    if len(words) <= window_size:
        # Text is shorter than window size, return as single window
        return [{
            'window_id': 0,
            'start_word': 0,
            'end_word': len(words),
            'text': ' '.join(words),
            'total_words': len(words)
        }]
    
    windows = []
    window_id = 0
    start_pos = 0
    
    while start_pos < len(words):
        end_pos = min(start_pos + window_size, len(words))
        
        window_text = ' '.join(words[start_pos:end_pos])
        
        windows.append({
            'window_id': window_id,
            'start_word': start_pos,
            'end_word': end_pos,
            'text': window_text,
            'total_words': len(words)
        })
        
        # Move to next window
        if end_pos >= len(words):
            break
            
        start_pos += (window_size - overlap)
        window_id += 1
    
    return windows

def load_csv_data(filepath: str) -> List[Dict]:
    """
    Load CSV data into a list of dictionaries.
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        List of dictionaries, one per row
    """
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data

def save_csv_data(filepath: str, data: List[Dict], fieldnames: List[str] = None):
    """
    Save list of dictionaries to CSV.
    
    Args:
        filepath: Path to output CSV file
        data: List of dictionaries to save
        fieldnames: Optional list of field names (uses keys from first dict if not provided)
    """
    if not data:
        print(f"Warning: No data to save to {filepath}")
        return
    
    if fieldnames is None:
        fieldnames = list(data[0].keys())
    
    with open(filepath, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    
    print(f"Saved {len(data)} rows to {filepath}")

def load_and_prepare_metadata(filepath: str = None) -> List[Dict]:
    """
    Load and prepare metadata.
    
    Args:
        filepath: Path to CSV file with metadata. 
                  Expected columns: publication_name, page_id, issue_date, text
    
    Returns:
        List of dictionaries with metadata
    """
    if filepath is None:
        # Try common filenames
        for fname in ['zinepage_metadata.csv', 'metadata.csv', 'data.csv', 'publications.csv']:
            if os.path.exists(fname):
                filepath = fname
                break
        
        if filepath is None:
            raise FileNotFoundError(
                "No data file specified. Please provide a CSV file with columns: "
                "publication_name, page_id, issue_date, text"
            )
    else:
        # Expand ~ and environment variables in the path
        filepath = os.path.expanduser(filepath)
        filepath = os.path.expandvars(filepath)
    
    print(f"Loading metadata from {filepath}...")
    data = load_csv_data(filepath)
    print(f"Loaded {len(data)} records")
    
    return data

# ============================================================================
# PARALLEL PROCESSING FUNCTIONS
# ============================================================================

def process_comparison_chunk(args):
    """
    Process a chunk of comparisons. This function will be executed in parallel.
    
    Args:
        args: Tuple of (chunk_data, working_data, similarity_threshold, ngram_size, 
              shingle_size, same_pub, use_windows)
    
    Returns:
        List of comparison results for this chunk
    """
    chunk_indices, working_data, similarity_threshold, ngram_size, shingle_size, same_pub, use_windows = args
    
    results = []
    
    for idx1 in chunk_indices:
        row1 = working_data[idx1]
        
        # Compare with all subsequent segments
        for idx2 in range(idx1 + 1, len(working_data)):
            row2 = working_data[idx2]
            
            # Skip if same publication and not comparing within same pub
            if not same_pub and row1.get('publication_name') == row2.get('publication_name'):
                continue
            
            # Skip if comparing same page to itself (in windowed mode)
            if use_windows and row1.get('page_id') == row2.get('page_id'):
                continue
            
            # Calculate similarity
            similarity = calculate_text_similarity(
                row1.get('text', ''),
                row2.get('text', ''),
                ngram_size=ngram_size,
                shingle_size=shingle_size
            )
            
            # Only keep if above threshold
            if similarity['combined_similarity'] >= similarity_threshold:
                result = {
                    'source_publication': row1.get('publication_name', ''),
                    'target_publication': row2.get('publication_name', ''),
                    'source_page_id': row1.get('page_id', ''),
                    'target_page_id': row2.get('page_id', ''),
                    'source_date': row1.get('issue_date', ''),
                    'target_date': row2.get('issue_date', ''),
                    'ngram_similarity': similarity['ngram_similarity'],
                    'shingle_similarity': similarity['shingle_similarity'],
                    'combined_similarity': similarity['combined_similarity'],
                    'shared_content': similarity['shared_content'],
                    'source_context': similarity['source_context'],
                    'target_context': similarity['target_context']
                }
                
                if use_windows:
                    result.update({
                        'source_window_id': row1.get('window_id', ''),
                        'target_window_id': row2.get('window_id', ''),
                        'source_combined_id': row1.get('combined_id', ''),
                        'target_combined_id': row2.get('combined_id', ''),
                        'source_start_word': row1.get('start_word', ''),
                        'source_end_word': row1.get('end_word', ''),
                        'target_start_word': row2.get('start_word', ''),
                        'target_end_word': row2.get('end_word', ''),
                        'source_total_page_words': row1.get('total_page_words', ''),
                        'target_total_page_words': row2.get('total_page_words', '')
                    })
                
                results.append(result)
    
    return results

def find_text_reuse_windowed(metadata: List[Dict],
                             similarity_threshold: float = 0.12,
                             ngram_size: int = 4,
                             shingle_size: int = 5,
                             same_pub: bool = False,
                             use_windows: bool = True,
                             window_size: int = 200,
                             overlap: int = 50,
                             n_cores: int = None) -> List[Dict]:
    """
    Find text reuse using parallelization for faster processing.
    
    Args:
        metadata: List of dictionaries with text data
        n_cores: Number of CPU cores to use. If None, uses all available cores.
    
    Returns:
        List of dictionaries with reuse matches
    """
    print("\n" + "="*70)
    print("TEXT REUSE DETECTION WITH PARALLEL PROCESSING")
    print("="*70)
    
    # Determine number of cores
    if n_cores is None:
        n_cores = cpu_count()
    print(f"Using {n_cores} CPU cores for parallel processing")
    
    # Prepare data
    if use_windows:
        print(f"Creating overlapping windows (size={window_size}, overlap={overlap})...")
        windowed_data = []
        for row in metadata:
            windows = create_text_windows(row.get('text', ''), window_size, overlap)
            for window in windows:
                windowed_data.append({
                    'publication_name': row.get('publication_name', ''),
                    'page_id': row.get('page_id', ''),
                    'issue_date': row.get('issue_date', ''),
                    'window_id': window['window_id'],
                    'combined_id': f"{row.get('page_id', '')}_w{window['window_id']}",
                    'start_word': window['start_word'],
                    'end_word': window['end_word'],
                    'text': window['text'],
                    'total_page_words': window['total_words']
                })
        working_data = windowed_data
        print(f"Created {len(working_data)} text windows from {len(metadata)} pages")
    else:
        working_data = []
        for i, row in enumerate(metadata):
            row_copy = row.copy()
            if 'page_id' not in row_copy or not row_copy['page_id']:
                row_copy['page_id'] = str(i)
            working_data.append(row_copy)
    
    print(f"Threshold: {similarity_threshold}")
    print(f"N-gram size: {ngram_size}")
    print(f"Shingle size: {shingle_size}")
    
    # Calculate total comparisons
    n = len(working_data)
    total_comparisons = n * (n - 1) // 2
    print(f"Total segments to compare: {n}")
    print(f"Total pairwise comparisons: {total_comparisons:,}")
    
    # Divide work into chunks for parallel processing
    indices = list(range(len(working_data)))
    chunk_size = max(1, len(indices) // (n_cores * 4))  # Create more chunks than cores for better load balancing
    chunks = [indices[i:i + chunk_size] for i in range(0, len(indices), chunk_size)]
    
    print(f"Dividing work into {len(chunks)} chunks for parallel processing...")
    
    # Prepare arguments for parallel processing
    chunk_args = [
        (chunk, working_data, similarity_threshold, ngram_size, shingle_size, same_pub, use_windows)
        for chunk in chunks
    ]
    
    # Process chunks in parallel
    print("Starting parallel processing...")
    with Pool(processes=n_cores) as pool:
        chunk_results = pool.map(process_comparison_chunk, chunk_args)
    
    # Combine results from all chunks
    results = []
    for chunk_result in chunk_results:
        results.extend(chunk_result)
    
    print(f"Matches found: {len(results)}")
    
    return results

# ============================================================================
# ANALYSIS AND CLASSIFICATION
# ============================================================================

def classify_reuse_type(row: Dict) -> str:
    """Classify the type of text reuse based on similarity scores."""
    combined_sim = row['combined_similarity']
    ngram_sim = row['ngram_similarity']
    
    # High similarity suggests substantial reuse
    if combined_sim > 0.7 and ngram_sim > 0.6:
        return 'substantial_reuse'
    elif combined_sim > 0.4 and ngram_sim > 0.3:
        return 'moderate_reuse'
    elif combined_sim > 0.2 and ngram_sim > 0.15:
        return 'partial_reuse'
    else:
        return 'minimal_reuse'

def filter_boilerplate(reuse_data: List[Dict], min_occurrences: int = 5) -> List[Dict]:
    """
    Improved boilerplate filtering: removes specific repeated content, not entire pairs
    """
    if len(reuse_data) == 0:
        return reuse_data
    
    print("\n" + "="*70)
    print("BOILERPLATE FILTERING")
    print("="*70)
    
    original_count = len(reuse_data)
    
    # Stage 1: Remove very short shared content
    min_words = 10
    filtered_data = []
    short_count = 0
    
    for row in reuse_data:
        if len(row['shared_content'].split()) >= min_words:
            filtered_data.append(row)
        else:
            short_count += 1
    
    print(f"Stage 1: Removed {short_count} matches with <{min_words} words")
    
    # Stage 2: Identify repeated content
    content_counts = Counter(row['shared_content'] for row in filtered_data)
    
    print(f"\nTop 10 most frequent shared content pieces:")
    for content, count in content_counts.most_common(10):
        print(f"  [{count}x] {content[:100]}...")
    
    # Only remove if appears 5+ times
    frequent_content = {content for content, count in content_counts.items() if count >= min_occurrences}
    
    result = []
    boilerplate_count = 0
    for row in filtered_data:
        if row['shared_content'] not in frequent_content:
            result.append(row)
        else:
            boilerplate_count += 1
    
    print(f"\nStage 2: Identified {len(frequent_content)} pieces appearing {min_occurrences}+ times")
    print(f"  Removing {boilerplate_count} boilerplate matches")
    
    print(f"\nFiltering summary:")
    print(f"  Started with: {original_count}")
    print(f"  Removed short: {short_count}")
    print(f"  Removed boilerplate: {boilerplate_count}")
    print(f"  Final count: {len(result)}")
    print("="*70)
    
    return result

def create_review_sample(reuse_data: List[Dict], sample_size: int = 50) -> List[Dict]:
    """Create a sample for manual review."""
    import random
    
    # Separate by similarity ranges
    high_sim = [r for r in reuse_data if r['combined_similarity'] > 0.5]
    med_sim = [r for r in reuse_data if 0.25 < r['combined_similarity'] <= 0.5]
    low_sim = [r for r in reuse_data if r['combined_similarity'] <= 0.25]
    
    # Sample from each range
    sample = []
    for sim_range in [high_sim, med_sim, low_sim]:
        if len(sim_range) > 0:
            n_samples = min(len(sim_range), sample_size // 3)
            sample.extend(random.sample(sim_range, n_samples))
    
    # Add empty columns for manual review
    for row in sample:
        row['manual_review_notes'] = ''
        row['verified_reuse_type'] = ''
    
    return sample

def calculate_statistics(reuse_data: List[Dict]) -> Dict:
    """Calculate summary statistics from reuse data."""
    if not reuse_data:
        return {}
    
    combined_sims = [r['combined_similarity'] for r in reuse_data]
    ngram_sims = [r['ngram_similarity'] for r in reuse_data]
    shingle_sims = [r['shingle_similarity'] for r in reuse_data]
    shared_lengths = [len(r['shared_content'].split()) for r in reuse_data]
    
    return {
        'count': len(reuse_data),
        'avg_combined_sim': sum(combined_sims) / len(combined_sims),
        'avg_ngram_sim': sum(ngram_sims) / len(ngram_sims),
        'avg_shingle_sim': sum(shingle_sims) / len(shingle_sims),
        'max_combined_sim': max(combined_sims),
        'min_combined_sim': min(combined_sims),
        'avg_shared_length': sum(shared_lengths) / len(shared_lengths)
    }

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main(data_filepath: str = None, n_cores: int = None):
    """
    Main execution function.
    
    Args:
        data_filepath: Path to input CSV file
        n_cores: Number of CPU cores to use. If None, uses all available cores.
    """
    print("="*70)
    print("N-GRAMS AND JACCARD SIMILARITY TEXT REUSE ANALYSIS (PARALLEL)")
    print("Feminist Publications 1970-1975")
    print("="*70)
    
    # Configuration
    USE_WINDOWS = True  # Set to False for traditional full-page analysis
    WINDOW_SIZE = 200   # Words per window
    OVERLAP = 50        # Overlapping words between windows
    SIMILARITY_THRESHOLD = 0.12
    
    # Create output directory
    output_dir = 'reuse_results'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory: {output_dir}")

    # Load data
    metadata = load_and_prepare_metadata(data_filepath)
    
    # Run analysis with parallel processing
    if USE_WINDOWS:
        print(f"\n🪟 Using windowed analysis (window_size={WINDOW_SIZE}, overlap={OVERLAP})")
        analysis_suffix = "windowed_parallel"
    else:
        print("\n📄 Using full-page analysis")
        analysis_suffix = "fullpage_parallel"
    
    reuse_results = find_text_reuse_windowed(
        metadata,
        similarity_threshold=SIMILARITY_THRESHOLD,
        ngram_size=4,
        shingle_size=5,
        same_pub=False,
        use_windows=USE_WINDOWS,
        window_size=WINDOW_SIZE,
        overlap=OVERLAP,
        n_cores=n_cores
    )
    
    if len(reuse_results) == 0:
        print("No text reuse found above threshold.")
        return
    
    # Classify reuse types
    for row in reuse_results:
        row['reuse_type'] = classify_reuse_type(row)
    
    # Filter boilerplate
    reuse_filtered = filter_boilerplate(reuse_results)
    
    # Save results with appropriate filenames
    results_file = os.path.join(output_dir, f'text_reuse_ngrams_{analysis_suffix}.csv')
    filtered_file = os.path.join(output_dir, f'text_reuse_ngrams_{analysis_suffix}_filtered.csv')
    
    save_csv_data(results_file, reuse_results)
    save_csv_data(filtered_file, reuse_filtered)
    
    # Create review sample
    if len(reuse_filtered) > 0:
        review_sample = create_review_sample(reuse_filtered)
        review_file = os.path.join(output_dir, f'text_reuse_ngrams_{analysis_suffix}_review.csv')
        save_csv_data(review_file, review_sample)
    
    # Summary statistics
    print("\n" + "="*50)
    print("ANALYSIS RESULTS")
    print("="*50)
    print(f"Total matches found: {len(reuse_results)}")
    print(f"After filtering: {len(reuse_filtered)}")
    
    if len(reuse_filtered) > 0:
        # Count reuse types
        reuse_type_counts = Counter(r['reuse_type'] for r in reuse_filtered)
        print("\nReuse type distribution:")
        for reuse_type, count in reuse_type_counts.most_common():
            print(f"  {reuse_type}: {count}")
        
        # Calculate statistics
        stats = calculate_statistics(reuse_filtered)
        print(f"\nSimilarity statistics:")
        print(f"Average combined similarity: {stats['avg_combined_sim']:.3f}")
        print(f"Average n-gram similarity: {stats['avg_ngram_sim']:.3f}")
        print(f"Average shingle similarity: {stats['avg_shingle_sim']:.3f}")
        print(f"Max combined similarity: {stats['max_combined_sim']:.3f}")
        print(f"Min combined similarity: {stats['min_combined_sim']:.3f}")
        print(f"Average shared content length: {stats['avg_shared_length']:.1f} words")
        
        # Window-specific stats if applicable
        if USE_WINDOWS:
            unique_pairs = set()
            for row in reuse_filtered:
                pair = (row['source_page_id'], row['target_page_id'])
                unique_pairs.add(pair)
            print(f"Unique page pairs with reuse: {len(unique_pairs)}")
    
    # Summary of outputs
    print(f"\n" + "="*50)
    print("✅ N-gram text reuse analysis complete!")
    print("="*50)
    print(f"Generated files in {output_dir}/:")
    print(f"  - text_reuse_ngrams_{analysis_suffix}.csv (all matches with context)")
    print(f"  - text_reuse_ngrams_{analysis_suffix}_filtered.csv (without boilerplate)")
    if len(reuse_filtered) > 0:
        print(f"  - text_reuse_ngrams_{analysis_suffix}_review.csv (sample for manual review)")
    print("="*50)

if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    data_filepath = "zinepage_metadata.csv"
    n_cores = 8
    
    if len(sys.argv) > 1:
        data_filepath = sys.argv[1]
    
    if len(sys.argv) > 2:
        try:
            n_cores = int(sys.argv[2])
            print(f"Using {n_cores} cores as specified in command line")
        except ValueError:
            print(f"Invalid number of cores: {sys.argv[2]}, using all available cores")
    
    main(data_filepath=data_filepath, n_cores=n_cores)