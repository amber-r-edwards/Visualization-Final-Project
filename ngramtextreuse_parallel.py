"""
N-grams and Jaccard Similarity Text Reuse Analysis (PARALLEL VERSION)
=================================================

This script analyzes text reuse between feminist publications using n-grams 
and Jaccard similarity, with parallel processing for faster execution.

"""

import pandas as pd # type: ignore
import numpy as np
from datetime import datetime
import re
from pathlib import Path
import json
import os
from typing import Set, Dict, List, Tuple
from multiprocessing import Pool, cpu_count
from functools import partial

# ============================================================================
# TEXT PREPROCESSING AND N-GRAM FUNCTIONS
# ============================================================================

def clean_text(text: str) -> str:
    """Clean and normalize text for n-gram analysis."""
    if pd.isna(text) or not text:
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

def load_and_prepare_metadata(filepath: str = 'zinepage_metadata.csv') -> pd.DataFrame:
    """Load and prepare metadata for analysis."""
    metadata = pd.read_csv(filepath)
    
    # Add page_id if not present
    if 'page_id' not in metadata.columns:
        metadata['page_id'] = range(len(metadata))
        metadata.to_csv(filepath, index=False)
        print("✅ Added page_id column to page_metadata.csv")
    
    # Convert dates
    metadata['issue_date'] = pd.to_datetime(metadata['issue_date'])
    
    # Sort by date for directionality
    metadata = metadata.sort_values('issue_date').reset_index(drop=True)
    
    # Clean text and calculate lengths
    metadata['text_clean'] = metadata['text'].apply(clean_text)
    metadata['text_length'] = metadata['text_clean'].apply(lambda x: len(x.split()))
    
    # Filter out very short texts
    metadata = metadata[metadata['text_length'] >= 10].copy()
    
    print(f"Loaded {len(metadata)} pages from {metadata['publication_name'].nunique()} publications")
    print(f"Date range: {metadata['issue_date'].min()} to {metadata['issue_date'].max()}")
    
    return metadata

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
        row1 = working_data.iloc[idx1]
        
        # Compare with all subsequent segments
        for idx2 in range(idx1 + 1, len(working_data)):
            row2 = working_data.iloc[idx2]
            
            # Skip if same publication and not comparing within same pub
            if not same_pub and row1['publication'] == row2['publication']:
                continue
            
            # Skip if comparing same page to itself (in windowed mode)
            if use_windows and row1['page_id'] == row2['page_id']:
                continue
            
            # Calculate similarity
            similarity = calculate_text_similarity(
                row1['text'],
                row2['text'],
                ngram_size=ngram_size,
                shingle_size=shingle_size
            )
            
            # Only keep if above threshold
            if similarity['combined_similarity'] >= similarity_threshold:
                result = {
                    'source_publication': row1['publication'],
                    'target_publication': row2['publication'],
                    'source_page_id': row1['page_id'],
                    'target_page_id': row2['page_id'],
                    'source_date': row1['date'],
                    'target_date': row2['date'],
                    'ngram_similarity': similarity['ngram_similarity'],
                    'shingle_similarity': similarity['shingle_similarity'],
                    'combined_similarity': similarity['combined_similarity'],
                    'shared_content': similarity['shared_content'],
                    'source_context': similarity['source_context'],
                    'target_context': similarity['target_context']
                }
                
                if use_windows:
                    result.update({
                        'source_window_id': row1['window_id'],
                        'target_window_id': row2['window_id'],
                        'source_combined_id': row1['combined_id'],
                        'target_combined_id': row2['combined_id'],
                        'source_start_word': row1['start_word'],
                        'source_end_word': row1['end_word'],
                        'target_start_word': row2['start_word'],
                        'target_end_word': row2['end_word'],
                        'source_total_page_words': row1['total_page_words'],
                        'target_total_page_words': row2['total_page_words']
                    })
                
                results.append(result)
    
    return results

def find_text_reuse_windowed(metadata: pd.DataFrame,
                             similarity_threshold: float = 0.12,
                             ngram_size: int = 4,
                             shingle_size: int = 5,
                             same_pub: bool = False,
                             use_windows: bool = True,
                             window_size: int = 200,
                             overlap: int = 50,
                             n_cores: int = None) -> pd.DataFrame:
    """
    Find text reuse using parallelization for faster processing.
    
    Args:
        n_cores: Number of CPU cores to use. If None, uses all available cores.
    """
    print("\n" + "="*70)
    print("TEXT REUSE DETECTION WITH PARALLEL PROCESSING")
    print("="*70)
    
    # Determine number of cores
    if n_cores is None:
        n_cores = cpu_count()
    print(f"Using {n_cores} CPU cores for parallel processing")
    
    # Prepare data (same as original)
    if use_windows:
        print(f"Creating overlapping windows (size={window_size}, overlap={overlap})...")
        windowed_data = []
        for idx, row in metadata.iterrows():
            windows = create_text_windows(row['text'], window_size, overlap)
            for window in windows:
                windowed_data.append({
                    'publication': row['publication'],
                    'page_id': row['page_id'],
                    'date': row['date'],
                    'window_id': window['window_id'],
                    'combined_id': f"{row['page_id']}_w{window['window_id']}",
                    'start_word': window['start_word'],
                    'end_word': window['end_word'],
                    'text': window['text'],
                    'total_page_words': window['total_words']
                })
        working_data = pd.DataFrame(windowed_data)
        print(f"Created {len(working_data)} text windows from {len(metadata)} pages")
    else:
        working_data = metadata.copy()
        working_data['page_id'] = working_data.index
    
    print(f"Threshold: {similarity_threshold}")
    print(f"N-gram size: {ngram_size}")
    print(f"Shingle size: {shingle_size}")
    
    # Calculate total comparisons
    n = len(working_data)
    total_comparisons = n * (n - 1) // 2
    print(f"Total segments to compare: {n}")
    print(f"Total pairwise comparisons: {total_comparisons:,}")
    
    # Divide work into chunks for parallel processing
    # Each chunk is a set of indices for the outer loop
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
    
    return pd.DataFrame(results)

# ============================================================================
# ANALYSIS AND CLASSIFICATION
# ============================================================================

def classify_reuse_type(row):
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

def filter_boilerplate(reuse_df: pd.DataFrame, min_occurrences: int = 5) -> pd.DataFrame:
    """
    Improved boilerplate filtering: removes specific repeated content, not entire pairs
    """
    if len(reuse_df) == 0:
        return reuse_df
    
    print("\n" + "="*70)
    print("BOILERPLATE FILTERING")
    print("="*70)
    
    original_count = len(reuse_df)
    
    # Stage 1: Remove very short shared content
    min_words = 10
    short_content = reuse_df['shared_content'].apply(lambda x: len(x.split()) < min_words)
    reuse_df = reuse_df[~short_content].copy()
    print(f"Stage 1: Removed {short_content.sum()} matches with <{min_words} words")
    
    # Stage 2: Identify repeated content
    content_counts = reuse_df['shared_content'].value_counts()
    print(f"\nTop 10 most frequent shared content pieces:")
    for content, count in content_counts.head(10).items():
        print(f"  [{count}x] {content[:100]}...")
    
    # Only remove if appears 5+ times
    frequent_content = content_counts[content_counts >= min_occurrences].index
    is_boilerplate = reuse_df['shared_content'].isin(frequent_content)
    
    print(f"\nStage 2: Identified {len(frequent_content)} pieces appearing {min_occurrences}+ times")
    print(f"  Removing {is_boilerplate.sum()} boilerplate matches")
    
    reuse_df = reuse_df[~is_boilerplate].copy()
    
    print(f"\nFiltering summary:")
    print(f"  Started with: {original_count}")
    print(f"  Removed short: {short_content.sum()}")
    print(f"  Removed boilerplate: {is_boilerplate.sum()}")
    print(f"  Final count: {len(reuse_df)}")
    print("="*70)
    
    return reuse_df

def create_review_sample(reuse_df: pd.DataFrame, metadata: pd.DataFrame, sample_size: int = 50) -> pd.DataFrame:
    """Create a sample for manual review (now simplified since context is already in main results)."""
    # Sample across different similarity ranges
    high_sim = reuse_df[reuse_df['combined_similarity'] > 0.5]
    med_sim = reuse_df[(reuse_df['combined_similarity'] > 0.25) & (reuse_df['combined_similarity'] <= 0.5)]
    low_sim = reuse_df[reuse_df['combined_similarity'] <= 0.25]
    
    # Sample from each range
    sample_high = high_sim.sample(min(len(high_sim), sample_size // 3)) if len(high_sim) > 0 else pd.DataFrame()
    sample_med = med_sim.sample(min(len(med_sim), sample_size // 3)) if len(med_sim) > 0 else pd.DataFrame()
    sample_low = low_sim.sample(min(len(low_sim), sample_size // 3)) if len(low_sim) > 0 else pd.DataFrame()
    
    # Combine samples
    review_sample = pd.concat([sample_high, sample_med, sample_low], ignore_index=True)
    
    # Add empty columns for manual review
    review_sample['manual_review_notes'] = ''
    review_sample['verified_reuse_type'] = ''
    
    return review_sample

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main(n_cores=None):
    """
    Main execution function.
    
    Args:
        n_cores: Number of CPU cores to use. If None, uses all available cores.
    """
    print("="*70)
    print("N-GRAMS AND JACCARD SIMILARITY TEXT REUSE ANALYSIS (PARALLEL)")
    print("Feminist Publications 1973-1974")
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
    metadata = load_and_prepare_metadata()
    
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
    reuse_results['reuse_type'] = reuse_results.apply(classify_reuse_type, axis=1)
    
    # Filter boilerplate
    reuse_filtered = filter_boilerplate(reuse_results)
    
    # Save results with appropriate filenames
    results_file = os.path.join(output_dir, f'text_reuse_ngrams_{analysis_suffix}.csv')
    filtered_file = os.path.join(output_dir, f'text_reuse_ngrams_{analysis_suffix}_filtered.csv')
    
    reuse_results.to_csv(results_file, index=False)
    reuse_filtered.to_csv(filtered_file, index=False)
    
    # Create review sample
    if len(reuse_filtered) > 0:
        review_sample = create_review_sample(reuse_filtered, metadata)
        review_file = os.path.join(output_dir, f'text_reuse_ngrams_{analysis_suffix}_review.csv')
        review_sample.to_csv(review_file, index=False)
    
    # Summary statistics
    print("\n" + "="*50)
    print("ANALYSIS RESULTS")
    print("="*50)
    print(f"Total matches found: {len(reuse_results)}")
    print(f"After filtering: {len(reuse_filtered)}")
    
    if len(reuse_filtered) > 0:
        print("\nReuse type distribution:")
        print(reuse_filtered['reuse_type'].value_counts())
        
        print(f"\nSimilarity statistics:")
        print(f"Average combined similarity: {reuse_filtered['combined_similarity'].mean():.3f}")
        print(f"Average n-gram similarity: {reuse_filtered['ngram_similarity'].mean():.3f}")
        print(f"Average shingle similarity: {reuse_filtered['shingle_similarity'].mean():.3f}")
        
        print(f"Max combined similarity: {reuse_filtered['combined_similarity'].max():.3f}")
        print(f"Min combined similarity: {reuse_filtered['combined_similarity'].min():.3f}")
        
        # Show shared content statistics
        avg_shared_length = reuse_filtered['shared_content'].apply(lambda x: len(x.split())).mean()
        print(f"Average shared content length: {avg_shared_length:.1f} words")
        
        # Window-specific stats if applicable
        if USE_WINDOWS:
            unique_page_pairs = reuse_filtered[['source_page_id', 'target_page_id']].drop_duplicates()
            print(f"Unique page pairs with reuse: {len(unique_page_pairs)}")
    
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
    
    # Allow specifying number of cores as command line argument
    n_cores = None
    if len(sys.argv) > 1:
        try:
            n_cores = int(sys.argv[1])
            print(f"Using {n_cores} cores as specified in command line")
        except ValueError:
            print(f"Invalid number of cores: {sys.argv[1]}, using all available cores")
    
    main(n_cores=n_cores)
