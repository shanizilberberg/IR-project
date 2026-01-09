from flask import Flask, request, jsonify
from collections import defaultdict, Counter
import math
import pickle
import io
import os
import re
import time

from google.cloud import storage
from inverted_index_gcp import InvertedIndex



# Tokenizer


RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

corpus_stopwords = ['category', 'references', 'also', 'links', 'extenal', 'see', 'thumb']

english_stopwords = frozenset([
    "during","as","whom","no","so","were","then","on","once","very","any","it's","it","be",
    "why","over","they","am","before","because","been","than","will","not","those","had",
    "this","through","again","into","did","should","above","does","now","down","their",
    "few","and","some","do","the","after","them","out","where","at","against","has","all",
    "being","he","its","that","more","by","who","there","too","other","an","here","between",
    "is","below","what","when","i","with","her","same","for","each","which","such","up",
    "only","most","of","me","she","in","a","if","but","these","him","both","my","yourself",
    "to","are","itself","themselves","just","have","don't","how","about","can","our",
    "from","under","while","off","or","your"
])

all_stopwords = english_stopwords.union(corpus_stopwords)


def tokenize(text):
    """
    Tokenize text using the staff-provided tokenizer and remove stopwords.
    """
    tokens = [token.group() for token in RE_WORD.finditer(text.lower())]
    return [tok for tok in tokens if tok not in all_stopwords]


# Alias for compatibility
staff_tokenize = tokenize


# GCS CONFIG


GCS_BUCKET = "badashbucket1"

INDEX_PKL_PATH = "PKL/postings_gcp_index.pkl"
POSTINGS_DIR = "index_files/bin_files"

TITLES_INDEX_PATH = "PKL/titles_index.pkl"
ANCHOR_INDEX_PATH = "PKL/anchor_index.pkl"
DOC_TITLES_MAPPING_PATH = "PKL/doc_titles.pkl"

PAGERANK_PATH = "PKL/pagerank.pkl"
PAGEVIEWS_PATH = "PKL/pageviews.pkl"



# GCS loader


def load_pickle_from_gcs(bucket_name, blob_path):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    data = blob.download_as_bytes()
    return pickle.load(io.BytesIO(data))


def load_body_index():
    print("Loading index from", INDEX_PKL_PATH, "...")
    idx = load_pickle_from_gcs(GCS_BUCKET, INDEX_PKL_PATH)
    print("Index loaded successfully")
    return idx


def load_titles_index():
    print("Loading titles index from", TITLES_INDEX_PATH, "...")
    idx = load_pickle_from_gcs(GCS_BUCKET, TITLES_INDEX_PATH)
    print("Titles index loaded successfully")
    return idx


def load_anchor_index():
    print("Loading anchor index from", ANCHOR_INDEX_PATH, "...")
    idx = load_pickle_from_gcs(GCS_BUCKET, ANCHOR_INDEX_PATH)
    print("Anchor index loaded successfully")
    return idx


def load_doc_titles():
    print("Loading doc titles from", DOC_TITLES_MAPPING_PATH, "...")
    titles = load_pickle_from_gcs(GCS_BUCKET, DOC_TITLES_MAPPING_PATH)
    print("Doc titles loaded successfully. Total documents:", len(titles))
    return titles


def load_pagerank():
    print("Loading pagerank from", PAGERANK_PATH, "...")
    pr = load_pickle_from_gcs(GCS_BUCKET, PAGERANK_PATH)
    print("Pagerank loaded successfully")
    return pr


def load_pageviews():
    print("Loading pageviews from", PAGEVIEWS_PATH, "...")
    pv = load_pickle_from_gcs(GCS_BUCKET, PAGEVIEWS_PATH)
    print("Pageviews loaded successfully")
    return pv



# Helper: read posting list with LIMIT (no caching)


def read_postings_limited(index_obj, postings_dir, term, bucket, limit):
    """
    Read up to `limit` postings for a given term.
    """
    postings = []
    for i, item in enumerate(index_obj.read_a_posting_list(postings_dir, term, bucket)):
        if i >= limit:
            break
        postings.append(item)
    return postings


def read_posting_list(index_obj, term, max_items=100000):
    """
    Read posting list for a term from the index.
    Returns list of (doc_id, frequency) tuples.
    """
    if not hasattr(index_obj, 'df') or term not in index_obj.df:
        return []

    try:
        # Check if posting list is directly available in memory
        if hasattr(index_obj, 'posting_lists') and term in index_obj.posting_lists:
            pl = index_obj.posting_lists[term]
            return pl[:max_items] if len(pl) > max_items else pl

        # Check if posting_locs exists to read from GCS
        if not hasattr(index_obj, 'posting_locs') or term not in index_obj.posting_locs:
            return []

        postings = []
        import struct

        # Create GCS client and read directly
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET)

        locs = index_obj.posting_locs[term]

        # Get the document frequency to know how many entries to read
        df = index_obj.df[term]
        TUPLE_SIZE = 6  # (doc_id: 4 bytes, freq: 2 bytes)
        bytes_to_read = df * TUPLE_SIZE

        for loc_entry in locs:
            # Parse the location entry
            if isinstance(loc_entry, tuple) and len(loc_entry) >= 2:
                bin_path = loc_entry[0]
                offset = loc_entry[1]
            else:
                continue

            # Clean the path - replace ALL backslashes with forward slashes
            bin_path_clean = str(bin_path).replace('\\', '/')

            # Build full path
            if not bin_path_clean.startswith('index_files/'):
                postings_dir = POSTINGS_DIR.replace('\\', '/')
                full_path = f"{postings_dir}/{bin_path_clean}"
            else:
                full_path = bin_path_clean

            # Remove any double slashes
            full_path = full_path.replace('//', '/')

            try:
                # Open the blob and read from specific offset
                blob = bucket.blob(full_path)

                # Use blob.open() for better control
                with blob.open('rb') as f:
                    # Seek to the offset
                    f.seek(offset)
                    # Read exactly the bytes we need
                    data = f.read(bytes_to_read)

                # Validate we got the right amount of data
                if len(data) < bytes_to_read:
                    print(f"Warning: Expected {bytes_to_read} bytes but got {len(data)} for term '{term}'")

                # Parse the posting list from the binary data
                num_tuples = min(len(data) // TUPLE_SIZE, df)

                for i in range(num_tuples):
                    if len(postings) >= max_items:
                        break
                    start_pos = i * TUPLE_SIZE
                    doc_id, freq = struct.unpack('<IH', data[start_pos:start_pos + TUPLE_SIZE])
                    postings.append((doc_id, freq))

            except Exception as e:
                print(f"Error reading bin file {full_path} at offset {offset}: {e}")
                import traceback
                traceback.print_exc()
                continue

            if len(postings) >= max_items:
                break

        return postings

    except Exception as e:
        print(f"Error reading posting list for term '{term}': {e}")
        import traceback
        traceback.print_exc()
        return []



# Load data from GCS (static preload)


print("Starting to load data from GCS bucket:", GCS_BUCKET)
index = load_body_index()
titles_index = load_titles_index()
anchor_index = load_anchor_index()
doc_titles = load_doc_titles()
pagerank = load_pagerank()
pageviews = load_pageviews()
print("All data loaded successfully from GCS!")

# Debug: Print index attributes
print("\n--- Index Debug Info ---")
print(f"Body index attributes: {dir(index)[:10]}")
print(f"Body index has 'posting_locs': {hasattr(index, 'posting_locs')}")
print(f"Body index has 'posting_lists': {hasattr(index, 'posting_lists')}")
if hasattr(index, 'df'):
    sample_terms = list(index.df.keys())[:3]
    print(f"Sample terms in body index: {sample_terms}")
    for term in sample_terms:
        if hasattr(index, 'posting_locs') and term in index.posting_locs:
            print(f"  '{term}' posting_locs: {index.posting_locs[term][:2] if len(index.posting_locs[term]) > 1 else index.posting_locs[term]}")
print("--- End Debug Info ---\n")



# Pre-compute title term -> docs mapping


title_term_to_docs = defaultdict(set)
for doc_id, title in doc_titles.items():
    if title:
        for term in tokenize(title):
            title_term_to_docs[term].add(doc_id)


def get_title_tokens(title):
    """
    Tokenize a title.
    """
    return set(tokenize(title))



# Flask app


app = Flask(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False


# /search


@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if not query:
        return jsonify(res)

    tokens = tokenize(query)
    if not tokens:
        return jsonify(res)

    query_terms = set(tokens)
    scores = defaultdict(float)
    doc_matched_terms = defaultdict(set)

    # Weights for ranking
    TITLE_WEIGHT = 5.0
    PAGERANK_WEIGHT = 2.0
    PAGEVIEW_WEIGHT = 0.001

    # Title matching - main signal
    for term in query_terms:
        for doc_id in title_term_to_docs.get(term, []):
            scores[doc_id] += TITLE_WEIGHT
            doc_matched_terms[doc_id].add(term)

    # Filter out docs that don't match enough terms
    min_required = 1 if len(query_terms) == 1 else max(1, len(query_terms) // 2)

    for doc_id in list(scores.keys()):
        matched = len(doc_matched_terms[doc_id])
        if matched < min_required:
            del scores[doc_id]
            continue

        # Boost by number of matched terms
        scores[doc_id] *= (1 + matched)

        # Strong boost if all query terms appear in title
        title = doc_titles.get(doc_id, "")
        if title and query_terms.issubset(get_title_tokens(title)):
            scores[doc_id] *= 3

    # Add PageRank & PageViews boost
    for doc_id in scores:
        pr_score = pagerank.get(doc_id, 0)
        pv_score = pageviews.get(doc_id, 0)

        scores[doc_id] += PAGERANK_WEIGHT * math.log(pr_score + 1, 10)
        scores[doc_id] += PAGEVIEW_WEIGHT * math.log(pv_score + 1, 10)

    # Sort and return top 1000
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:1000]
    return jsonify([(doc_id, doc_titles.get(doc_id, "")) for doc_id, _ in ranked])


# /search_body


@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the
        staff-provided tokenizer from Assignment 3 (GCP part) to do the
        tokenization and remove stopwords.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)

    tokens = staff_tokenize(query)
    if not tokens:
        return jsonify([])

    q_tf = Counter(tokens)

    # Determine corpus size
    N = None
    if hasattr(index, 'N'):
        N = index.N
    elif hasattr(index, 'n'):
        N = index.n
    elif hasattr(index, 'DL') and hasattr(index.DL, '__len__'):
        N = len(index.DL)

    if N is None or N == 0:
        N = len(doc_titles)

    doc_dot = defaultdict(float)
    doc_sq = defaultdict(float)
    doc_term_freq = defaultdict(int)  # Track raw term frequency
    q_sq = 0.0

    for term, tf_q in q_tf.items():
        df = index.df.get(term, 0) if hasattr(index, "df") else 0
        if df <= 0:
            continue

        idf = math.log10(N / df)
        w_tq = tf_q * idf
        q_sq += w_tq * w_tq

        pl = read_posting_list(index, term)

        # Sort posting list by term frequency (descending) to get most relevant docs first
        pl_sorted = sorted(pl, key=lambda x: x[1], reverse=True)

        # Only process top documents per term to avoid memory issues
        for doc_id, tf_d in pl_sorted[:10000]:
            w_td = tf_d * idf
            doc_dot[doc_id] += w_td * w_tq
            doc_sq[doc_id] += w_td * w_td
            doc_term_freq[doc_id] += tf_d  # Sum raw frequencies

    q_norm = math.sqrt(q_sq)
    if q_norm == 0.0:
        return jsonify([])

    scored = []
    for doc_id, num in doc_dot.items():
        den = math.sqrt(doc_sq[doc_id]) * q_norm
        if den > 0:
            cosine_sim = num / den
            # Boost score by term frequency (documents with more occurrences rank higher)
            final_score = cosine_sim * (1 + math.log10(doc_term_freq[doc_id] + 1))
            scored.append((int(doc_id), final_score))

    scored.sort(key=lambda x: x[1], reverse=True)
    top100 = scored[:100]

    res = [(doc_id, doc_titles.get(doc_id, f"Unknown Title ({doc_id})")) for doc_id, _ in top100]

    return jsonify(res)



# /search_title


@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF
        DISTINCT QUERY WORDS that appear in the title. DO NOT use stemming. DO
        USE the staff-provided tokenizer from Assignment 3 (GCP part) to do the
        tokenization and remove stopwords. For example, a document
        with a title that matches two distinct query words will be ranked before a
        document with a title that matches only one distinct query word,
        regardless of the number of times the term appeared in the title (or
        query).

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if not query:
        return jsonify(res)

    q_terms = set(tokenize(query))
    doc_matches = defaultdict(set)

    for term in q_terms:
        for doc_id in title_term_to_docs.get(term, []):
            doc_matches[doc_id].add(term)

    ranked = sorted(doc_matches.items(), key=lambda x: len(x[1]), reverse=True)
    return jsonify([(doc_id, doc_titles.get(doc_id, "")) for doc_id, _ in ranked])



# /search_anchor


@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE ANCHOR TEXT of articles, ordered in descending order of the
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page.
        DO NOT use stemming. DO USE the staff-provided tokenizer from Assignment
        3 (GCP part) to do the tokenization and remove stopwords. For example,
        a document with a anchor text that matches two distinct query words will
        be ranked before a document with anchor text that matches only one
        distinct query word, regardless of the number of times the term appeared
        in the anchor text (or query).

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    '''
    query = request.args.get('query', '')
    if not query:
        return jsonify([])

    query_terms = staff_tokenize(query)
    if not query_terms:
        return jsonify([])

    query_terms_set = set(query_terms)

    # doc_id -> total frequency (sum of all term frequencies)
    doc_total_freq = defaultdict(int)
    # doc_id -> set of distinct terms
    doc_matched_terms = defaultdict(set)

    for term in query_terms_set:
        if not hasattr(anchor_index, 'df') or term not in anchor_index.df:
            continue

        try:
            posting_list = read_posting_list(anchor_index, term)

            # Process more postings per term for better coverage
            for doc_id, freq in posting_list[:10000]:
                doc_matched_terms[doc_id].add(term)
                doc_total_freq[doc_id] += freq  # Sum the frequencies
        except Exception as e:
            print(f"Error reading term '{term}': {e}")
            continue

    # Score = (number of distinct terms) * 10000 + total frequency
    # This prioritizes documents with more distinct terms, then by frequency
    scored = []
    for doc_id in doc_matched_terms:
        distinct_count = len(doc_matched_terms[doc_id])
        total_freq = doc_total_freq[doc_id]
        score = distinct_count * 10000 + total_freq
        scored.append((doc_id, score))

    scored.sort(key=lambda x: x[1], reverse=True)

    # Return top 500 results - good balance between coverage and performance
    top_results = scored[:500]

    res = [(doc_id, doc_titles.get(doc_id, f"Unknown Title ({doc_id})")) for doc_id, _ in top_results]

    return jsonify(res)



# PageRank / PageViews


@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    res = [pagerank.get(int(doc_id), 0.0) for doc_id in wiki_ids]
    return jsonify(res)


@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the
          provided list article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    res = [pageviews.get(int(doc_id), 0) for doc_id in wiki_ids]
    return jsonify(res)



# Main
def run(**options):
    app.run(**options)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)