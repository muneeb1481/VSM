import streamlit as st
import numpy as np
import math
from nltk.stem import WordNetLemmatizer
import ast
import os

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Custom CSS for styling
st.markdown("""
    <style>
        .header {
            font-size: 24px !important;
            font-weight: bold !important;
            color: #2e86ab !important;
            margin-bottom: 20px !important;
        }
        .subheader {
            font-size: 18px !important;
            color: #4a4a4a !important;
            margin-bottom: 10px !important;
        }
        .result-card {
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 15px;
            border-left: 5px solid #2e86ab;
        }
        .score {
            font-weight: bold;
            color: #e63946;
        }
        .footer {
            margin-top: 30px;
            padding-top: 10px;
            border-top: 1px solid #eee;
            font-size: 14px;
            color: #666;
        }
        .highlight {
            padding: 2px 5px;
            border-radius: 3px;
            font-weight: bold;
            color: #2e86ab;
        }
        .processed-terms {
            margin-bottom: 15px;
            font-size: 16px;
        }
        .alpha-control {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 15px;
        }
        .alpha-value {
            font-weight: bold;
            min-width: 60px;
            text-align: center;
        }
        .eval-section {
            margin-top: 30px;
            padding: 15px;
            border-radius: 10px;
        }
        .alpha-input {
            width: 80px;
        }
        .retrieved-doc {
    padding: 15px;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    margin-bottom: 15px;
    border-left: 5px solid #2e86ab;
}

.relevant .score {
    color: #28a745; /* green */
    font-weight: bold;
}

.not-relevant .score {
    color: #dc3545; /* red */
    font-weight: bold;
}

.retrieved-doc b {
    font-size: 18px;
}

    </style>
""", unsafe_allow_html=True)

# Load term-to-index mapping from file
def load_index_data(file_path):
    index_map = {}
    with open(file_path, 'r') as file:
        for line in file:
            if ' : ' in line:
                term, doc_ids = line.strip().split(' : ')
                index_map[term] = ast.literal_eval(doc_ids)
    return index_map

# Load vector space from .npy file
def load_vector_space(file_path):
    vector_space = np.load(file_path, allow_pickle=True)
    if isinstance(vector_space, np.ndarray):
        if vector_space.size == 1:
            return vector_space.item()
        else:
            return {i: vector_space[i] for i in range(len(vector_space))}
    elif isinstance(vector_space, dict):
        return vector_space
    else:
        raise ValueError("Unsupported vector space format")

# Read document content
def read_document_content(doc_id):
    file_path = f"Abstracts/{doc_id + 1}.txt"  # Adjust for 1-indexed filenames
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    else:
        return "Document content not found."

# Load all necessary data
@st.cache_data
def load_all_data():
    try:
        term_index_map = load_index_data('term_index_map.txt')
        vector_space_index = load_vector_space('vector_space_index.npy')
        inverted_index = load_index_data('inverted_index.txt')
        return term_index_map, vector_space_index, inverted_index
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return {}, {}, {}

# Preprocess query input
def preprocess_query(query):
    query = query.lower()
    query = ''.join([char for char in query if char.isalpha() or char == ' '])
    tokens = query.split()
    return [lemmatizer.lemmatize(token) for token in tokens]

# Convert query terms to vector
def query_to_vector(query_terms, term_index_map, inverted_index, num_docs):
    query_vector = np.zeros(len(term_index_map))
    for term in query_terms:
        if term in term_index_map:
            term_idx = term_index_map[term]
            df = len(inverted_index.get(term, {}))
            idf = math.log(num_docs / (df + 1)) if df > 0 else 0
            tf = query_terms.count(term)
            query_vector[term_idx] = tf * idf
    return query_vector

# Compute cosine similarity
def cosine_similarity(query_vector, doc_vector):
    dot_product = np.dot(query_vector, doc_vector)
    query_norm = np.linalg.norm(query_vector)
    doc_norm = np.linalg.norm(doc_vector)
    return dot_product / (query_norm * doc_norm) if (query_norm * doc_norm) != 0 else 0

# Rank documents based on similarity
def rank_documents(query_vector, vector_space_index, alpha=0.01):
    similarities = {
        doc_id: cosine_similarity(query_vector, doc_vector)
        for doc_id, doc_vector in vector_space_index.items()
        if cosine_similarity(query_vector, doc_vector) > alpha
    }
    return sorted(similarities.items(), key=lambda x: x[1], reverse=True)

GOLDEN_QUERIES = {
    "deep": [21, 24, 174, 175, 176, 177, 213, 245, 246, 247, 250, 254, 267, 273, 278, 279, 280, 281, 325, 345, 346, 347, 348, 352, 358, 360, 362, 374, 376, 380, 396, 397, 398, 401, 405, 415, 421, 432],
    
    "weak heuristic": [1, 35, 93, 101, 172, 174, 213, 257, 299, 306, 361, 391, 413, 429, 435],
    
    "principle component analysis": [45, 53, 102, 112, 134, 310, 311, 315, 357, 364, 426, 434, 445],
    
    "human interaction": [7, 10, 21, 22, 23, 26, 30, 83, 98, 101, 127, 145, 162, 164, 171, 174, 186, 187, 191, 194, 203, 230, 247, 249, 250, 255, 256, 265, 273, 289, 345, 369, 383, 391, 395, 403, 426, 428, 436, 444],
    
    "supervised kernel k-means cluster": [31, 53, 122, 123, 124, 125, 158, 167, 173, 177, 241, 242, 243, 244, 245, 264, 275, 280, 281, 291, 334, 368, 383, 427, 430, 447],
    
    "patients depression anxiety": [37, 40, 62, 72, 80, 168, 225, 259, 263, 328, 332, 333, 355, 368, 391, 400, 433, 447, 448],
    
    "local global clusters": [19, 21, 23, 26, 30, 38, 54, 76, 113, 125, 126, 134, 136, 156, 158, 168, 179, 196, 211, 215, 242, 257, 266, 271, 295, 331, 335, 336, 342, 361, 377, 394, 407, 423],
    
    "synergy analysis": [38, 102, 112, 134, 315, 357, 434],
    
    "github mashup apis": [178, 362],
    
    "Bayesian nonparametric": [16, 35, 39, 62, 65, 93, 117, 118, 119, 155, 196, 243, 244, 255, 271, 290, 324, 332, 370, 440, 442, 448],
    
    "diabetes and obesity": [72, 148, 391],
    
    "bootstrap": [181, 193, 379],
    
    "ensemble": [1, 2, 3, 5, 32, 52, 89, 105, 120, 171, 198, 229, 256, 262, 268, 284, 310, 311, 327, 352, 378, 386, 425],
    
    "markov": [11, 16, 22, 69, 110, 129, 149, 197, 230, 251, 257, 260, 289, 305, 312, 323, 335, 381, 439, 445],
    
    "prioritize and critical correlate": [37, 44, 52, 101, 104, 112, 118, 138, 140, 166, 195, 208, 218, 227, 230, 239, 250, 257, 281, 283, 298, 318, 322, 354, 370, 422, 426, 436]
}

# Header UI
st.markdown('<div class="header">Vector Space Model Search Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Information Retrieval Assignment #2</div>', unsafe_allow_html=True)

# Sidebar UI
with st.sidebar:
    st.markdown("### About")
    st.markdown("This search engine uses the Vector Space Model with TF-IDF weighting and cosine similarity for document ranking.")
    st.markdown("---")
    
    # Alpha control with buttons and direct input
    st.markdown("### Similarity Threshold (α)")
    
    # Create columns for the alpha control
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("−", help="Decrease alpha by 0.001"):
            st.session_state.alpha = max(0.0, st.session_state.get('alpha', 0.001) - 0.001)
    with col2:
        # Direct alpha input
        new_alpha = st.number_input(
            "Set alpha value:",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.get('alpha', 0.001),
            step=0.001,
            format="%.3f",
            key="alpha_input",
            label_visibility="collapsed"
        )
        st.session_state.alpha = new_alpha
    with col3:
        if st.button("+", help="Increase alpha by 0.001"):
            st.session_state.alpha = min(1.0, st.session_state.get('alpha', 0.001) + 0.001)
    
    st.markdown("---")
    st.markdown("### Developer Information")
    st.markdown("*Name:* Muneeb Ur Rehman")
    st.markdown("*Assignment:* IR Assignment #2")
    st.markdown("*Model:* Vector Space Model")

# Load data
term_index_map, vector_space_index, inverted_index = load_all_data()
num_docs = len(vector_space_index)

# Main search interface
with st.form(key="search_form"):
    query = st.text_input("Enter your search query:", placeholder="Type your query here...")
    submit = st.form_submit_button("Search")

# Search logic
if submit and query:
    alpha = st.session_state.get('alpha', 0.001)
    with st.spinner('Processing your query...'):
        processed_terms = preprocess_query(query)
        if not processed_terms:
            st.warning("No valid terms after preprocessing. Please try a different query.")
        else:
            st.markdown(f"""
                <div class="processed-terms">
                    <b>Processed query terms:</b> {' '.join([f'<span class="highlight">{term}</span>' for term in processed_terms])}
                </div>
            """, unsafe_allow_html=True)
            
            query_vector = query_to_vector(processed_terms, term_index_map, inverted_index, num_docs)
            results = rank_documents(query_vector, vector_space_index, alpha)

            if not results:
                st.warning(f"No documents found matching your query with α={alpha:.3f}. Try adjusting the similarity threshold or using different terms.")
            else:
                st.success(f"Found {len(results)} matching documents (sorted by relevance, α={alpha:.3f}):")
                
                for rank, (doc_id, score) in enumerate(results, 1):
                    display_doc_id = doc_id + 1  # Adjust for 1-indexed document filenames

                    with st.container():
                        st.markdown(f"""
                            <div class="result-card">
                                <h3>#{rank}: Document {display_doc_id} <span class="score">(Score: {score:.4f})</span></h3>
                            </div>
                        """, unsafe_allow_html=True)

                        content = read_document_content(doc_id)
                        if "not found" not in content:
                            with st.expander("View Document Content", expanded=False):
                                st.text_area(f"Document {display_doc_id} Content:", value=content, height=200, 
                                            key=f"content_{doc_id}", label_visibility="collapsed")
                        else:
                            st.warning(f"Document {display_doc_id} not found in the Abstracts directory.")

# Evaluation section
with st.expander("Golden Queries Evaluation", expanded=False):
    st.markdown("""
        <div class="eval-section">
            <h3>Evaluate with Golden Queries</h3>
            <p>Select a predefined query to evaluate the search engine's performance against known relevant documents.</p>
        </div>
    """, unsafe_allow_html=True)
    
    selected_query = st.selectbox("Select a golden query:", list(GOLDEN_QUERIES.keys()))
    
    if st.button(f"Evaluate '{selected_query}'"):
        alpha = st.session_state.get('alpha', 0.001)
        with st.spinner(f'Evaluating query: "{selected_query}"...'):
            processed_terms = preprocess_query(selected_query)
            query_vector = query_to_vector(processed_terms, term_index_map, inverted_index, num_docs)
            results = rank_documents(query_vector, vector_space_index, alpha)
            
            # Get relevant doc IDs (adjusting for 0-indexing)
            relevant_docs = [doc_id - 1 for doc_id in GOLDEN_QUERIES[selected_query]]
            
            # Calculate metrics
            retrieved_docs = [doc_id for doc_id, _ in results]
            relevant_retrieved = set(retrieved_docs) & set(relevant_docs)
            
            precision = len(relevant_retrieved) / len(retrieved_docs) if len(retrieved_docs) > 0 else 0
            recall = len(relevant_retrieved) / len(relevant_docs) if len(relevant_docs) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            st.markdown(f"""
                <div style="margin-bottom: 20px;">
                    <h4>Evaluation Results for: <span class="highlight">{selected_query}</span></h4>
                    <p><b>Relevant documents:</b> {len(relevant_docs)}</p>
                    <p><b>Retrieved documents:</b> {len(retrieved_docs)}</p>
                    <p><b>Relevant retrieved:</b> {len(relevant_retrieved)}</p>
                    <p><b>Precision:</b> {precision:.2%}</p>
                    <p><b>Recall:</b> {recall:.2%}</p>
                    <p><b>F1 Score:</b> {f1_score:.2%}</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Show retrieved documents with relevance indicators
            st.markdown("<h4>Retrieved Documents:</h4>", unsafe_allow_html=True)
            for rank, (doc_id, score) in enumerate(results, 1):
                display_doc_id = doc_id + 1
                is_relevant = doc_id in relevant_docs
                
                relevance_class = "relevant" if is_relevant else "not-relevant"
                relevance_icon = "✅" if is_relevant else "❌"
                relevance_label = "Relevant" if is_relevant else "Not relevant"
                
                st.markdown(f"""
                     <div class="retrieved-doc {relevance_class}">
                         <b>#{rank}: Document {display_doc_id}</b>
                         (<span class="score">Score: {score:.4f}</span>) – {relevance_icon} {relevance_label}
                    </div>
               """, unsafe_allow_html=True)    
# Footer
st.markdown("""
    <div class="footer">
        <p>Developed by Muneeb Ur Rehman | Information Retrieval Assignment #2</p>
    </div>
""", unsafe_allow_html=True)