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

# Header UI
st.markdown('<div class="header">Vector Space Model Search Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Information Retrieval Assignment #2</div>', unsafe_allow_html=True)

# Sidebar UI
with st.sidebar:
    st.markdown("### About")
    st.markdown("This search engine uses the Vector Space Model with TF-IDF weighting and cosine similarity for document ranking.")
    st.markdown("---")
    alpha = st.slider("Similarity threshold (α)", 0.0, 1.0, 0.01, 0.001, 
                      help="Adjust to filter out documents with low similarity scores")
    st.markdown("---")
    st.markdown("### Developer Information")
    st.markdown("*Name:* Muneeb Ur Rehman")
    st.markdown("*Assignment:* IR Assignment #2")
    st.markdown("*Model:* Vector Space Model")

# Load data
term_index_map, vector_space_index, inverted_index = load_all_data()
num_docs = len(vector_space_index)

# Query input with Enter key support
with st.form(key="search_form"):
    query = st.text_input("Enter your search query:", placeholder="Type your query here...")
    submit = st.form_submit_button("Search")

# Search logic
if submit and query:
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
                st.warning("No documents found matching your query. Try adjusting the similarity threshold or using different terms.")
            else:
                st.success(f"Found {len(results)} matching documents (sorted by relevance):")
                
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

# Footer
st.markdown("""
    <div class="footer">
        <p>Developed by Muneeb Ur Rehman | Information Retrieval Assignment #2</p>
    </div>
""", unsafe_allow_html=True)
