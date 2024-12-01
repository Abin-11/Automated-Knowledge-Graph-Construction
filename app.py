#!/usr/bin/env python
# coding: utf-8

# Install dependencies (Uncomment if needed)
# pip install spacy networkx neo4j pdfplumber nltk gensim py2neo openai tiktoken pandas matplotlib pytesseract easyocr

import pdfplumber
import spacy
from spacy.matcher import Matcher
from py2neo import Graph, Node, Relationship
import streamlit as st
import requests
from bs4 import BeautifulSoup
import easyocr
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from collections import defaultdict
import os

# Streamlit app title
st.title("Automated Knowledge Graph Construction")

# Initialize Neo4j connection
graph = Graph("bolt://28f4-202-38-180-199.ngrok-free.app:7687", auth=("neo4j", "DBMSDBMS"))

# Function to parse PDF
def parse_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Function to parse web pages
def parse_web_page(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    return soup.get_text()

# Parse image text using EasyOCR
def parse_image_with_easyocr(image_path):
    reader = easyocr.Reader(['en'])
    result = reader.readtext(image_path)
    text = ' '.join([item[1] for item in result])
    return text

# Extract entities and relationships
nlp = spacy.load("en_core_web_sm")
def extract_entities_and_relationships(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]

    relationships = []
    for token in doc:
        if token.dep_ == "ROOT":
            subject = [child for child in token.children if child.dep_ == "nsubj"]
            object_ = [child for child in token.children if child.dep_ == "dobj"]
            if subject and object_:
                relationships.append((subject[0].text, token.text, object_[0].text))

    return entities, relationships

# Dynamic schema inference using clustering
def infer_schema(entities):
    entity_texts = [e[0] for e in entities]
    vectorizer = spacy.load("en_core_web_sm")
    vectors = [vectorizer(e).vector for e in entity_texts]
    
    kmeans = KMeans(n_clusters=3, random_state=42).fit(vectors)
    clustered_entities = defaultdict(list)
    for idx, label in enumerate(kmeans.labels_):
        clustered_entities[label].append(entity_texts[idx])
    
    inferred_schema = {f"Type_{i}": terms for i, terms in clustered_entities.items()}
    return inferred_schema

# Populate knowledge graph
def populate_knowledge_graph(entities, relationships):
    for entity, label in entities:
        node = Node(label, name=entity)
        graph.merge(node, label, "name")
    for subj, rel, obj in relationships:
        subj_node = graph.nodes.match(name=subj).first()
        obj_node = graph.nodes.match(name=obj).first()
        if subj_node and obj_node:
            graph.merge(Relationship(subj_node, rel, obj_node))

# Visualize the graph
def visualize_graph(entities, relationships):
    G = nx.DiGraph()
    for entity, _ in entities:
        G.add_node(entity)
    for subj, rel, obj in relationships:
        G.add_edge(subj, obj, label=rel)

    pos = nx.spring_layout(G)
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="gray", font_weight="bold")
    edge_labels = nx.get_edge_attributes(G, "label")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.savefig("graph_plot.png")
    plt.close()

# File uploader for various formats
uploaded_file = st.file_uploader("Upload a PDF or image file:", type=["pdf", "png", "jpg"])
if uploaded_file:
    if uploaded_file.name.endswith(".pdf"):
        document_text = parse_pdf(uploaded_file)
    elif uploaded_file.name.endswith((".png", ".jpg")):
        document_text = parse_image_with_easyocr(uploaded_file)

    st.write("Preview of Extracted Text:")
    st.write(document_text[:500])  # Preview first 500 characters

    # Entity and relationship extraction
    entities, relationships = extract_entities_and_relationships(document_text)
    st.write("Extracted Entities:", entities)
    st.write("Extracted Relationships:", relationships)

    # Schema inference
    schema = infer_schema(entities)
    st.write("Inferred Schema:", schema)

    # Populate knowledge graph
    populate_knowledge_graph(entities, relationships)

    # Schema refinement prompts
    st.subheader("Refine Schema")
    new_entity = st.text_input("Add New Entity:")
    new_relationship = st.text_input("Add New Relationship (Format: subject, verb, object):")
    if st.button("Submit Refinements"):
        if new_entity:
            entities.append((new_entity, "CUSTOM"))
        if new_relationship:
            try:
                subj, rel, obj = new_relationship.split(", ")
                relationships.append((subj, rel, obj))
            except ValueError:
                st.error("Invalid relationship format. Use: subject, verb, object")
        st.write("Updated Entities:", entities)
        st.write("Updated Relationships:", relationships)

    # Visualize the graph
    if entities and relationships:
        visualize_graph(entities, relationships)
        if os.path.exists("graph_plot.png"):
            st.image("graph_plot.png", caption="Knowledge Graph")
    else:
        st.warning("No entities or relationships to visualize. Please check the input data.")
