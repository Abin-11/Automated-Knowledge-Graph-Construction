# Automated Knowledge Graph Construction

This project is an **Automated Knowledge Graph Construction** application that extracts entities and relationships from documents (PDFs, images) and visualizes them in a knowledge graph. It also uses clustering to infer the schema of the data and allows for schema refinement.

## Features

- **Document Parsing:** Extracts text from PDFs and images (using EasyOCR).
- **Entity & Relationship Extraction:** Identifies named entities and relationships using SpaCy NLP models.
- **Schema Inference:** Uses clustering (KMeans) to dynamically infer the schema of extracted entities.
- **Neo4j Integration:** Stores the knowledge graph in a Neo4j database.
- **Graph Visualization:** Displays the knowledge graph as a visual graph using NetworkX and Matplotlib.
- **Schema Refinement:** Allows users to manually refine the entities and relationships.

## Requirements

Before running the app, you need to install the required dependencies. You can do this by running the following command:

```bash
pip install spacy networkx neo4j pdfplumber nltk gensim py2neo openai tiktoken pandas matplotlib pytesseract easyocr

python -m spacy download en_core_web_sm
