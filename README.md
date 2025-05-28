# Exhibition Agent
Museum Artifact Knowledge Graph Construction Project README

## Main Features
- Data Acquisition and Processing: Utilized APIs from the Victoria and Albert Museum website to batch crawl data, followed by cleaning JSON-structured data and converting it into CSV format and DataFrame.
  
- Data Relational Transformation and Visualization: Starting with the artifact's systemNumber as the initial node, storing artifact data in a node-edge-node structure, and using the Neo4j library for data visualization.
  
- Cross-modal Retrieval: Employed the CLIP model to align images and text within the same feature space, enabling cross-modal retrieval based on both images and text.
  
- Question Answering System: Adopted the BERT encoder to capture semantic information within sentences, supporting precise information retrieval and Q&A services.
  
- Recommendation System: Integrated the BGE model to generate high-quality text embeddings, facilitating real-time recommendations of relevant artifact information based on user behavior data.

## Technology

- Data Preprocessing: Python was used for data cleaning and conversion into suitable structures for analysis.
- 
- Database Management: Neo4j was utilized for storing and visualizing relational graphs of artifacts and related information.
- 
- Cross-modal Retrieval: The CLIP model was applied for similarity calculations between images and text.
- 
- Question Answering System: Based on the BERT model, this system supports accurate querying of artifact-related information.
- 
- Recommendation System: The BGE model was employed for predicting user interests and recommending relevant content.
