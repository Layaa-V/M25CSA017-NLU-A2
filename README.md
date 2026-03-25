This repository contains the implementations and analysis for two Natural Language Understanding (NLU) Assignment-2. 

## Description of the files present in this repository - 
* **NLU_A2_Q1.ipynb** - Contains the solution code for problem 1
* **NLU_A2_Q2.ipynb** - Contains the solution code for problem 2
* **NLU_A2_report.pdf** - Contains detailed analysis about both problems 1 and 2
* **data_pdfs** - PDFs used in Problem 1 for data extraction
* **corpus.txt** - Cleaned corpus ontained in Problem 1
* **TrainingNames.txt** - LLM generated txt file of 1000 Indian Names

---

## Problem 1: Learning Word Embeddings
**Objective:** To train and compare Continuous Bag of Words (CBOW) and Skip-gram Word2Vec models on a custom text corpus scraped from the official IIT Jodhpur website and its academic regulation PDFs.

### Key Implementations:
* **Data Extraction:** We use web scraping (BeautifulSoup) and PDF extraction (PyPDF2) to build a localized academic corpus
* **Text Preprocessing:** Tokenization, stopword removal, lowercasing and filtering
* **PyTorch Word2Vec:** Implemented custom PyTorch architectures for both CBOW and Skip-gram
* **Semantic Analysis:** Calculated Cosine Similarity for nearest-neighbor lookups and word analogies
* **Visualization:** Using t-SNE (for both CBOW and Skip-gram)

---

## Problem 2: Character-Level Name Generation
**Objective:** To design, train, and evaluate sequence models for autoregressive character-level generation of Indian names.

### Key Implementations:
* **Dataset:** Generated a custom training corpus of 1,000 unique Indian names
* **Architectures Evaluated:**
  1. **Vanilla RNN:** Standard unidirectional recurrent network
  2. **Bidirectional LSTM (BLSTM):** Processes sequences in both forward and backward directions
  3. **RNN + Attention:** Unidirectional RNN utilizing a self-attention mechanism to compute weighted context vectors over past states
* **Evaluation Metrics:** Calculated Novelty Rate (percentage of generated names not in the training set) and Diversity Rate (percentage of unique names in the generated batch).

---

## How to Run - 
* The google colab link to both code files NLU_A2_Q1.ipynb and NLU_A2_Q2.ipynb are provided with the code. It can be directly run from there, end to end.
    1. **Prerequisites for running Problem 1 code** - The folder 'data_pdfs' should be present in same runtime repository(files section)
    2. **Prerequisites for running Problem 2 code** - The file TrainingNames.txt should be present in same runtime repository(files section)
* The files can also be converted to python scripts(.py) format and be run on terminal by downloading the following dependencies -
    1. **For Problem 1** - pip install requests beautifulsoup4 PyPDF2 nltk wordcloud matplotlib scikit-learn
    2. **For Problem 2** - pip install torch numpy
    
