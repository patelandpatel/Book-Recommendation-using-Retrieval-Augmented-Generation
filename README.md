# Book-Recommendation-using-Retrieval-Augmented-Generation
The project develops a chatbot using Retrieval Augmented Generation with a vector database to provide personalized book recommendations. It understands text similarity and generates relevant suggestions based on a user’s liked books, enhancing recommendation accuracy and user experience
### **Personalized Book Recommendation Chatbot using RAG**

## **📌 Overview**
This project develops a **chatbot for personalized book recommendations** using **Retrieval Augmented Generation (RAG)** and a **vector database**. The system efficiently retrieves relevant book data and generates high-quality recommendations by leveraging **embedding models** and **LLMs (Large Language Models)**.

## **🚀 Features**
- **Retrieval-Augmented Generation (RAG):** Combines retrieval-based and generative models for better accuracy.
- **Vector Database:** Uses **FAISS** for fast and scalable similarity searches.
- **Embedding Models:** Leverages **BERT-based embeddings** for contextual text understanding.
- **LLM-powered Recommendations:** Generates book suggestions based on user preferences.
- **Efficient Data Processing:** Indexes and retrieves book descriptions and genres from a **10K book dataset**.
- **Customizable:** Can be extended to support more genres, authors, or datasets.

---

## **📂 Dataset**
The dataset is sourced from **Kaggle** and consists of:
- **10,000+** books
- **Attributes:** Book title, author, description, genres, ratings, and Goodreads URL
- **Filtered Features:** Book name, genres, and description


---

## **🛠️ Tech Stack**
- **Programming Language:** Python
- **Frameworks & Libraries:**
  - **LangChain** (for LLM orchestration)
  - **Hugging Face Transformers** (for embeddings & generation)
  - **FAISS** (for vector search)
  - **ChromaDB** (alternative vector storage)
  - **NumPy, Pandas** (data processing)
- **LLM Provider:** Hugging Face (`google/flan-t5-base`)
- **Deployment:** Google Colab with T4 GPU

---

## **📖 Methodology**
### **1️⃣ Data Preprocessing**
- Loaded data using **CSVLoader (LangChain)**
- Split text into smaller chunks (**CharacterTextSplitter**) for efficient indexing
- Converted text into **embeddings** using `bert-base-uncased`

### **2️⃣ Vector Storage & Retrieval**
- Indexed embeddings in **FAISS** for fast similarity search
- Used **cosine similarity** to compare books based on description and genre
- Implemented **retriever models** to fetch the most relevant books

### **3️⃣ LLM-based Generation**
- Integrated **Hugging Face LLMs** (`google/flan-t5-base`) for contextual book recommendations
- Tuned model **temperature** for balanced creativity and accuracy
- Constructed **prompt engineering** for structured and complete responses

### **4️⃣ Evaluation & Experiments**
- Compared multiple **LLMs (GPT, Falcon, Pythia, etc.)**
- Tested **different embedding models (BERT, Sentence-BERT, USE)**
- Evaluated **retrieval accuracy** using **cosine similarity**

---

## **📌 Setup & Installation**
### **🔹 Prerequisites**
- Python **3.x**
- Google Colab (recommended) or local Python environment

### **🔹 Install Dependencies**
```bash
pip install chromadb langchain tiktoken sentence-transformers faiss-cpu
```

### **🔹 Clone the Repository**
```bash
git clone https://github.com/your-username/book-recommendation-rag.git
cd book-recommendation-rag
```

### **🔹 Run the Project**
1. Load the dataset:
   ```python
   from langchain.document_loaders.csv_loader import CSVLoader
   loader = CSVLoader(file_path="book_genre.csv")
   data = loader.load()
   ```

2. Generate embeddings:
   ```python
   from langchain.embeddings import HuggingFaceEmbeddings
   embeddings = HuggingFaceEmbeddings(model_name='bert-base-uncased')
   ```

3. Create FAISS vector store:
   ```python
   from langchain.vectorstores import FAISS
   docsearch = FAISS.from_documents(texts, embeddings)
   ```

4. Query the chatbot:
   ```python
   from langchain.chains import RetrievalQA
   qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever())
   print(qa.run({"query": "Recommend books similar to 'To Kill a Mockingbird'"}))
   ```

---

## **📊 Results & Analysis**
- Achieved **higher relevance** in recommendations by incorporating **retrieved knowledge** into LLM responses.
- **Compared different vector databases (FAISS vs. Chroma)** and found FAISS to be **faster**.
- **Experimented with various LLMs**, but **google/flan-t5-base** provided the most coherent responses.

---

## **🔧 Future Enhancements**
- **Expand dataset** with user reviews and detailed metadata
- **Improve embedding models** with fine-tuned transformers
- **Implement multi-turn conversations** for better recommendations
- **Deploy as a web application** using Flask or FastAPI

---

## **📜 License**
This project is licensed under the **MIT License**

---

## **💡 Acknowledgments**
- **Hugging Face & LangChain** for powerful NLP tools
- **FAISS & ChromaDB** for efficient similarity search
- **Kaggle** for the dataset


