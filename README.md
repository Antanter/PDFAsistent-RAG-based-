# PDFAsistent-RAG-based-

PDF text manager. Uses RAG technology: GPT creates embeddings from the PDF doc, the embeddings are saved to ChromaDB, and then GPT also used to answer the questions additionaly based on the information in ChromaDB.
To use:

1. Open console folder, run ``` py main.py ```
2. Then, either run ```exit``` or ```quit``` to exit the application,
3. Or ```add_doc```, and then name of the PDF file,
4. Or anything other, to ask generative model to answer
