import os
import fitz
from textwrap import wrap
from openai import OpenAI
import chromadb
from chromadb.config import Settings

MAX_N = 5

class PDFHelper:
    clientGen = OpenAI(api_key=os.getenv("API_KEY"))

    clientRet = chromadb.Client(Settings(anonymized_telemetry=False))
    collection = clientRet.get_or_create_collection(name="pdf_chunks")

    def read_pdf_stream(self, path):
        doc = fitz.open(path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            yield page_num, text

    def chunk_pdf(self, path, chunk_size=1000, overlap=200):
        for _, page_text in self.read_pdf_stream(path):
            chunks = wrap(page_text, chunk_size - overlap)
            for i in range(len(chunks)):
                start = max(0, i - 1)
                chunk = " ".join(chunks[start:i + 1])
                yield chunk

    def get_embeddings(self, text, model="text-embedding-3-small"):
        response = self.clientGen.embeddings.create(
            model=model,
            input=[text],
        )
        return response.data[0].embedding

    def create_collection(self, filename):
        chunks = list(self.chunk_pdf(filename))
        embeddings = self.get_embeddings(chunks)

        self.collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=[f"chunk-{i}" for i in range(len(chunks))]
        )

    def search_for_query(self, query):
        results = self.collection.query(
            query_texts=[query],
            n_results=MAX_N
        )

        return results["documents"][0]
    
    def search_for_question(self, question, model="gpt-4o-mini"):
        context = "\n\n".join(self.search_for_query(question))

        prompt = f"""
        You're a PDF assistant. The question to you is: {question}

        Answer the question, using the following context:
        {context}

        Answer:
        """

        response = self.clientGen.responses.create(
            model=model,
            input=prompt,
            store=True,
        )

        print(response.output_text)

    def activate_loop_mode(self):
        question = ""
        while True:
            question = input("Ask something: ")
            if question == "exit" or question == "quit":
                break;

            elif question == "add_doc":
                filename = input("Type the filename: ")
                try:
                    self.create_collection(filename)
                except FileNotFoundError:
                    print(f"File '{filename}' not found. Pass.")

            else:
                self.search_for_question(question)

if __name__ == "__main__":
    PDFhelper = PDFHelper()
    PDFhelper.activate_loop_mode()