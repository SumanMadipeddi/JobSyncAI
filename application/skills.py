import pandas as pd
import os
from dotenv import load_dotenv
from google import genai
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)
embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)

class skills:
    def __init__(self, file_path="suman_projects.csv"):
        self.file_path = file_path
        self.data = pd.read_csv(file_path)

        self.documents = self.data["Skills Used"].astype(str).tolist()
        self.metadatas = self.data[["Project Name"]].rename(columns={"Project Name": "Projects"}).to_dict(orient="records")


        # Build the FAISS vector index
        self.vectordb = FAISS.from_texts(self.documents, embedding_model, metadatas=self.metadatas)

    def query_skills(self, skill_list):
        results = self.vectordb.similarity_search_with_score(" ".join(skill_list), k=2)
        return [res[0].page_content for res in results]

    def query_projects(self, skill_list):
        results = self.vectordb.similarity_search_with_score(" ".join(skill_list), k=2)
        return [res[0].metadata for res in results]

    def generate_project_ideas(self, skills):
        prompt = f"""
        Generate a possible, advanced, and industry-relevant project idea based on these skills: {', '.join(skills)}.
        The project should follow this format:
        Project 1: <Project Title>
        Industry Case: <Why this project is important and where it fits in real-world company use cases in a few lines>
        Project Description: <A high-level summary of how this project works and what it solves. Focus on realism and industry alignment. Mention technologies only if they make sense for the problem. give me in few lines only>

        Only return 2 project. Do NOT use bullet points. Do not explain anything before or after. This should look like it belongs in a professional portfolio.
        """
        model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=api_key, temperature=0.2)
        response = model.invoke(prompt)
        return response.content if response else "Sorry, I couldn't generate a response."
