import pandas as pd
import chromadb
import uuid
import os
from dotenv import load_dotenv
from google import genai
from langchain_google_genai import ChatGoogleGenerativeAI
load_dotenv()
api_key=os.getenv("GEMINI_API_KEY")

client=genai.Client(api_key=api_key)

class skills:
    def __init__(self, file_path= "C:\\Users\\madip\\Downloads\\cold_email\\application\\suman_projects.csv"):
        self.file_path = file_path
        self.data = pd.read_csv(file_path)
        self.chroma_client = chromadb.PersistentClient('vectorstore')
        self.collection = self.chroma_client.get_or_create_collection(name="skills")

    def load_portfolio(self):
        if not collection.count(): # type: ignore
            for _, row in df.iterrows(): # type: ignore
                metadata = {"Projects": row['Projects']}
                Collection.add(documents=[row['Techstack']], # type: ignore
                        metadatas=[metadata],
                        ids=[str(uuid.uuid4())])

    def query_skills(self, skills):
        results = self.collection.query(query_texts=skills, n_results=2)
        skills = results.get('documents', [])
        return skills

    def query_projects(self, skills):
        results = self.collection.query(query_texts=skills, n_results=2)
        projects = results.get('metadatas', [])
        return projects
    
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
        response=model.invoke(prompt)
        return response.content if response else "Sorry, I couldn't generate a response."
