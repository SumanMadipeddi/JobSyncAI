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
        You are an expert AI career assistant generating portfolio-ready project ideas tailored to a specific job description.

        Based on the following skills: {', '.join(skills)}, generate **2 realistic, advanced, and industry-relevant project ideas** that are:

        - Closely aligned with the job responsibilities typically associated with these skills
        - Feasible to build using open-source tools, public datasets, or widely available APIs
        - Valuable to real-world companies, with clear industry relevance
        Project 1: <Project Title>
        Industry Case:
        <1–2 lines explaining why this project is valuable or in demand in the industry>

        Project Description:
        <2–3 lines on what the project does, how it solves a real problem, and what technologies or approaches are used (mention tools only if necessary)>

        Project 2: <Project Title>
        Industry Case:
        <...>

        Project Description:
        <...>

        Only return 2 project. Do NOT use bullet points. Do not explain anything before or after. This should look like it belongs in a professional portfolio.
        """
        model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=api_key, temperature=0.2)
        response = model.invoke(prompt)
        return response.content if response else "Sorry, I couldn't generate a response."
    