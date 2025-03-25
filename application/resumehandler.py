from langchain_community.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv

load_dotenv()

class ResumeHandler:
    def __init__(self, api_key, pdf_paths=[]):
        self.api_key = api_key
        self.pdf_paths = pdf_paths
        self.gemini_embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004", google_api_key=self.api_key
        )
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.vectordb = None

    def embed_text(self, text):
        return self.model.encode(text).reshape(1, -1)

    def extract_text_from_pdf(self, pdf_path):
        loader = PyPDFLoader(pdf_path)
        pages = loader.load_and_split()
        return " ".join([p.page_content for p in pages])

    def load_resumes(self, jd):
        from chain import Chain
        if not self.pdf_paths:
            return "No resumes uploaded yet.", None, {}

        job_text = f"""
        Role: {jd['role']}
        Experience: {jd['experience']}
        Skills: {', '.join(jd['skills'])}
        Description: {jd['description']}
        """
        jd_embedding = self.embed_text(job_text)

        best_score = 0
        best_resume = ""
        best_content = ""
        resume_scores = {}

        for pdf in self.pdf_paths:
            full_text = self.extract_text_from_pdf(pdf)
            embedding = self.embed_text(full_text)

            semantic_score = cosine_similarity(embedding, jd_embedding)[0][0]
            matches = sum(1 for skill in jd['skills'] if skill.lower() in full_text.lower())
            keyword_score = matches / len(jd['skills']) if jd['skills'] else 0
            final_score = (semantic_score * 0.6) + (keyword_score * 0.4)

            resume_scores[os.path.basename(pdf)] = {
                "Semantic Score": round(float(semantic_score), 3),
                "Keyword Match": round(float(keyword_score), 3),
                "ATS Score": round(float(final_score), 3)
            }

            if final_score > best_score:
                best_score = final_score
                best_resume = pdf
                best_content = full_text

        self.vectordb = FAISS.from_texts([best_content], embedding=self.gemini_embeddings)
        # self.vectordb.save_local("faiss_index")

        # if os.path.exists("faiss_index"):
        #     self.vectordb = FAISS.load_local("faiss_index", self.gemini_embeddings, allow_dangerous_deserialization=True)

        query = f"Role: {jd['role']}, Description: {jd['description']}"
        chain = RetrievalQA.from_chain_type(llm=Chain().llm, retriever=self.vectordb.as_retriever())
        result = chain({"query": query}, return_only_outputs=True)
        resume = result["result"]
        return resume, best_resume, resume_scores

    def suggest_skills(self, jd_skills, resume_path):
        resume_text = self.extract_text_from_pdf(resume_path).lower()
        missing_skills = [skill for skill in jd_skills if skill.lower() not in resume_text]
        return missing_skills
