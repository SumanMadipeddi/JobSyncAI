from langchain_community.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from difflib import SequenceMatcher
import os
import re
import shutil
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

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

        # Clear in-memory vectorstore
        self.vectordb = None

        # Clean up any previously saved FAISS indexes if they exist
        if os.path.exists("faiss_index"):
            shutil.rmtree("faiss_index")

        if not self.pdf_paths:
            return "No resumes uploaded yet.", None, {}

        job_text = f"""
        Role: {jd.get('role', '')}
        Experience: {jd.get('experience', '')}
        Skills: {', '.join(jd.get('skills', []))}
        Description: {jd.get('description', '')}
        """
        jd_embedding = self.embed_text(job_text)

        best_score = 0
        best_resume = ""
        best_content = ""
        resume_scores = {}

        for pdf in self.pdf_paths:
            full_text = self.extract_text_from_pdf(pdf).lower()
            embedding = self.embed_text(full_text)
            semantic_score = cosine_similarity(embedding, jd_embedding)[0][0]

            matches = sum(1 for skill in jd.get('skills', []) if skill.lower() in full_text)
            keyword_score = matches / len(jd.get('skills', [])) if jd.get('skills') else 0

            role = jd.get('role', '').lower()
            role_terms = [role] + role.split()
            role_score = 1.0 if any(term in full_text for term in role_terms) else 0.0

            exp_match = re.findall(r'(\d+)\+?\s+years? of experience', full_text)
            resume_yoe = max([int(y) for y in exp_match], default=0)
            jd_exp_match = re.findall(r'(\d+)\+?\s+years?', jd.get('experience', '').lower())
            job_yoe = max([int(y) for y in jd_exp_match], default=0)
            exp_score = min(resume_yoe / job_yoe, 1.0) if job_yoe else 0.5

            has_project_section = "project" in full_text
            project_skill_match = any(skill.lower() in full_text for skill in jd.get('skills', []))
            project_relevance = 1.0 if has_project_section and project_skill_match else 0.0

            final_score = (
                semantic_score * 0.5 +
                keyword_score * 0.2 +
                role_score * 0.1 +
                exp_score * 0.1 +
                project_relevance * 0.1
            )

            resume_scores[os.path.basename(pdf)] = {
                "Semantic Score": round(float(semantic_score), 2),
                "Keyword Match": round(float(keyword_score), 2),
                "Role Alignment": round(role_score, 2),
                "Experience Fit": round(exp_score, 2),
                "Project Relevance": round(project_relevance, 2),
                "ATS Score": round(final_score, 2)
            }

            if final_score > best_score:
                best_score = final_score
                best_resume = pdf
                best_content = full_text

        self.vectordb = FAISS.from_texts([best_content], embedding=self.gemini_embeddings)

        query = f"Role: {jd.get('role', '')}, Description: {jd.get('description', '')}"
        chain = RetrievalQA.from_chain_type(llm=Chain().llm, retriever=self.vectordb.as_retriever())
        result = chain({"query": query}, return_only_outputs=True)
        resume = result.get("result", "")
        return resume, best_resume, resume_scores

    def suggest_skills(self, jd_skills, resume_path):
        resume_text = self.extract_text_from_pdf(resume_path).lower().split()

        def is_similar(skill, words):
            return any(SequenceMatcher(None, skill.lower(), word).ratio() > 0.85 for word in words)

        if isinstance(jd_skills, str):
            if "," in jd_skills:
                jd_skills = [s.strip() for s in jd_skills.split(",")]
            else:
                jd_skills = jd_skills.split()

        missing_skills = [skill for skill in jd_skills if skill and not is_similar(skill, resume_text)]
        return missing_skills
