import os
from  google import genai
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv
from utils import clean_text

load_dotenv()

class Chain:
    def __init__(self):
        self.llm=ChatGroq(model_name="llama3-8b-8192")

    def extract_jobs(self, cleaned_text):
        prompt = PromptTemplate.from_template(
        """
        ### SCRAPED TEXT FROM WEBSITE:
        {data}
        ### INSTRUCTION:
        Your job is to extract the job postings and return them in JSON format containing the 
        following keys: `company`, `role`, `experience`, `skills`, and `description`.

        - `skills`: This must be a **comma-separated list** of specific skills only (e.g., "Python, NLP, TensorFlow, Git, LLM fine-tuning").
        Do NOT include full sentences or project descriptions.
        Only return names of tools, libraries, models, or techniques (e.g., "RAG", "RLHF", "LangChain", "PEFT", etc.).

        Only return the valid JSON. Remove the outer list.""")

        chain_extract = prompt | self.llm
        extracted_JD = chain_extract.invoke({"data": cleaned_text})

        try:
            json_parser = JsonOutputParser()
            jd = json_parser.parse(extracted_JD.content)
        except OutputParserException:
            raise OutputParserException("Context too big. Unable to parse jobs.")
        return jd

    def write_mail(self, jd, resume, role, company, skills, projects):
        prompt_email = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION:
            Tailor the email to this job description: {job_description}

            ### INSTRUCTION:

            Write a short, compelling, and professional cold email to a recruiter for the role of {role} at {company}.
            
            Keep it:
            - Within 120–140 words
            - Personalized and warm
            - Focused only on relevant skills, projects, and experience
            You are name on the resume, recent role and company and the latest degree and university from my {resume} and use that to include the most relevant project titles and {Projects}, along with highly relevant {skills}. If any are unrelated to the role, skip them. Avoid repetition.

            Refer to my most recent role and education in the resume. You may also extract my graduation year, email, or phone if relevant. The email should sound enthusiastic and natural, with a strong and actionable close.

            Do NOT include any intro/preamble — only output the email.

            ### EMAIL (NO PREAMBLE):
            """
        )

        chain_email = prompt_email | self.llm
        res_email = chain_email.invoke({"job_description":jd, "resume":resume, "role": role, "company": company, "skills": skills,"Projects":projects })

        return res_email.content


