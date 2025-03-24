import os
from  google import genai
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv
from resumehandler import ResumeHandler
from skills import skills
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
        The scraped text is from the career's page of a website.
        Your job is to extract the job postings and return them in JSON format containing the 
        following keys: `company`,`role`, `experience`, `skills` and `description`.
        Only return the valid JSON.
        IN this remove the outer list
        ### VALID JSON (NO PREAMBLE):    
        """ )

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
        Please tailor the email to the specific job description {job_description}

        ### INSTRUCTION:

        You are Suman Madipeddi AI and LLMs engineer at Minor Chores and Robotics graduate student at ASU.
        
        Your job is to write a cold and skilled in crafting professional cold emails for job applications. Write a concise and engaging cold email to a recruiter for the role of {role} at {company}. The email should highlight my key skills, projects, and experiences that align with the job description, but dont put much which i dont have. eep the email personalized and professional while maintaining a warm and approachable tone. Ensure the subject line is attention-grabbing. Keep the body of the email within 150-200 words and include a compelling call-to-action for further discussion.


        Use my {resume} and add the most relevant project titles and {Projects} and some highly related {skills}. if any duplicates are there remove those.

        Remember you are name inside the resume AI and LLMs engineer at Minor Chores and Robotics graduate student at ASU, graduating in May 2025. The email should sound natural, enthusiastic, and action-driven, ensuring it creates a strong impression.

        Do not provide a preamble.
        ### EMAIL (NO PREAMBLE):
        """
        )

        chain_email = prompt_email | self.llm
        res_email = chain_email.invoke({"job_description":jd, "resume":resume, "role": role, "company": company, "skills": skills,"Projects":projects })

        return res_email.content


