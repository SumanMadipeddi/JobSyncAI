import streamlit as st
from resumehandler import ResumeHandler
from langchain_community.document_loaders import WebBaseLoader
from chain import Chain
from google import genai
from skills import skills
import os
from datetime import datetime
import time

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

st.set_page_config(page_title="AI Chatbot", layout="wide")
load_dotenv()


api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

chain = Chain()   
resume_handler = ResumeHandler(api_key, pdf_paths=[])


col1, col2 = st.columns([1.5, 1])

with col1:
    st.title("Job Resume Generator and Email Writer")
    st.write("Upload your resume PDF and provide a job description link.")

    if not os.path.exists("resumes"):
        os.makedirs("resumes")

    uploaded_files = st.file_uploader("\U0001F4C2 Upload Resumes", accept_multiple_files=True, type=['pdf'])

    if uploaded_files:
        resume_paths = [f"resumes/{file.name}" for file in uploaded_files]
        for file in uploaded_files:
            with open(f"resumes/{file.name}", "wb") as f:
                f.write(file.read())
        resume_handler.pdf_paths = resume_paths

    job_desc_url = st.text_input("\U0001F517 Job Description URL")

    def extract_job_description_from_link(url):
        try:
            loader = WebBaseLoader(url)
            data = loader.load().pop().page_content
            return data
        except Exception as e:
            st.error(f"Error extracting job description: {str(e)}")
            return ""

    if st.button("\U0001F680 Process Resume and Job Description"):
        if resume_handler.pdf_paths and job_desc_url:
            job_desc = extract_job_description_from_link(job_desc_url)
            if job_desc:
                job_details = chain.extract_jobs(job_desc)
                skills_obj = skills()
                jd_skills = job_details.get('skills', [])
                ext_skills = skills_obj.query_skills(jd_skills)
                projects = skills_obj.query_projects(jd_skills)

                st.markdown("📌 **Extracted Job Details:**")
                st.markdown(f"""
                - **Company:** {job_details.get('company', 'N/A')}
                - **Role:** {job_details.get('role', 'N/A')}
                
                **Experience Required:** 
                {job_details.get('experience', 'N/A')}

                **Skills Required:**  
                {', '.join(job_details.get('skills', []))}


                **Job Description:**  
                {job_details.get('description', 'N/A')}
                """)

                resume, best_resume_path, resume_scores = resume_handler.load_resumes(job_details)

                st.write("📄 **ATS Resume Matching Scores:**")
                for name, score_info in resume_scores.items():
                    st.markdown(f"""
                    **{name}**
                    - Semantic Match: `{score_info['Semantic Score']}`
                    - Keyword Match: `{score_info['Keyword Match']}`
                    - 🏆 ATS Score: `{score_info['ATS Score']}`
                    """)

                st.success(f"✅ Best Resume: `{os.path.basename(best_resume_path)}`")
                # Suggest missing skills for the best resume
                suggested_skills = resume_handler.suggest_skills(job_details['skills'], best_resume_path)

                if suggested_skills:
                    inline_skills = ", ".join(suggested_skills)
                    st.markdown(f"📌 **Suggested Skills to Add:** {inline_skills}")
                else:
                    st.success("✅ Your resume includes all the key kills")

                suggested_projects = skills_obj.generate_project_ideas(jd_skills)
                st.write("🧠 Suggested Project Ideas:")
                st.write(suggested_projects)

                email = chain.write_mail(job_desc, resume, job_details['role'], job_details['company'], ext_skills, projects)
                st.write("\U0001F4E7 Generated Cold Email:")
                st.text(email)
        else:
            st.error("Please upload your resume PDFs and provide a job description URL.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def render_message(message, role, timestamp):
    is_user = role == "user"
    alignment = "flex-end" if is_user else "flex-start"

    st.markdown(
        f"""
        <div style="display: flex; justify-content: {alignment}; margin-bottom: 10px;">
                <div style="font-weight: bold; font-size: 14px;"> {role.capitalize()}</div>
                <div>{message}</div>
                <div style="font-size: 10px; color: #888; text-align: right;">{timestamp}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    import streamlit.components.v1 as components

    st.title("🤖 AI Chatbot")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Step 1: Process input (but DON'T render anything yet)
    user_input = st.chat_input("Type your message here...")  # stays at bottom automatically

    if user_input:
        current_time = datetime.now().strftime("%I:%M %p")
        st.session_state.chat_history.append(("user", (user_input, current_time)))

        with st.spinner("Bot is typing..."):
            time.sleep(1)

            history_text = "\n".join([
                f"{role.capitalize()}: {msg}" for role, (msg, _) in st.session_state.chat_history
            ])

            chat_prompt = f"""
            You are a highly intelligent, friendly, and context-aware AI assistant. You engage users with a natural and adaptive tone, tailoring your responses to their intent and prior messages.
            Use the latest 5 conversation history and the user's latest input to generate helpful, relevant, and thoughtful responses.
            ---
            **User's Latest Input**:
            {user_input}
            **Conversation History**:
            {history_text}
            ---
            **Instructions**:
            - Maintain consistency and context throughout the conversation.
            - If the user references past details (e.g., their name), recall them from history.
            - For casual chat, be friendly, natural, and ask engaging follow-up questions.
            - For resumes, job applications, and professional emails:
            - Refine text for clarity, persuasion, and impact.
            - Offer structured, insightful advice.
            - Respond concisely, yet helpfully.
            - Your tone should be warm, human-like, and intelligent.

            **Important**: Do **not** describe yourself or include preambles. Only respond as the assistant.
            """

            model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=api_key, temperature=0.2)
            response = model.invoke(chat_prompt)
            bot_reply = response.content if response else "Sorry, I couldn't generate a response."

        bot_time = datetime.now().strftime("%I:%M %p")
        st.session_state.chat_history.append(("bot", (bot_reply, bot_time)))

    # Step 2: Render the chat history *after* processing new input
    chat_html = """
        <style>
            .chat-container {
                height: 420px;
                overflow-y: auto;
                display: flex;
                flex-direction: column-reverse;
                border: 1px solid #ccc;
                padding: 10px;
                background-color: #f7f7f7;
                border-radius: 8px;
                margin-bottom: 10px;
            }
            .chat-message {
                margin-bottom: 10px;
                font-size: 14px;
                padding: 8px 12px;
                border-radius: 10px;
                max-width: 80%;
                word-wrap: break-word;
            }
            .chat-user {
                align-self: flex-end;
                background-color: #d1e7dd;
                color: #0f5132;
            }
            .chat-bot {
                align-self: flex-start;
                background-color: #e7f1ff;
                color: #084298;
            }
            .timestamp {
                font-size: 10px;
                color: #888;
                margin-top: 4px;
                text-align: right;
            }
        </style>

        <div class="chat-container" id="chat-box">
    """
    for role, (msg, timestamp) in reversed(st.session_state.chat_history):
        chat_html += f"""
            <div class="chat-message chat-{role}">
                {msg}
                <div class="timestamp">{timestamp}</div>
            </div>
        """
    chat_html += "</div>"
    # Now display chat messages *above* the input field
    components.html(chat_html, height=500)
