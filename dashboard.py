import streamlit as st
import pandas as pd
import plotly.express as px
import os
import json
from pathlib import Path
from datetime import datetime
from groq import Groq
from dotenv import load_dotenv
import urllib.parse
import requests
from typing import List, Dict, Optional

# Load environment variables
load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Constants
ASSESSMENTS_DIR = Path("assessments")
MAX_COURSES = 5
MAX_JOBS = 5

# Initialize directories
ASSESSMENTS_DIR.mkdir(exist_ok=True, parents=True)

class CareerAgent:
    def __init__(self, domain: str):
        self.domain = domain
        self.user_level = "beginner"  # Would be detected from assessment results
        
    # def fetch_course_links(self) -> List[tuple]:
    #     """Generate free course search links for the domain"""
    #     query = urllib.parse.quote_plus(f"free {self.domain} course")
    #     course_links = [
    #         ("Coursera", f"https://www.coursera.org/search?query={query}&price=1"),
    #         ("edX", f"https://www.edx.org/search?q={query}&price=Free"),
    #         ("Udemy", f"https://www.udemy.com/courses/search/?q={query}&price=price-free"),
    #         ("freeCodeCamp", f"https://www.google.com/search?q=site%3Afreecodecamp.org+{query}"),
    #         ("YouTube", f"https://www.youtube.com/results?search_query={query}")
    #     ]
    #     return course_links


    # def search_job_portals(self, location: str = "") -> List[tuple]:
    #     """Agentic job search with optimized queries and robust fallback"""
    #     titles_prompt = f"""
    #     Suggest 3 entry-level job titles in {self.domain} for someone with {self.user_level} skills.
    #     Return titles only, comma-separated.
    #     """

    #     try:
    #         titles_response = groq_client.chat.completions.create(
    #             model="llama3-70b-8192",
    #             messages=[{"role": "user", "content": titles_prompt}],
    #             temperature=0.5
    #         )

    #         raw_output = titles_response.choices[0].message.content.strip()
    #         titles = [t.strip() for t in raw_output.split(",") if t.strip()]
    #         if not titles:
    #             raise ValueError("Empty LLM response")
    #     except Exception as e:
    #         st.warning(f"⚠️ Failed to fetch job titles from LLM. Using fallback titles.\n{e}")
    #         titles = ["Junior Data Analyst", "Machine Learning Intern", "Business Intelligence Trainee"]

    #     # Build search links
    #     job_links = []
    #     base_platforms = {
    #         "LinkedIn": "https://www.linkedin.com/jobs/search/?keywords={query}&location={location}",
    #         "Indeed": "https://in.indeed.com/jobs?q={query}&l={location}",
    #         "Glassdoor": "https://www.glassdoor.com/Job/jobs.htm?sc.keyword={query}"
    #     }

    #     for title in titles:
    #         query = urllib.parse.quote_plus(title)
    #         loc = urllib.parse.quote_plus(location) if location else ""
    #         for platform, url_template in base_platforms.items():
    #             url = url_template.format(query=query, location=loc)
    #             job_links.append((f"{title} ({platform})", url))

    #     return job_links[:MAX_JOBS]
    def fetch_course_links(self) -> List[tuple]:
        """Generate free course search links for the domain"""
        query = urllib.parse.quote_plus(f"free {self.domain} course")
        course_links = [
            ("Coursera", f"https://www.coursera.org/search?query={query}&price=1"),
            ("edX", f"https://www.edx.org/search?q={query}&price=Free"),
            ("Udemy", f"https://www.udemy.com/courses/search/?q={query}&price=price-free"),
            ("freeCodeCamp", f"https://www.google.com/search?q=site%3Afreecodecamp.org+{query}"),
            ("YouTube", f"https://www.youtube.com/results?search_query={query}")
        ]
        return course_links

    def search_job_portals(self, location: str = "") -> List[tuple]:
        """Fetch job links using LLM-suggested job titles and build URLs"""
        titles_prompt = f"""
        Suggest 3 entry-level job titles in {self.domain} for someone with beginner skills.
        Just return the job titles as a comma-separated list. Do NOT include any explanation or extra text.
        """
        try:
            titles_response = groq_client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[{"role": "user", "content": titles_prompt}],
                temperature=0.3
            )
            # Clean and parse titles
            raw = titles_response.choices[0].message.content.strip()
            clean_text = raw.replace("\n", "").replace("•", "").replace("1.", "").replace("2.", "").replace("3.", "")
            titles = [t.strip() for t in clean_text.split(",") if len(t.strip()) > 2]

            if not titles:
                raise ValueError("No valid titles extracted.")
        except Exception as e:
            st.warning(f"⚠️ Couldn't get job titles from LLM. Using fallback. \n\nError: {e}")
            titles = ["Junior Data Analyst", "Business Analyst Intern", "Data Reporting Assistant"]

        job_links = []
        base_platforms = {
            "LinkedIn": "https://www.linkedin.com/jobs/search/?keywords={query}&location={location}",
            "Indeed": "https://in.indeed.com/jobs?q={query}&l={location}",
            "Glassdoor": "https://www.glassdoor.com/Job/jobs.htm?sc.keyword={query}"
        }

        for title in titles:
            query = urllib.parse.quote_plus(title)
            loc = urllib.parse.quote_plus(location) if location else ""
            for platform, url_template in base_platforms.items():
                url = url_template.format(query=query, location=loc)
                job_links.append((f"{title} ({platform})", url))

        return job_links[:MAX_JOBS]


# Helper functions
def load_latest_assessment_results(file_prefix: str) -> Optional[Dict]:
    """Load assessment results from JSON files"""
    files = list(ASSESSMENTS_DIR.glob(f"{file_prefix}_*.json"))
    if not files:
        return None
    latest_file = max(files, key=lambda f: f.stat().st_mtime)
    with open(latest_file) as f:
        return json.load(f)

def transform_mcq_data(raw_data: Dict) -> List[Dict]:
    """Transform MCQ results to standardized format"""
    return [{
        "question": q.get("question", ""),
        "user_answer": q.get("user_answer", ""),
        "correct_answer": q.get("answer", ""),
        "is_correct": q.get("is_correct", False),
        "difficulty": q.get("difficulty", "N/A"),
        "explanation": q.get("explanation", "")
    } for q in raw_data.get("questions", [])] if raw_data else []

def transform_technical_data(raw_data: Dict) -> List[Dict]:
    """Transform technical assessment data"""
    return [{
        "question": q.get("question", ""),
        "answer": q.get("answer", ""),
        "score": q.get("evaluation", {}).get("score", 0),
        "feedback": q.get("evaluation", {}).get("feedback", ""),
        "difficulty": q.get("difficulty", "N/A")
    } for q in raw_data.get("questions", [])] if raw_data else []

def calculate_scores(questions: List[Dict], assessment_type: str) -> Dict:
    """Calculate assessment scores"""
    if not questions:
        return {"correct": 0, "total": 0, "average": 0, "percentage": 0}
    
    if assessment_type == "mcq":
        correct = sum(1 for q in questions if q.get("is_correct", False))
        total = len(questions)
        percentage = (correct / total) * 100 if total > 0 else 0
        return {
            "correct": correct,
            "total": total,
            "percentage": percentage,
            "average": percentage / 10
        }
    else:
        scores = [q.get("score", 0) for q in questions]
        average = sum(scores) / len(scores) if scores else 0
        return {
            "average": average,
            "percentage": average * 10,
            "correct": sum(1 for s in scores if s >= 5),
            "total": len(scores)
        }

def display_question_analysis(mcq_questions: List[Dict], tech_questions: List[Dict]):
    """Display detailed question analysis"""
    st.subheader("🔍 Question Analysis")
    
    tab1, tab2 = st.tabs(["MCQ Questions", "Technical Questions"])
    
    with tab1:
        if mcq_questions:
            st.dataframe(pd.DataFrame([{
                "Question": q["question"][:50] + "...",
                "Your Answer": q["user_answer"],
                "Correct": q["correct_answer"],
                "Result": "✅" if q["is_correct"] else "❌",
                "Explanation": q["explanation"]
            } for q in mcq_questions]), hide_index=True)
        else:
            st.warning("No MCQ data available")
    
    with tab2:
        if tech_questions:
            st.dataframe(pd.DataFrame([{
                "Question": q["question"][:50] + "...",
                "Score": f"{q['score']:.1f}/10",
                "Feedback": q["feedback"],
                "Difficulty": q["difficulty"]
            } for q in tech_questions]), hide_index=True)
        else:
            st.warning("No technical data available")

def generate_ai_recommendations(domain: str, mcq_questions: List[Dict], tech_questions: List[Dict]):
    """Generate personalized recommendations using LLM"""
    st.header("🧠 AI Performance Analysis")
    
    with st.spinner("Analyzing your results..."):
        try:
            context = "\n".join([
                f"Q: {q['question']}\nA: {q.get('user_answer', q.get('answer', ''))}"
                for q in (mcq_questions + tech_questions)[:5]  # Sample first 5
            ])
            
            response = groq_client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[{
                    "role": "user",
                    "content": f"""Analyze this {domain} assessment performance:
                    {context}
                    
                    Provide:
                    1. Key strengths
                    2. Main improvement areas  
                    3. Suggested focus areas"""
                }],
                temperature=0.5
            )
            
            st.markdown(response.choices[0].message.content)
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")

def show_dashboard():
    """Main dashboard interface"""
    st.title("🎯 Career Assessment Dashboard")
    
    # Load assessment data
    mcq_data = load_latest_assessment_results("mcq")
    tech_data = load_latest_assessment_results("technical")
    
    if not mcq_data and not tech_data:
        st.warning("No assessment data found")
        return
    
    # Transform data
    mcq_questions = transform_mcq_data(mcq_data)
    tech_questions = transform_technical_data(tech_data)
    domain = mcq_data.get("metadata", {}).get("domain") if mcq_data else tech_data.get("domain", "Unknown")
    
    # Display domain and scores
    st.subheader(f"Domain: {domain}")
    mcq_scores = calculate_scores(mcq_questions, "mcq")
    tech_scores = calculate_scores(tech_questions, "technical")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("MCQ Score", 
                f"{mcq_scores['correct']}/{mcq_scores['total']}", 
                f"{mcq_scores['percentage']:.1f}%")
    with col2:
        st.metric("Technical Score",
                f"{tech_scores['average']:.1f}/10",
                f"{tech_scores['percentage']:.1f}%")
    
    # Visualization
    st.plotly_chart(px.bar(
        x=["MCQ", "Technical"],
        y=[mcq_scores['percentage'], tech_scores['percentage']],
        title="Performance Comparison"
    ), use_container_width=True)
    
    # Detailed analysis
    display_question_analysis(mcq_questions, tech_questions)
    generate_ai_recommendations(domain, mcq_questions, tech_questions)
    
    # Initialize career agent
    agent = CareerAgent(domain)
    
    # Course recommendations
    st.header("📚 Learning Recommendations")
    with st.expander("Search Courses"):
        course_links = agent.fetch_course_links()
        for name, url in course_links:
            st.markdown(f"- [{name}]({url})")
    
    # Job recommendations
    st.header("💼 Job Opportunities")
    location = st.text_input("Preferred location (optional):")
    if st.button("Find Jobs"):
        with st.spinner("Searching job portals..."):
            job_links = agent.search_job_portals(location)
            for title, url in job_links:
                st.markdown(f"- [{title}]({url})")

if __name__ == "__main__":
    show_dashboard()