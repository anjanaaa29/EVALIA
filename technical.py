import json
import os
from datetime import datetime
from typing import List, Dict, Optional
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class TechnicalAssessment:
    def __init__(self):
        """Initialize the technical assessment system."""
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        self.client = Groq(api_key=api_key)
        self.domain = None
        self.results_dir = "assessments"
        self.ensure_results_dir_exists()

    def ensure_results_dir_exists(self):
        """Ensure the assessments directory exists."""
        os.makedirs(self.results_dir, exist_ok=True)

    def set_domain(self, domain: str) -> None:
        """Set the assessment domain."""
        if not domain.strip():
            raise ValueError("Domain cannot be empty")
        self.domain = domain.strip()

    def generate_questions(self, job_description: str) -> List[Dict]:
        """Generate technical questions using LLM."""
        if not self.domain:
            raise ValueError("Domain must be set before generating questions")
            
        if not job_description.strip():
            raise ValueError("Job description cannot be empty")

        prompt = f"""Generate 3 technical theory interview questions for {self.domain} role.
        
        Job Context:
        {job_description}

        For each question include:
        - question: The technical question
        - difficulty: Easy/Medium/Hard
        - do not repeat the questions
        - do not ask coding questions
        - do not include evaluation criteria

        Return as JSON with 'questions' array containing these objects."""

        try:
            response = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a technical interviewer. Generate clear, relevant questions without evaluation criteria."
                    },
                    {"role": "user", "content": prompt}
                ],
                model="llama3-70b-8192",
                response_format={"type": "json_object"},
                temperature=0.6,
                timeout=30
            )
            
            result = json.loads(response.choices[0].message.content)
            return result.get("questions", [])
            
        except Exception as e:
            print(f"Error generating questions: {str(e)}")
            return []

    def evaluate_response(self, question: str,answer: str) -> Dict:
        """Evaluate a technical answer."""
        prompt = f"""Evaluate this technical response (0-10 scale):
        
        Question: {question}
        Response: {answer}

        Provide:
        - score: Numerical assessment (0-10)
        - feedback: Detailed constructive feedback
        - suggested_study_topics: Array of relevant topics to improve

        Return JSON with these fields."""

        try:
            response = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "Be a strict but fair technical evaluator."
                    },
                    {"role": "user", "content": prompt}
                ],
                model="llama3-70b-8192",
                response_format={"type": "json_object"},
                temperature=0.2,
                timeout=30
            )
            
            evaluation = json.loads(response.choices[0].message.content)
            
            if not all(key in evaluation for key in ['score', 'feedback', 'suggested_study_topics']):
                raise ValueError("Invalid evaluation response format")
            
            return evaluation
            
        except Exception as e:
            print(f"Evaluation failed: {str(e)}")
            return {
                "score": 0,
                "feedback": f"Evaluation failed: {str(e)}",
                "suggested_study_topics": []
            }

    def run_assessment(self, job_description: str, num_questions: int = 3) -> Dict:
        """Run complete technical assessment."""
        if not self.domain:
            raise ValueError("Domain must be set before running assessment")
            
        questions = self.generate_questions(job_description)
        if not questions:
            raise RuntimeError("Failed to generate questions")
            
        results = {
            "domain": self.domain,
            "job_description": job_description[:500],
            "start_time": datetime.now().isoformat(),
            "questions": [],
            "total_score": 0,
            "average_score": 0,
            "scores": []
        }
        
        for i, question_data in enumerate(questions[:num_questions], 1):
            question = question_data.get('question', '')
            dummy_answer = f"Sample answer to {question[:30]}..."
            
            evaluation = self.evaluate_response(
                question=question,
                answer=dummy_answer
            )
            
            question_result = {
                "number": i,
                "question": question,
                "answer": dummy_answer,
                "score": evaluation.get("score", 0),
                "feedback": evaluation.get("feedback", ""),
                "suggested_topics": evaluation.get("suggested_study_topics", [])
            }
            
            results["questions"].append(question_result)
            results["total_score"] += question_result["score"]
            results["scores"].append(question_result["score"])
        
        results["end_time"] = datetime.now().isoformat()
        results["average_score"] = results["total_score"] / len(results["questions"])
        
        self.save_assessment_results(results)
        return results

    def save_assessment_results(self, results: Dict) -> str:
        """Save complete assessment results and return file path."""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"assessment_{self.domain}_{timestamp}.json"
            filepath = os.path.join(self.results_dir, filename)
            
            with open(filepath, "w") as f:
                json.dump(results, f, indent=2)
                
            return filepath
        except Exception as e:
            print(f"Failed to save assessment results: {str(e)}")
            return ""

    def get_latest_assessment(self) -> Optional[Dict]:
        """Get the most recent assessment results."""
        try:
            if not os.path.exists(self.results_dir):
                return None
                
            files = [f for f in os.listdir(self.results_dir) if f.endswith('.json')]
            if not files:
                return None
                
            # Get most recent file
            files.sort(reverse=True)
            latest_file = os.path.join(self.results_dir, files[0])
            
            with open(latest_file, "r") as f:
                return json.load(f)
                
        except Exception as e:
            print(f"Error loading assessment: {str(e)}")
            return None

    def get_all_assessments(self) -> List[Dict]:
        """Get all assessment results."""
        assessments = []
        try:
            if os.path.exists(self.results_dir):
                for filename in os.listdir(self.results_dir):
                    if filename.endswith('.json'):
                        filepath = os.path.join(self.results_dir, filename)
                        with open(filepath, "r") as f:
                            assessments.append(json.load(f))
        except Exception as e:
            print(f"Error loading assessments: {str(e)}")
        return assessments