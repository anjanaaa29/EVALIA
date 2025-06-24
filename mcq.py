import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class MCQGenerator:
    def __init__(self):
        """Initialize MCQ generator with proper configuration."""
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        self.client = Groq(api_key=api_key)
        self.domain = None
        self.session_id = f"mcq_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.results = {
            "metadata": {
                "domain": None,
                "date": datetime.now().isoformat(),
                "total_questions": 0,
                "model": "llama3-70b-8192"
            },
            "questions": [],
            "score": {
                "correct": 0,
                "total": 0,
                "percentage": 0.0
            }
        }
        Path("assessments").mkdir(exist_ok=True)

    def set_domain(self, domain: str) -> None:
        """Set the assessment domain with validation."""
        if not domain.strip():
            raise ValueError("Domain cannot be empty")
        self.domain = domain.strip()
        self.results["metadata"]["domain"] = self.domain

    def generate_mcq(self, job_description: str) -> Optional[Dict]:
        """Generate a validated MCQ question."""
        if not self.domain:
            raise ValueError("Domain must be set before generating questions")
            
        if not job_description.strip():
            raise ValueError("Job description cannot be empty")

        prompt = f"""Generate one high-quality multiple-choice question for a {self.domain} position based on the job description.

            Job Context:
            {job_description}

            Requirements:
            - Question must be specific to the domain and job requirements
            - Focus on practical, scenario-based questions rather than pure definitions
            - Vary question types (conceptual, situational, technical, problem-solving)
            - Avoid simple recall questions unless absolutely necessary

            Output Format (JSON):
            {{
                "question": "Clear, specific question text",
                "options": {{
                    "a": "Option text",
                    "b": "Option text", 
                    "c": "Option text",
                    "d": "Option text"
                }},
                "answer": "Correct option key (a-d)",
                "explanation": "Brief but meaningful explanation of why this is correct",
                "difficulty": "Easy/Medium/Hard (judged relative to entry-level candidate)",
                "question_type": "technical/situational/conceptual/problem-solving"
            }}

            Rules:
            1. NEVER repeat the same question concept you've generated before
            2. Make all options plausible (no obviously wrong options)
            3. For technical questions, include specific technologies mentioned in the JD
            4. For situational questions, make scenarios realistic
            5. Difficulty should match the seniority level implied by the JD

            Example (for software engineering):
            {{
                "question": "When implementing a microservice that needs to process payments, what's the best approach for handling failed transactions?",
                "options": {{
                    "a": "Silently log the failure and continue",
                    "b": "Implement exponential backoff and retry logic",
                    "c": "Crash the service to trigger alerts",
                    "d": "Queue all transactions for manual review"
                }},
                "answer": "b",
                "explanation": "Exponential backoff prevents overwhelming systems during outages while ensuring eventual processing",
                "difficulty": "Medium",
                "question_type": "technical"
            }}"""
        

        try:
            response = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a technical assessment expert. Generate valid JSON."
                    },
                    {"role": "user", "content": prompt}
                ],
                model="llama3-70b-8192",
                temperature=0.7,
                response_format={"type": "json_object"},
                max_tokens=300,
                timeout=30
            )
            
            # Safely parse and validate response
            mcq = json.loads(response.choices[0].message.content)
            
            # Validate required fields
            required_keys = ['question', 'options', 'answer', 'explanation', 'difficulty']
            if not all(key in mcq for key in required_keys):
                raise ValueError("Invalid MCQ structure")
                
            # Validate options
            if not all(k in mcq['options'] for k in ['a', 'b', 'c', 'd']):
                raise ValueError("Options must include a-d")
                
            if mcq['answer'] not in ['a', 'b', 'c', 'd']:
                raise ValueError("Answer must be a-d")
                
            return mcq
            
        except Exception as e:
            print(f"Error generating MCQ: {str(e)}")
            return None

    def present_question(self, mcq: Dict) -> None:
        """Display and evaluate an MCQ question."""
        if not mcq:
            return
            
        question_num = len(self.results["questions"]) + 1
        
        print(f"\nQuestion {question_num} ({mcq.get('difficulty', 'N/A')}):")
        print(mcq["question"])
        
        for opt, text in mcq["options"].items():
            print(f"{opt}) {text}")
        
        # Get valid user input
        while True:
            user_ans = input("Your answer (a-d): ").lower()
            if user_ans in ['a', 'b', 'c', 'd']:
                break
            print("Invalid input. Please enter a-d")
        
        is_correct = user_ans == mcq["answer"]
        
        # Store results
        self.results["questions"].append({
            **mcq,
            "user_answer": user_ans,
            "is_correct": is_correct,
            "timestamp": datetime.now().isoformat()
        })
        
        # Update score
        self.results["score"]["correct"] += int(is_correct)
        self.results["score"]["total"] += 1
        self.results["score"]["percentage"] = (
            self.results["score"]["correct"] / self.results["score"]["total"] * 100
        )
        
        # Provide feedback
        print(f"\n{'✅ Correct!' if is_correct else '❌ Incorrect!'}")
        print(f"Explanation: {mcq['explanation']}")
        print(f"Correct answer: {mcq['answer']}) {mcq['options'][mcq['answer']]}")
        time.sleep(1)

    def _save_results(self) -> str:
        """Save results to JSON file with error handling."""
        filename = f"assessments/{self.session_id}.json"
        try:
            with open(filename, 'w') as f:
                json.dump(self.results, f, indent=2)
            return filename
        except Exception as e:
            print(f"Error saving results: {str(e)}")
            return ""

    def run_assessment(self, job_description: str, num_questions: int = 7) -> None:
        """Run complete MCQ assessment with validation."""
        if not self.domain:
            raise ValueError("Domain must be set before assessment")
            
        print(f"\nStarting {self.domain} Assessment (Session: {self.session_id})...\n")
        self.results["metadata"]["total_questions"] = num_questions
        
        generated_count = 0
        while generated_count < num_questions:
            mcq = self.generate_mcq(job_description)
            if mcq:
                self.present_question(mcq)
                generated_count += 1
        
        # Save and show results
        filename = self._save_results()
        print(f"\nAssessment Complete!")
        print(f"Final Score: {self.results['score']['correct']}/{num_questions}")
        print(f"Percentage: {self.results['score']['percentage']:.1f}%")
        print(f"Results saved to: {filename}")