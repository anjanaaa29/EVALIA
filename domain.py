import os
from typing import Optional
from groq import Groq
from dotenv import load_dotenv

class DomainIdentifier:
    def __init__(self):
        # Load environment variables from .env file
        load_dotenv()
        
        # Initialize Groq client
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        if not self.client:
            raise ValueError("GROQ_API_KEY not found in environment variables")
            
        self.system_prompt = """You are an expert technical recruiter analyzing job descriptions. 
        Identify the SINGLE most specific professional domain this job belongs to.

        Rules:
        1. Focus on required technical skills
        2. Respond with 1-3 word domain name only
        3. Be specific (e.g., "iOS Development" not just "Mobile")
        4. Use standard industry terminology

        Examples:
        - "Natural Language Processing"
        - "Blockchain Development"
        - "Site Reliability Engineering"
        - "Technical Writing"
        - "Game Development\""""

    def predict_domain(self, job_description: str) -> Optional[str]:
        """
        Predict the most relevant technical domain for a job description.
        
        Args:
            job_description: Text of the job description
            
        Returns:
            str: Predicted domain (e.g., "Data Engineering")
            None: If prediction fails
        """
        if not job_description or not isinstance(job_description, str):
            return None

        try:
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"Job Description:\n{job_description}"}
                ],
                model="llama3-70b-8192",
                temperature=0.3,
                max_tokens=15,
                stop=["\n", ".", ","]
            )
            
            return self._clean_response(response.choices[0].message.content)
            
        except Exception as e:
            print(f"Domain prediction error: {str(e)}")
            return None

    def _clean_response(self, text: str) -> str:
        """Clean and standardize the LLM response"""
        text = text.strip()
        
        # Remove surrounding quotes if present
        if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
            text = text[1:-1]
            
        # Standardize to title case
        return text.title()

# Example Usage
if __name__ == "__main__":
    try:
        identifier = DomainIdentifier()
        
        test_jds = [
            "Looking for a PySpark engineer with experience in big data pipelines and AWS EMR",
            "Seeking UI/UX designer proficient in Figma and user research methodologies",
            "Need Kubernetes administrator with Istio service mesh experience",
            ""
        ]
        
        for jd in test_jds:
            domain = identifier.predict_domain(jd)
            print(f"Input: {jd[:80]}{'...' if len(jd) > 80 else ''}")
            print(f"Domain: {domain or 'Unable to determine'}")
            print("-" * 60)
            
    except Exception as e:
        print(f"Initialization error: {str(e)}")