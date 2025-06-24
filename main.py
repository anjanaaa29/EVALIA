import json
import time
import logging
import traceback
from datetime import datetime, timedelta
from pathlib import Path

import streamlit as st
from streamlit_chat import message

from src.domain import DomainIdentifier
from src.mcq import MCQGenerator
from src.technical import TechnicalAssessment
from src.dashboard import show_dashboard
from src.chatbot import chatbot_page

# Constants
BOT_ICON = "🤖"
USER_ICON = "👤"
THINKING_ICON = "💭"
ASSESSMENTS_DIR = Path("assessments")
ASSESSMENTS_DIR.mkdir(exist_ok=True)
TOTAL_TIME_LIMIT = 15 * 60  # 15 minutes in seconds for entire interview

# Initialize session state
def init_session_state():
    if "current_stage" not in st.session_state:
        st.session_state.current_stage = "welcome"
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "mcq_answers" not in st.session_state:
        st.session_state.mcq_answers = []
    if "technical_answers" not in st.session_state:
        st.session_state.technical_answers = []
    if "current_mcq_index" not in st.session_state:
        st.session_state.current_mcq_index = 0
    if "current_tech_index" not in st.session_state:
        st.session_state.current_tech_index = 0
    if "domain" not in st.session_state:
        st.session_state.domain = ""
    if "jd_text" not in st.session_state:
        st.session_state.jd_text = ""
    if "assessment_started" not in st.session_state:
        st.session_state.assessment_started = False
    if "interview_start_time" not in st.session_state:
        st.session_state.interview_start_time = None
    if "time_remaining" not in st.session_state:
        st.session_state.time_remaining = TOTAL_TIME_LIMIT
    if "timer_expired" not in st.session_state:
        st.session_state.timer_expired = False

def save_assessment_data():
    """Save all assessment data to JSON files"""
    timestamp = int(time.time())
    candidate_id = f"user_{timestamp}"
    
    # Save MCQ data
    if st.session_state.mcq_answers:
        mcq_data = {
            "metadata": {
                "candidate_id": candidate_id,
                "domain": st.session_state.domain,
                "timestamp": timestamp,
                "time_taken": TOTAL_TIME_LIMIT - st.session_state.time_remaining
            },
            "questions": st.session_state.mcq_answers
        }
        with open(ASSESSMENTS_DIR / f"mcq_{timestamp}.json", "w") as f:
            json.dump(mcq_data, f)
    
    # Save Technical data
    if st.session_state.technical_answers:
        tech_data = {
            "candidate_id": candidate_id,
            "domain": st.session_state.domain,
            "timestamp": timestamp,
            "questions": st.session_state.technical_answers
        }
        with open(ASSESSMENTS_DIR / f"technical_{timestamp}.json", "w") as f:
            json.dump(tech_data, f)

def add_bot_message(content):
    """Add a bot message to the chat"""
    st.session_state.messages.append({"role": "bot", "content": content})

def add_user_message(content):
    """Add a user message to the chat"""
    st.session_state.messages.append({"role": "user", "content": content})

def display_timer():
    """Display countdown timer for the entire interview"""
    if st.session_state.current_stage in ["mcq", "technical"] and st.session_state.interview_start_time:
        mins, secs = divmod(st.session_state.time_remaining, 60)
        timer_text = f"⏰ Time Remaining: {mins:02d}:{secs:02d}"
        
        # Change color when time is running low
        if st.session_state.time_remaining < 300:  # 5 minutes
            timer_color = "red"
        else:
            timer_color = "green"
            
        st.markdown(
            f"""
            <div style="
                position: fixed;
                top: 10px;
                right: 10px;
                background: white;
                padding: 5px 10px;
                border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                border: 1px solid #ddd;
                color: {timer_color};
                font-weight: bold;
                z-index: 1000;
            ">
                {timer_text}
            </div>
            """,
            unsafe_allow_html=True
        )

def check_time_limit():
    """Check if time limit has been reached"""
    if st.session_state.interview_start_time and not st.session_state.timer_expired:
        elapsed = time.time() - st.session_state.interview_start_time
        st.session_state.time_remaining = max(0, TOTAL_TIME_LIMIT - int(elapsed))
        
        if st.session_state.time_remaining <= 0:
            st.session_state.timer_expired = True
            handle_time_expired()

def handle_time_expired():
    """Handle what happens when time expires"""
    # Mark unanswered MCQ questions as skipped
    if st.session_state.current_stage == "mcq":
        for i in range(st.session_state.current_mcq_index, len(st.session_state.mcq_answers)):
            st.session_state.mcq_answers[i]["user_answer"] = "Skipped (Time expired)"
            st.session_state.mcq_answers[i]["is_correct"] = False
    
    # Mark unanswered technical questions as skipped
    elif st.session_state.current_stage == "technical":
        for i in range(st.session_state.current_tech_index, len(st.session_state.technical_answers)):
            st.session_state.technical_answers[i]["answer"] = "Skipped (Time expired)"
            st.session_state.technical_answers[i]["evaluation"] = {
                "score": 0,
                "feedback": "Question not answered due to time limit"
            }
    
    add_bot_message(f"{BOT_ICON} ⏰ Time's up! The interview has ended automatically.")
    save_assessment_data()
    st.session_state.current_stage = "dashboard"
    st.rerun()

def initialize_timer(duration_minutes=15):
    """Initialize the timer state variables"""
    if 'initial_time' not in st.session_state:
        st.session_state.initial_time = duration_minutes * 60
    if 'time_remaining' not in st.session_state:
        st.session_state.time_remaining = st.session_state.initial_time
    if 'timer_start_time' not in st.session_state:
        st.session_state.timer_start_time = time.time()
    if 'timer_expired' not in st.session_state:
        st.session_state.timer_expired = False

def display_chat():
    """Display the chat interface"""
    st.title("EVALIA")
    st.subheader("Your Interview Prep Bot")
    
    # Show timer if in assessment stages
    if st.session_state.current_stage in ["mcq", "technical"]:
        display_timer()
    
    # Chat container
    chat_container = st.container()
    
    # Display chat messages
    with chat_container:
        for i, msg in enumerate(st.session_state.messages):
            if msg["role"] == "user":
                message(msg["content"], is_user=True, key=f"user_{i}", avatar_style="thumbs")
            else:
                message(msg["content"], key=f"bot_{i}", avatar_style="bottts")
    
    # Handle different stages
    if st.session_state.current_stage == "welcome":
        handle_welcome_stage()
    elif st.session_state.current_stage == "domain":
        handle_domain_stage()
    elif st.session_state.current_stage == "mcq":
        handle_mcq_stage()
    elif st.session_state.current_stage == "technical":
        handle_technical_stage()
    elif st.session_state.current_stage == "dashboard":
        show_dashboard()
        display_next_steps_buttons()
    elif st.session_state.current_stage == "chatbot":
        chatbot_page()

def handle_welcome_stage():
    """Initial welcome message and JD input"""
    if not st.session_state.messages:
        welcome_msg = f"""
        {BOT_ICON} **Hi! I'm Evalia, your interview prep bot**
        
        I'll guide you through a career assessment that includes:  
        - 10 MCQ questions  
        - 3 Technical questions  
        
        You'll have a total of 15 minutes to complete the entire interview.
        
        Please paste the Job Description you're targeting below:
        """
        add_bot_message(welcome_msg)
        st.rerun()
    
    if st.session_state.messages[-1]["role"] == "bot" and "Job Description" in st.session_state.messages[-1]["content"]:
        with st.container():
            jd_text = st.text_area("Type your Job Description here:", key="jd_input", label_visibility="collapsed")
            
            if st.button("Submit JD"):
                if jd_text.strip():
                    st.session_state.jd_text = jd_text
                    add_user_message(jd_text)
                    st.session_state.current_stage = "domain"
                    st.rerun()
                else:
                    st.warning("Please enter a job description")

def handle_domain_stage():
    """Identify domain from JD"""
    if len(st.session_state.messages) == 2:  # Only welcome and JD submitted
        with st.spinner(f"{THINKING_ICON} Analyzing your JD..."):
            domain_identifier = DomainIdentifier()
            domain = domain_identifier.predict_domain(st.session_state.jd_text)
            st.session_state.domain = domain
            
            domain_msg = f"""
            {BOT_ICON} I've analyzed your JD and identified the domain as:  
            **{domain}**  
            
            The assessment will include:  
            - 10 MCQ questions  
            - 3 Technical questions  
            
            You have a total of 15 minutes to complete the entire interview.
            
            Ready to begin? (Type 'yes' to continue)
            """
            add_bot_message(domain_msg)
            st.rerun()
    
    if st.session_state.messages[-1]["role"] == "bot" and "Ready to begin" in st.session_state.messages[-1]["content"]:
        with st.container():
            user_input = st.text_input("Type 'yes' to continue or 'no' to change JD:", key="confirm_start", label_visibility="collapsed")
            
            if st.button("Submit"):
                if user_input.lower() == 'yes':
                    add_user_message("yes")
                    st.session_state.current_stage = "mcq"
                    st.session_state.assessment_started = True
                    st.session_state.interview_start_time = time.time()  # Start the timer here
                    st.rerun()
                elif user_input.lower() == 'no':
                    add_user_message("no")
                    st.session_state.messages = []
                    st.session_state.current_stage = "welcome"
                    st.rerun()
                else:
                    st.warning("Please type 'yes' or 'no'")

def handle_mcq_stage():
    """MCQ assessment section with timer"""
    # Initialize timer if not already started
    if "mcq_start_time" not in st.session_state:
        st.session_state.mcq_start_time = time.time()
        st.session_state.time_remaining = 15 * 60  # 15 minutes in seconds
        st.session_state.timer_expired = False
        st.session_state.time_remaining = st.session_state.initial_time
    
    # Calculate remaining time
    current_time = time.time()
    elapsed = current_time - st.session_state.mcq_start_time
    st.session_state.time_remaining = max(0, 15*60 - int(elapsed))
    
    # Display the timer prominently
    display_timer()
    
    # Check if time has expired
    if st.session_state.time_remaining <= 0 and not st.session_state.timer_expired:
        handle_time_expiration()
        return
    
    # Generate questions if not already done
    if not st.session_state.mcq_answers:
        with st.spinner(f"{THINKING_ICON} Generating MCQ questions..."):
            mcq_generator = MCQGenerator()
            mcq_generator.domain = st.session_state.domain
            
            questions = []
            attempts = 0
            max_attempts = 15
            
            while len(questions) < 10 and attempts < max_attempts:
                question = mcq_generator.generate_mcq(job_description=st.session_state.jd_text)
                if question:
                    transformed = {
                        "question": question["question"],
                        "options": list(question["options"].values()),
                        "answer": question["options"][question["answer"]],
                        "explanation": question["explanation"],
                        "difficulty": question["difficulty"]
                    }
                    questions.append(transformed)
                attempts += 1
            
            if len(questions) < 10:
                st.error(f"Only generated {len(questions)} valid questions out of 10")
            
            st.session_state.mcq_answers = questions
            st.rerun()
    
    # If we haven't shown the first question yet
    if st.session_state.current_mcq_index == 0 and not any("MCQ Question 1" in msg["content"] for msg in st.session_state.messages):
        add_bot_message(f"{BOT_ICON} Let's begin with the MCQ round! You have 15 minutes to complete 10 questions.")
        show_next_mcq_question()
        st.rerun()
    
    # Handle user response to current question
    if st.session_state.messages[-1]["role"] == "bot" and "MCQ Question" in st.session_state.messages[-1]["content"]:
        with st.container():
            user_answer = st.text_input("Enter your answer (a, b, c, or d):", 
                                     key=f"mcq_answer_{st.session_state.current_mcq_index}", 
                                     label_visibility="collapsed").lower()
            
            if st.button("Submit Answer"):
                if user_answer in ['a', 'b', 'c', 'd']:
                    # Record answer
                    current_question = st.session_state.mcq_answers[st.session_state.current_mcq_index]
                    option_index = ord(user_answer) - ord('a')
                    user_answer_text = current_question["options"][option_index]
                    
                    add_user_message(f"Answer: {user_answer}) {user_answer_text}")
                    
                    # Check if correct
                    is_correct = (user_answer_text == current_question["answer"])
                    st.session_state.mcq_answers[st.session_state.current_mcq_index]["user_answer"] = user_answer_text
                    st.session_state.mcq_answers[st.session_state.current_mcq_index]["is_correct"] = is_correct
                    
                    # Show feedback
                    feedback = f"{BOT_ICON} Your answer is {'correct' if is_correct else 'incorrect'}."
                    if "explanation" in current_question:
                        feedback += f"\n\nExplanation: {current_question['explanation']}"
                    add_bot_message(feedback)
                    
                    # Move to next question or end
                    st.session_state.current_mcq_index += 1
                    if st.session_state.current_mcq_index < len(st.session_state.mcq_answers):
                        show_next_mcq_question()
                    else:
                        add_bot_message(f"{BOT_ICON} ✅ MCQ round completed! Moving to technical questions...")
                        st.session_state.current_stage = "technical"
                    st.rerun()
                else:
                    st.warning("Please enter a valid option (a, b, c, or d)")

def initialize_timer(duration_minutes=15):
    """Initialize the timer state variables"""
    if 'initial_time' not in st.session_state:
        st.session_state.initial_time = duration_minutes * 60
    if 'time_remaining' not in st.session_state:
        st.session_state.time_remaining = st.session_state.initial_time
    if 'timer_start_time' not in st.session_state:
        st.session_state.timer_start_time = time.time()
    if 'timer_expired' not in st.session_state:
        st.session_state.timer_expired = False

def display_timer():
    """Display the countdown timer with visual cues"""
    # Initialize timer if not already done
    if 'initial_time' not in st.session_state:
        initialize_timer()
    
    # Calculate time remaining (smooth countdown)
    try:
        elapsed = time.time() - st.session_state.timer_start_time
        st.session_state.time_remaining = max(0, st.session_state.initial_time - int(elapsed))
        
        minutes, seconds = divmod(st.session_state.time_remaining, 60)
        
        # Determine timer style based on remaining time
        if st.session_state.time_remaining <= 60:  # Less than 1 minute
            timer_color = "red"
            timer_style = "blinking"
            timer_icon = "⏰ "
        elif st.session_state.time_remaining <= 300:  # Less than 5 minutes
            timer_color = "orange"
            timer_style = ""
            timer_icon = "⏳ "
        else:
            timer_color = "green"
            timer_style = ""
            timer_icon = "⏱️ "
        
        # Create the timer display
        st.markdown(f"""
        <style>
            .blinking {{
                animation: blink 1s linear infinite;
            }}
            @keyframes blink {{
                0% {{ opacity: 1; }}
                50% {{ opacity: 0.5; }}
                100% {{ opacity: 1; }}
            }}
            .timer-container {{
                background: #c8cbcc;
                padding: 10px 15px;
                border-radius: 8px;
                margin-bottom: 20px;
                border-left: 5px solid {timer_color};
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
        </style>
        <div class="timer-container">
            <h3 style="color: {timer_color}; margin: 0; {timer_style}">
                {timer_icon}Time Remaining: {minutes:02d}:{seconds:02d}
            </h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Check if time has expired
        if st.session_state.time_remaining <= 0 and not st.session_state.timer_expired:
            handle_time_expiration()
            
    except Exception as e:
        st.error(f"Timer error: {str(e)}")
        # Attempt to recover by reinitializing
        initialize_timer()

def handle_time_expiration():
    """Handle what happens when time expires"""
    st.session_state.timer_expired = True
    
    # Show time's up message
    time_up_msg = f"""
    {BOT_ICON} ⏰ <span style='color: red; font-weight: bold;'>TIME'S UP!</span> ⏰
    
    The 15-minute MCQ round has ended. Here's what happens next:
    
    - Your answers have been automatically saved
    - {len(st.session_state.mcq_answers) - st.session_state.current_mcq_index} unanswered questions marked as skipped
    - We'll now proceed to your results dashboard
    """
    add_bot_message(time_up_msg)
    
    # Mark unanswered questions as skipped
    for i in range(st.session_state.current_mcq_index, len(st.session_state.mcq_answers)):
        if "user_answer" not in st.session_state.mcq_answers[i]:
            st.session_state.mcq_answers[i]["user_answer"] = "Skipped (Time expired)"
            st.session_state.mcq_answers[i]["is_correct"] = False
    
    save_assessment_data()
    st.session_state.current_stage = "dashboard"
    st.rerun()

def show_next_mcq_question():
    """Display the next MCQ question in the chat"""
    current_question = st.session_state.mcq_answers[st.session_state.current_mcq_index]
    question_text = f"{BOT_ICON} **MCQ Question {st.session_state.current_mcq_index + 1}/{len(st.session_state.mcq_answers)}**\n\n"
    question_text += f"{current_question['question']}\n\n"
    
    for i, option in enumerate(current_question["options"]):
        question_text += f"{chr(97 + i)}) {option}\n"
    
    question_text += f"\nDifficulty: {current_question.get('difficulty', 'Medium')}"
    add_bot_message(question_text)

def handle_technical_stage():
    """Technical assessment section"""
    # Check time limit at the start of each interaction
    check_time_limit()
    if st.session_state.timer_expired:
        return
    
    if not st.session_state.technical_answers:
        with st.spinner(f"{THINKING_ICON} Generating technical questions..."):
            tech_assessment = TechnicalAssessment()
            tech_assessment.set_domain(st.session_state.domain)
            
            try:
                questions = tech_assessment.generate_questions(st.session_state.jd_text)
                
                if not questions:
                    st.error("Failed to generate technical questions")
                    st.stop()
                
                st.session_state.technical_answers = [
                    {
                        "question": q.get("question", ""),
                        "difficulty": q.get("difficulty", "Medium"),
                        "answer": None,
                        "evaluation": None
                    }
                    for q in questions[:3]
                ]
                st.rerun()
            except Exception as e:
                st.error(f"Error generating questions: {str(e)}")
                st.stop()
    
    # If we haven't shown the first question yet
    if st.session_state.current_tech_index == 0 and not any("Technical Question 1" in msg["content"] for msg in st.session_state.messages):
        add_bot_message(f"{BOT_ICON} Let's begin the technical round!")
        show_next_tech_question()
        st.rerun()
    
    # Handle user response to current question
    if st.session_state.messages[-1]["role"] == "bot" and "Technical Question" in st.session_state.messages[-1]["content"]:
        with st.container():
            user_answer = st.text_area("Type your answer here:", key=f"tech_answer_{st.session_state.current_tech_index}", height=150, label_visibility="collapsed")
            
            if st.button("Submit Answer"):
                if user_answer.strip():
                    # Store answer
                    add_user_message(user_answer)
                    current_question = st.session_state.technical_answers[st.session_state.current_tech_index]
                    current_question["answer"] = user_answer
                    
                    # Evaluate answer
                    with st.spinner(f"{THINKING_ICON} Evaluating your answer..."):
                        tech_assessment = TechnicalAssessment()
                        tech_assessment.set_domain(st.session_state.domain)
                        
                        evaluation = tech_assessment.evaluate_response(
                            question=current_question["question"],
                            answer=user_answer
                        )
                        
                        current_question["evaluation"] = evaluation
                        
                        # Show feedback
                        feedback = f"{BOT_ICON} **Feedback for Question {st.session_state.current_tech_index + 1}:**\n\n"
                        feedback += f"Score: {evaluation.get('score', 0)}/10\n\n"
                        feedback += f"Evaluation: {evaluation.get('feedback', 'No feedback available')}\n\n"
                        
                        if "suggested_study_topics" in evaluation:
                            feedback += "Suggested Study Topics:\n"
                            for topic in evaluation["suggested_study_topics"]:
                                feedback += f"- {topic}\n"
                        
                        add_bot_message(feedback)
                        
                        # Move to next question or end
                        st.session_state.current_tech_index += 1
                        if st.session_state.current_tech_index < len(st.session_state.technical_answers):
                            show_next_tech_question()
                        else:
                            add_bot_message(f"{BOT_ICON} ✅ Technical round completed! Generating your dashboard...")
                            save_assessment_data()
                            st.session_state.current_stage = "dashboard"
                        st.rerun()
                else:
                    st.warning("Please enter an answer before submitting")

def show_next_tech_question():
    """Display the next technical question in the chat"""
    current_question = st.session_state.technical_answers[st.session_state.current_tech_index]
    question_text = f"{BOT_ICON} **Technical Question {st.session_state.current_tech_index + 1}/{len(st.session_state.technical_answers)}**\n\n"
    question_text += f"{current_question['question']}\n\n"
    question_text += f"*Difficulty: {current_question.get('difficulty', 'Medium')}*"
    
    # Add criteria as expandable
    # question_text += f"\n\n<details><summary>Evaluation Criteria</summary>{current_question.get('criteria', 'No criteria provided')}</details>"
    
    add_bot_message(question_text)

def display_next_steps_buttons():
    """Display buttons for next steps after assessment completion"""
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Restart Assessment", use_container_width=True):
            reset_assessment()
    
    with col2:
        if st.button("💬 Talk to Interview Coach", use_container_width=True):
            st.session_state.current_stage = "chatbot"
            st.rerun()

def reset_assessment():
    """Reset the assessment to initial state"""
    st.session_state.messages = []
    st.session_state.current_stage = "welcome"
    st.session_state.mcq_answers = []
    st.session_state.technical_answers = []
    st.session_state.current_mcq_index = 0
    st.session_state.current_tech_index = 0
    st.session_state.assessment_started = False
    st.session_state.interview_start_time = None
    st.session_state.time_remaining = TOTAL_TIME_LIMIT
    st.session_state.timer_expired = False
    st.rerun()

def main():
    # Configure page
    st.set_page_config(
        page_title="EVALIA - Interview Prep Bot",
        page_icon="🤖",
        layout="wide"
    )
    
    # Initialize session state
    init_session_state()
    
    # Check time limit on each rerun (for cases where user isn't actively interacting)
    if st.session_state.current_stage in ["mcq", "technical"]:
        check_time_limit()
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
        .stTextArea textarea {
            min-height: 150px;
        }
        .stButton button {
            width: 100%;
        }
        .stChatMessage {
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 12px;
        }
        .stMarkdown {
            line-height: 1.6;
        }
        details {
            background-color: #2b2b2c;
            padding: 8px;
            border-radius: 4px;
            margin-top: 8px;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Main app logic
    if st.session_state.current_stage == "chatbot":
        chatbot_page()
    else:
        display_chat()

if __name__ == "__main__":
    main()