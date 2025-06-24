import streamlit as st
from groq import Groq
import os
from dotenv import load_dotenv

def chatbot_page():
    load_dotenv()
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    if st.button("⬅️ Back to Interview"):
        st.session_state.current_stage="dashboard"
        st.rerun()

    st.title("💬 DISCUSSION CHATBOT")

    # Inject CSS for styled chat bubbles
    st.markdown("""
    <style>
    .chat-container {
        display: flex;
        flex-direction: column;
        gap: 20px;
        margin-top: 30px;
    }
    .chat-row {
        display: flex;
        align-items: flex-start;
    }
    .chat-row.user {
        justify-content: flex-end;
    }
    .chat-row.bot {
        justify-content: flex-start;
    }
    .chat-bubble {
        max-width: 65%;
        padding: 12px 16px;
        border-radius: 18px;
        font-size: 16px;
        line-height: 1.4;
        word-break: break-word;
    }
    .user .chat-bubble {
        background-color: #EFEEEA;
        color: #000;
        border-bottom-right-radius: 0;
    }
    .bot .chat-bubble {
        background-color: #94B4C1;
        color: #000;
        border-bottom-left-radius: 0;
    }
    .avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        font-weight: bold;
        color: #fff;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 10px;
        flex-shrink: 0;
    }
    .avatar.user {
        background-color: #ec407a;
    }
    .avatar.bot {
        background-color: #9c27b0;
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown("""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">
""", unsafe_allow_html=True)

    if "chatbot_history" not in st.session_state:
        st.session_state.chatbot_history = [
            {"role": "system", "content": (
                "You are an AI interview assistant designed to help users with study-related topics, job preparation, job search guidance, and interview improvement strategies. "
                "You can assist with questions about technical concepts, soft skills, resume building, mock interviews, domain-specific topics, and job roles. "
                "Only answer questions that are related to education, career development, job search, interview preparation, or learning strategies. "
                "If the user asks something unrelated (e.g., entertainment, personal issues, unrelated opinions), politely respond that your scope is limited to study and career-related support."
            )}
        ]

    user_input = st.chat_input("Ask me anything...")

    if user_input:
        st.session_state.chatbot_history.append({"role": "user", "content": user_input})
        try:
            response = client.chat.completions.create(
                model="llama3-70b-8192",
                messages=st.session_state.chatbot_history
            )
            reply = response.choices[0].message.content
        except Exception as e:
            reply = f"❌ Error: {e}"

        st.session_state.chatbot_history.append({"role": "assistant", "content": reply})

    # Render chat UI
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for msg in st.session_state.chatbot_history[1:]:  # Skip system prompt
        role = msg["role"]
        content = msg["content"].replace("\n", "<br>")

        if role == "user":
            st.markdown(f'''
        <div class="chat-row user">
            <div class="chat-bubble user">{content}</div>
            <div class="avatar user"><i class="fa-solid fa-circle-user"></i></div>
        </div>
        ''', unsafe_allow_html=True)
        elif role == "assistant":
            st.markdown(f'''
            <div class="chat-row bot">
                <div class="avatar bot"><i class="fa-brands fa-bots"></i></div>
                <div class="chat-bubble bot">{content}</div>
            </div>
        ''', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    chatbot_page()