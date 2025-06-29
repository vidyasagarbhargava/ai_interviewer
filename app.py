import streamlit as st
import requests
import json
from typing import List, Dict, Any
import time
import pandas as pd
from datetime import datetime

# Configure page
st.set_page_config(
    page_title="AI Interview Assistant",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'interview_started' not in st.session_state:
    st.session_state.interview_started = False
if 'current_question' not in st.session_state:
    st.session_state.current_question = 0
if 'questions' not in st.session_state:
    st.session_state.questions = []
if 'answers' not in st.session_state:
    st.session_state.answers = []
if 'scores' not in st.session_state:
    st.session_state.scores = []
if 'feedback' not in st.session_state:
    st.session_state.feedback = []
if 'interview_complete' not in st.session_state:
    st.session_state.interview_complete = False

class AIInterviewer:
    def __init__(self, model_type: str, config: Dict[str, Any]):
        self.model_type = model_type
        self.config = config
    
    def call_azure_openai(self, messages: List[Dict[str, str]]) -> str:
        """Call Azure OpenAI API"""
        try:
            headers = {
                "Content-Type": "application/json",
                "api-key": self.config['api_key']
            }
            
            data = {
                "messages": messages,
                "max_tokens": self.config.get('max_tokens', 1000),
                "temperature": self.config.get('temperature', 0.7)
            }
            
            url = f"{self.config['endpoint']}/openai/deployments/{self.config['deployment_name']}/chat/completions?api-version={self.config['api_version']}"
            
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            st.error(f"Azure OpenAI API Error: {str(e)}")
            return None
    
    def call_ollama(self, messages: List[Dict[str, str]]) -> str:
        """Call Ollama API"""
        try:
            # Convert messages to Ollama format
            prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
            
            data = {
                "model": self.config['model_name'],
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.config.get('temperature', 0.7),
                    "num_predict": self.config.get('max_tokens', 1000)
                }
            }
            
            response = requests.post(f"{self.config['base_url']}/api/generate", json=data)
            response.raise_for_status()
            
            return response.json()['response']
        except Exception as e:
            st.error(f"Ollama API Error: {str(e)}")
            return None
    
    def generate_response(self, messages: List[Dict[str, str]]) -> str:
        """Generate response based on model type"""
        if self.model_type == "Azure OpenAI":
            return self.call_azure_openai(messages)
        elif self.model_type == "Ollama":
            return self.call_ollama(messages)
        else:
            st.error("Invalid model type")
            return None

def create_sidebar():
    """Create sidebar for model configuration"""
    st.sidebar.title("🤖 AI Model Configuration")
    
    # Model selection
    model_type = st.sidebar.selectbox(
        "Select AI Model",
        ["Azure OpenAI", "Ollama"],
        help="Choose between Azure OpenAI or local Ollama model"
    )
    
    config = {}
    
    if model_type == "Azure OpenAI":
        st.sidebar.subheader("Azure OpenAI Settings")
        config['endpoint'] = st.sidebar.text_input(
            "Azure Endpoint",
            placeholder="https://your-resource.openai.azure.com",
            help="Your Azure OpenAI endpoint URL"
        )
        config['api_key'] = st.sidebar.text_input(
            "API Key",
            type="password",
            help="Your Azure OpenAI API key"
        )
        config['deployment_name'] = st.sidebar.text_input(
            "Deployment Name",
            placeholder="gpt-4",
            help="Your Azure OpenAI deployment name"
        )
        config['api_version'] = st.sidebar.text_input(
            "API Version",
            value="2024-02-15-preview",
            help="Azure OpenAI API version"
        )
    
    elif model_type == "Ollama":
        st.sidebar.subheader("Ollama Settings")
        config['base_url'] = st.sidebar.text_input(
            "Ollama Base URL",
            value="http://localhost:11434",
            help="Ollama server URL"
        )
        config['model_name'] = st.sidebar.text_input(
            "Model Name",
            placeholder="llama2, mistral, codellama, etc.",
            help="Ollama model name (e.g., llama2, mistral)"
        )
    
    # Common settings
    st.sidebar.subheader("Model Parameters")
    config['temperature'] = st.sidebar.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=0.7,
        step=0.1,
        help="Controls randomness of responses"
    )
    config['max_tokens'] = st.sidebar.slider(
        "Max Tokens",
        min_value=100,
        max_value=2000,
        value=1000,
        step=100,
        help="Maximum number of tokens in response"
    )
    
    return model_type, config

def validate_config(model_type: str, config: Dict[str, Any]) -> bool:
    """Validate model configuration"""
    if model_type == "Azure OpenAI":
        required_fields = ['endpoint', 'api_key', 'deployment_name']
        return all(config.get(field) for field in required_fields)
    elif model_type == "Ollama":
        required_fields = ['base_url', 'model_name']
        return all(config.get(field) for field in required_fields)
    return False

def start_interview(interviewer: AIInterviewer, topic: str, num_questions: int):
    """Start the interview process"""
    st.session_state.interview_started = True
    st.session_state.current_question = 0
    st.session_state.questions = []
    st.session_state.answers = []
    st.session_state.scores = []
    st.session_state.feedback = []
    st.session_state.interview_complete = False
    st.session_state.topic = topic
    st.session_state.num_questions = num_questions
    
    # Generate first question
    system_prompt = f"""You are a Deep Learning and NLP Research Interviewer. Your task is to interview the candidate on the topic of {topic}.

You will ask exactly {num_questions} questions total. Include a mix of foundational, advanced, and "what-if" scenario-based questions to test both depth and flexibility of understanding.

Please ask the first question about {topic}. Just ask ONE question, nothing else."""
    
    messages = [{"role": "system", "content": system_prompt}]
    
    # Get first question
    first_question = interviewer.generate_response(messages)
    if first_question:
        st.session_state.questions.append(first_question.strip())

def process_answer(interviewer: AIInterviewer, answer: str, topic: str, num_questions: int):
    """Process user answer and get next question or final assessment"""
    current_q = st.session_state.current_question
    
    # Store the answer
    st.session_state.answers.append(answer)
    
    # Check if this is the last question
    if current_q + 1 >= num_questions:
        # This was the last question - get final assessment
        conversation_history = ""
        for i in range(len(st.session_state.questions)):
            conversation_history += f"Question {i+1}: {st.session_state.questions[i]}\n"
            if i < len(st.session_state.answers):
                conversation_history += f"Answer {i+1}: {st.session_state.answers[i]}\n\n"
        
        final_prompt = f"""Based on this interview on {topic}, please provide:

1. Rate the final answer out of 10 with brief justification
2. Then provide a comprehensive assessment in this format:

PERFORMANCE SUMMARY:
Create a table with columns: Question Number | Question | Answer Summary | Score (out of 10) | Feedback

OVERALL ASSESSMENT:
Strengths:
- [List key strengths]

Weaknesses:  
- [List areas needing improvement]

Areas to focus on for improvement:
- [Specific recommendations]

Interview History:
{conversation_history}"""
        
        messages = [
            {"role": "system", "content": f"You are evaluating an interview on {topic}. Provide detailed feedback."},
            {"role": "user", "content": final_prompt}
        ]
        
        final_response = interviewer.generate_response(messages)
        if final_response:
            st.session_state.final_assessment = final_response
            st.session_state.interview_complete = True
    else:
        # Get next question and rate current answer
        conversation_history = ""
        for i in range(len(st.session_state.questions)):
            conversation_history += f"Q{i+1}: {st.session_state.questions[i]}\n"
            if i < len(st.session_state.answers):
                conversation_history += f"A{i+1}: {st.session_state.answers[i]}\n\n"
        
        next_prompt = f"""Based on this interview on {topic}:

{conversation_history}

Please:
1. Rate the most recent answer (A{current_q+1}) out of 10 with brief justification
2. Ask the next question (Question {current_q+2} of {num_questions}) about {topic}

Format your response as:
SCORE: [Rating]/10 - [Brief justification]

NEXT QUESTION: [Your next question]"""
        
        messages = [
            {"role": "system", "content": f"You are interviewing on {topic}. Ask question {current_q+2} of {num_questions}."},
            {"role": "user", "content": next_prompt}
        ]
        
        response = interviewer.generate_response(messages)
        if response:
            # Parse response to extract score and next question
            lines = response.split('\n')
            score_line = ""
            question_lines = []
            
            found_score = False
            found_question = False
            
            for line in lines:
                line = line.strip()
                if line.startswith('SCORE:') or 'score' in line.lower() or '/10' in line:
                    score_line = line
                    found_score = True
                elif line.startswith('NEXT QUESTION:') or (found_score and line and not found_question):
                    if line.startswith('NEXT QUESTION:'):
                        line = line.replace('NEXT QUESTION:', '').strip()
                    question_lines.append(line)
                    found_question = True
                elif found_question and line:
                    question_lines.append(line)
            
            # Store feedback
            if score_line:
                st.session_state.feedback.append(score_line)
            else:
                st.session_state.feedback.append("Score: 7/10 - Good answer")
            
            # Store next question
            if question_lines:
                next_question = ' '.join(question_lines).strip()
                st.session_state.questions.append(next_question)
                st.session_state.current_question += 1
            else:
                # Fallback: generate a simple next question
                fallback_prompt = f"Ask question {current_q+2} of {num_questions} about {topic}. Just ask the question, nothing else."
                fallback_messages = [{"role": "user", "content": fallback_prompt}]
                fallback_response = interviewer.generate_response(fallback_messages)
                if fallback_response:
                    st.session_state.questions.append(fallback_response.strip())
                    st.session_state.current_question += 1

def display_interview_interface(interviewer: AIInterviewer, topic: str, num_questions: int):
    """Display the main interview interface"""
    st.header(f"📋 Interview: {topic}")
    
    if not st.session_state.interview_complete:
        # Show progress
        progress = (st.session_state.current_question + 1) / num_questions
        st.progress(progress)
        st.write(f"Question {st.session_state.current_question + 1} of {num_questions}")
        
        # Show current question
        if st.session_state.questions and st.session_state.current_question < len(st.session_state.questions):
            current_question = st.session_state.questions[st.session_state.current_question]
            st.subheader("❓ Question:")
            st.write(current_question)
            
            # Answer input
            answer = st.text_area(
                "Your Answer:",
                height=150,
                key=f"answer_{st.session_state.current_question}",
                placeholder="Type your answer here..."
            )
            
            col1, col2 = st.columns([1, 4])
            with col1:
                if st.button("Submit Answer", type="primary"):
                    if answer.strip():
                        with st.spinner("Processing your answer..."):
                            process_answer(interviewer, answer, topic, num_questions)
                            st.rerun()
                    else:
                        st.warning("Please provide an answer before submitting.")
            
            # Show previous Q&A if any
            if st.session_state.current_question > 0:
                with st.expander("Previous Questions & Answers"):
                    for i in range(st.session_state.current_question):
                        if i < len(st.session_state.questions) and i < len(st.session_state.answers):
                            st.write(f"**Q{i+1}:** {st.session_state.questions[i]}")
                            st.write(f"**A{i+1}:** {st.session_state.answers[i]}")
                            if i < len(st.session_state.feedback):
                                st.write(f"**Feedback:** {st.session_state.feedback[i]}")
                            st.divider()
        else:
            st.error("No question available. Please restart the interview.")
    else:
        # Show final results
        st.success("🎉 Interview Complete!")
        
        # Display final summary
        st.subheader("📊 Final Assessment")
        if hasattr(st.session_state, 'final_assessment'):
            st.write(st.session_state.final_assessment)
        
        # Create results table
        if st.session_state.questions and st.session_state.answers:
            results_data = []
            for i in range(min(len(st.session_state.questions), len(st.session_state.answers))):
                question = st.session_state.questions[i]
                answer = st.session_state.answers[i]
                feedback = st.session_state.feedback[i] if i < len(st.session_state.feedback) else "No feedback"
                
                results_data.append({
                    "Question": f"Q{i+1}: {question[:100]}..." if len(question) > 100 else f"Q{i+1}: {question}",
                    "Answer": answer[:100] + "..." if len(answer) > 100 else answer,
                    "Feedback": feedback
                })
            
            df = pd.DataFrame(results_data)
            st.dataframe(df, use_container_width=True)
        
        # Restart option
        if st.button("Start New Interview"):
            for key in ['interview_started', 'current_question', 'questions', 'answers', 'scores', 'feedback', 'interview_complete', 'topic', 'num_questions', 'final_assessment']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

def main():
    """Main application function"""
    st.title("🎤 AI Interview Assistant")
    st.markdown("Practice interviews with AI on any topic!")
    
    # Create sidebar
    model_type, config = create_sidebar()
    
    # Validate configuration
    if not validate_config(model_type, config):
        st.warning("⚠️ Please configure your AI model in the sidebar first.")
        st.info("Fill in all required fields to start using the interview assistant.")
        return
    
    # Create AI interviewer instance
    interviewer = AIInterviewer(model_type, config)
    
    # Main interface
    if not st.session_state.interview_started:
        st.subheader("🚀 Setup Your Interview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            topic = st.text_input(
                "Interview Topic",
                placeholder="e.g., Deep Learning, Natural Language Processing, Machine Learning",
                help="Enter the topic you want to be interviewed on"
            )
        
        with col2:
            num_questions = st.number_input(
                "Number of Questions",
                min_value=1,
                max_value=20,
                value=5,
                help="How many questions should the interviewer ask?"
            )
        
        if st.button("🎯 Start Interview", type="primary"):
            if topic.strip():
                with st.spinner("Initializing interview..."):
                    start_interview(interviewer, topic, num_questions)
                    st.rerun()
            else:
                st.warning("Please enter a topic for the interview.")
    
    else:
        # Display interview interface
        topic = st.session_state.get('topic', 'Unknown')
        num_questions = st.session_state.get('num_questions', 5)
        display_interview_interface(interviewer, topic, num_questions)

if __name__ == "__main__":
    main()