import json
import numpy as np
import streamlit as st
from typing import List, Dict, Any
import pandas as pd
from datetime import datetime
import os
from langchain.llms import Cohere
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

load_dotenv()

# Course Dataset (30+ courses as required)
COURSES = [
    {"id": 1, "title": "Machine Learning Specialization", "provider": "Coursera", "description": "Learn machine learning fundamentals including supervised learning, unsupervised learning, and neural networks. Covers linear regression, logistic regression, decision trees, and deep learning basics.", "skill_level": "intermediate", "tags": ["machine-learning", "python", "data-science", "neural-networks"], "duration": "6 months", "rating": 4.8},
    {"id": 2, "title": "Python for Data Science and Machine Learning", "provider": "Udemy", "description": "Complete Python programming course for data science. Learn pandas, numpy, matplotlib, seaborn, scikit-learn, and machine learning algorithms.", "skill_level": "beginner", "tags": ["python", "data-science", "pandas", "numpy", "beginner"], "duration": "3 months", "rating": 4.6},
    {"id": 3, "title": "Deep Learning Specialization", "provider": "Coursera", "description": "Advanced deep learning course covering neural networks, CNNs, RNNs, LSTMs, and transformer models. Includes practical projects with TensorFlow.", "skill_level": "advanced", "tags": ["deep-learning", "tensorflow", "cnn", "rnn", "advanced"], "duration": "4 months", "rating": 4.9},
    {"id": 4, "title": "Data Structures and Algorithms", "provider": "edX", "description": "Comprehensive course on data structures and algorithms. Covers arrays, linked lists, trees, graphs, sorting, searching, and dynamic programming.", "skill_level": "intermediate", "tags": ["algorithms", "data-structures", "programming", "problem-solving"], "duration": "5 months", "rating": 4.7},
    {"id": 5, "title": "Full Stack Web Development", "provider": "Udemy", "description": "Learn full stack web development with HTML, CSS, JavaScript, React, Node.js, Express, and MongoDB. Build complete web applications.", "skill_level": "beginner", "tags": ["web-development", "javascript", "react", "nodejs", "fullstack"], "duration": "6 months", "rating": 4.5},
    {"id": 6, "title": "AWS Cloud Practitioner", "provider": "AWS", "description": "Introduction to Amazon Web Services cloud computing. Learn AWS core services, security, pricing, and cloud architecture fundamentals.", "skill_level": "beginner", "tags": ["cloud", "aws", "devops", "infrastructure", "beginner"], "duration": "2 months", "rating": 4.4},
    {"id": 7, "title": "Natural Language Processing", "provider": "Coursera", "description": "Learn NLP techniques including text preprocessing, sentiment analysis, named entity recognition, and language models. Uses Python and NLTK.", "skill_level": "intermediate", "tags": ["nlp", "text-processing", "python", "nltk", "linguistics"], "duration": "4 months", "rating": 4.6},
    {"id": 8, "title": "Computer Vision Fundamentals", "provider": "edX", "description": "Introduction to computer vision concepts including image processing, feature detection, object recognition, and deep learning for vision.", "skill_level": "intermediate", "tags": ["computer-vision", "image-processing", "opencv", "deep-learning"], "duration": "3 months", "rating": 4.5},
    {"id": 9, "title": "SQL for Data Analysis", "provider": "Udemy", "description": "Master SQL for data analysis. Learn queries, joins, aggregations, window functions, and database design for data science applications.", "skill_level": "beginner", "tags": ["sql", "database", "data-analysis", "queries", "beginner"], "duration": "2 months", "rating": 4.7},
    {"id": 10, "title": "DevOps Engineering", "provider": "Coursera", "description": "Learn DevOps practices including CI/CD, containerization with Docker, Kubernetes, infrastructure as code, and monitoring.", "skill_level": "intermediate", "tags": ["devops", "docker", "kubernetes", "ci-cd", "automation"], "duration": "5 months", "rating": 4.6},
    {"id": 11, "title": "React Native Mobile Development", "provider": "Udemy", "description": "Build iOS and Android apps using React Native. Learn navigation, state management, API integration, and app deployment.", "skill_level": "intermediate", "tags": ["react-native", "mobile-development", "javascript", "ios", "android"], "duration": "4 months", "rating": 4.3},
    {"id": 12, "title": "Cybersecurity Fundamentals", "provider": "Coursera", "description": "Introduction to cybersecurity concepts including network security, cryptography, risk assessment, and security frameworks.", "skill_level": "beginner", "tags": ["cybersecurity", "networking", "cryptography", "security", "beginner"], "duration": "3 months", "rating": 4.5},
    {"id": 13, "title": "Blockchain Development", "provider": "Udemy", "description": "Learn blockchain technology and smart contract development using Ethereum, Solidity, and Web3.js.", "skill_level": "intermediate", "tags": ["blockchain", "ethereum", "solidity", "web3", "cryptocurrency"], "duration": "5 months", "rating": 4.4},
    {"id": 14, "title": "UI/UX Design Masterclass", "provider": "Coursera", "description": "Comprehensive UI/UX design course covering user research, wireframing, prototyping, and design systems using Figma.", "skill_level": "beginner", "tags": ["ui-ux", "design", "figma", "prototyping", "user-research"], "duration": "4 months", "rating": 4.7},
    {"id": 15, "title": "Android App Development with Kotlin", "provider": "Google", "description": "Learn Android development using Kotlin. Covers activities, fragments, databases, API integration, and Material Design.", "skill_level": "intermediate", "tags": ["android", "kotlin", "mobile-development", "material-design"], "duration": "6 months", "rating": 4.6},
    {"id": 16, "title": "iOS Development with Swift", "provider": "Apple", "description": "Build iOS applications using Swift and SwiftUI. Learn iOS frameworks, app architecture, and App Store deployment.", "skill_level": "intermediate", "tags": ["ios", "swift", "swiftui", "mobile-development", "xcode"], "duration": "6 months", "rating": 4.8},
    {"id": 17, "title": "Google Cloud Platform Fundamentals", "provider": "Google", "description": "Introduction to GCP services including Compute Engine, Cloud Storage, BigQuery, and Kubernetes Engine.", "skill_level": "beginner", "tags": ["gcp", "cloud", "kubernetes", "bigquery", "infrastructure"], "duration": "3 months", "rating": 4.5},
    {"id": 18, "title": "Data Engineering with Apache Spark", "provider": "edX", "description": "Learn big data processing with Apache Spark. Covers RDDs, DataFrames, Spark SQL, and streaming data processing.", "skill_level": "advanced", "tags": ["spark", "big-data", "data-engineering", "scala", "streaming"], "duration": "4 months", "rating": 4.7},
    {"id": 19, "title": "Digital Marketing Analytics", "provider": "Coursera", "description": "Learn digital marketing analytics using Google Analytics, social media metrics, and marketing attribution models.", "skill_level": "beginner", "tags": ["marketing", "analytics", "google-analytics", "social-media", "business"], "duration": "3 months", "rating": 4.4},
    {"id": 20, "title": "Java Programming Masterclass", "provider": "Udemy", "description": "Complete Java programming course covering OOP, collections, multithreading, and enterprise Java development.", "skill_level": "beginner", "tags": ["java", "programming", "oop", "multithreading", "enterprise"], "duration": "8 months", "rating": 4.6},
    {"id": 21, "title": "Game Development with Unity", "provider": "Unity", "description": "Learn game development using Unity engine. Covers 2D/3D games, physics, animations, and mobile game development.", "skill_level": "intermediate", "tags": ["unity", "game-development", "c-sharp", "3d", "mobile-games"], "duration": "5 months", "rating": 4.5},
    {"id": 22, "title": "Kubernetes Administration", "provider": "Linux Foundation", "description": "Learn Kubernetes container orchestration. Covers cluster management, networking, storage, and security.", "skill_level": "advanced", "tags": ["kubernetes", "containers", "devops", "orchestration", "docker"], "duration": "4 months", "rating": 4.8},
    {"id": 23, "title": "Python Web Development with Django", "provider": "Udemy", "description": "Build web applications using Django framework. Covers models, views, templates, authentication, and deployment.", "skill_level": "intermediate", "tags": ["django", "python", "web-development", "backend", "mvc"], "duration": "5 months", "rating": 4.7},
    {"id": 24, "title": "Business Intelligence with Power BI", "provider": "Microsoft", "description": "Learn business intelligence and data visualization using Microsoft Power BI. Create dashboards and reports.", "skill_level": "beginner", "tags": ["power-bi", "business-intelligence", "data-visualization", "microsoft", "dashboards"], "duration": "3 months", "rating": 4.4},
    {"id": 25, "title": "Artificial Intelligence Ethics", "provider": "edX", "description": "Explore ethical considerations in AI development including bias, fairness, transparency, and responsible AI practices.", "skill_level": "intermediate", "tags": ["ai-ethics", "responsible-ai", "bias", "fairness", "philosophy"], "duration": "2 months", "rating": 4.6},
    {"id": 26, "title": "Salesforce Administration", "provider": "Salesforce", "description": "Learn Salesforce platform administration including user management, workflows, reports, and customization.", "skill_level": "beginner", "tags": ["salesforce", "crm", "administration", "workflows", "business"], "duration": "4 months", "rating": 4.5},
    {"id": 27, "title": "C++ Programming for Competitive Programming", "provider": "Coursera", "description": "Master C++ for competitive programming. Covers STL, algorithms, data structures, and optimization techniques.", "skill_level": "advanced", "tags": ["cpp", "competitive-programming", "algorithms", "stl", "optimization"], "duration": "6 months", "rating": 4.8},
    {"id": 28, "title": "Tableau Data Visualization", "provider": "Tableau", "description": "Create interactive data visualizations and dashboards using Tableau. Learn advanced charting and storytelling with data.", "skill_level": "intermediate", "tags": ["tableau", "data-visualization", "dashboards", "business-intelligence", "analytics"], "duration": "3 months", "rating": 4.6},
    {"id": 29, "title": "Microsoft Azure Fundamentals", "provider": "Microsoft", "description": "Introduction to Microsoft Azure cloud services including virtual machines, storage, networking, and security.", "skill_level": "beginner", "tags": ["azure", "cloud", "microsoft", "virtual-machines", "security"], "duration": "2 months", "rating": 4.5},
    {"id": 30, "title": "Network Security and Penetration Testing", "provider": "Udemy", "description": "Learn ethical hacking and penetration testing techniques. Covers vulnerability assessment and security tools.", "skill_level": "advanced", "tags": ["penetration-testing", "ethical-hacking", "network-security", "vulnerability", "kali-linux"], "duration": "6 months", "rating": 4.7},
    {"id": 31, "title": "MongoDB Database Development", "provider": "MongoDB", "description": "Learn NoSQL database development with MongoDB. Covers document modeling, queries, indexing, and aggregation.", "skill_level": "intermediate", "tags": ["mongodb", "nosql", "database", "aggregation", "indexing"], "duration": "3 months", "rating": 4.4},
    {"id": 32, "title": "Rust Programming Language", "provider": "Rust Foundation", "description": "Learn systems programming with Rust. Covers memory safety, concurrency, and performance optimization.", "skill_level": "advanced", "tags": ["rust", "systems-programming", "memory-safety", "concurrency", "performance"], "duration": "5 months", "rating": 4.8},
    {"id": 33, "title": "Vue.js Frontend Development", "provider": "Vue", "description": "Build modern web applications using Vue.js framework. Covers components, routing, state management, and testing.", "skill_level": "intermediate", "tags": ["vuejs", "frontend", "javascript", "spa", "components"], "duration": "4 months", "rating": 4.6},
    {"id": 34, "title": "TensorFlow for Deep Learning", "provider": "TensorFlow", "description": "Deep learning with TensorFlow. Covers neural networks, CNN, RNN, transfer learning, and model deployment.", "skill_level": "advanced", "tags": ["tensorflow", "deep-learning", "neural-networks", "cnn", "transfer-learning"], "duration": "5 months", "rating": 4.9},
    {"id": 35, "title": "Go Programming Language", "provider": "Google", "description": "Learn Go programming for backend development and microservices. Covers concurrency, web services, and cloud development.", "skill_level": "intermediate", "tags": ["golang", "backend", "microservices", "concurrency", "cloud"], "duration": "4 months", "rating": 4.7}
]

class CourseRecommendationSystem:
    def __init__(self):
        self.courses = COURSES
        self.user_feedback = {}  # Store user feedback
        self.user_preferences = {}  # Store learned preferences
        
        # Pre-compute course embeddings using TF-IDF
        self._compute_course_embeddings()
        
    def _compute_course_embeddings(self):
        """Pre-compute TF-IDF vectors for all courses"""
        course_texts = []
        for course in self.courses:
            # Combine title, description, and tags for embedding
            text = f"{course['title']} {course['description']} {' '.join(course['tags'])}"
            course_texts.append(text)
        
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        self.course_embeddings = self.vectorizer.fit_transform(course_texts)
    
    def create_user_profile_embedding(self, background: str, interests: str, goals: str, skills: str = ""):
        """Create user profile embedding using TF-IDF"""
        profile_text = f"Background: {background}. Interests: {interests}. Goals: {goals}. Skills: {skills}"
        return self.vectorizer.transform([profile_text])
    
    def get_recommendations(self, user_profile: str, top_k: int = 5) -> List[Dict]:
        """Get course recommendations based on user profile"""
        # Parse user profile
        parts = user_profile.split("|")
        background = parts[0] if len(parts) > 0 else ""
        interests = parts[1] if len(parts) > 1 else ""
        goals = parts[2] if len(parts) > 2 else ""
        skills = parts[3] if len(parts) > 3 else ""
        
        # Create user embedding
        user_embedding = self.create_user_profile_embedding(background, interests, goals, skills)
        
        # Calculate similarities
        similarities = cosine_similarity(user_embedding, self.course_embeddings).flatten()
        
        # Apply feedback adjustments if available
        if user_profile in self.user_preferences:
            similarities = self._apply_preference_adjustments(similarities, self.user_preferences[user_profile])
        
        # Get top recommendations
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        recommendations = []
        for idx in top_indices:
            course = self.courses[idx].copy()
            course['similarity_score'] = float(similarities[idx])
            recommendations.append(course)
        
        return recommendations
    
    def _apply_preference_adjustments(self, similarities: np.ndarray, preferences: Dict) -> np.ndarray:
        """Apply user preference adjustments to similarity scores"""
        adjusted_similarities = similarities.copy()
        
        for i, course in enumerate(self.courses):
            # Boost scores for liked tags
            for tag in course['tags']:
                if tag in preferences.get('liked_tags', []):
                    adjusted_similarities[i] *= 1.2
                elif tag in preferences.get('disliked_tags', []):
                    adjusted_similarities[i] *= 0.8
        
        return adjusted_similarities
    
    def record_feedback(self, user_profile: str, course_id: int, feedback: str):
        """Record user feedback and update preferences"""
        if user_profile not in self.user_feedback:
            self.user_feedback[user_profile] = []
            self.user_preferences[user_profile] = {'liked_tags': [], 'disliked_tags': []}
        
        # Store feedback
        self.user_feedback[user_profile].append({
            'course_id': course_id,
            'feedback': feedback,
            'timestamp': datetime.now().isoformat()
        })
        
        # Update preferences based on feedback
        course = next(c for c in self.courses if c['id'] == course_id)
        
        if feedback == 'like':
            for tag in course['tags']:
                if tag not in self.user_preferences[user_profile]['liked_tags']:
                    self.user_preferences[user_profile]['liked_tags'].append(tag)
        elif feedback == 'dislike':
            for tag in course['tags']:
                if tag not in self.user_preferences[user_profile]['disliked_tags']:
                    self.user_preferences[user_profile]['disliked_tags'].append(tag)
    
    def answer_learning_question(self, question: str, user_profile: str = "") -> str:
        """
        LLM-powered Q&A using Langchain with Cohere
        """
        # Check if API key is available
        if not os.getenv("COHERE_API_KEY"):
            return "Cohere API key not found. Please set the COHERE_API_KEY environment variable to use this feature."
        
        # Initialize Cohere LLM
        try:
            llm = Cohere(
                cohere_api_key=os.getenv("COHERE_API_KEY"),
                temperature=0.7,
                model="command"
            )
            
            # Create prompt template
            template = """You are an expert education advisor. Given the following question about learning paths and career development, provide a helpful and structured response.

            Question: {question}

            User Profile: {profile}

            Please provide a clear, step-by-step answer with practical recommendations and explanations."""

            prompt = PromptTemplate(
                input_variables=["question", "profile"],
                template=template
            )

            # Create chain
            chain = LLMChain(llm=llm, prompt=prompt)

            # Get response
            response = chain.run(question=question, profile=user_profile)
            
            return response
        except Exception as e:
            return f"Error generating response: {str(e)}"

# Streamlit UI
def main():
    st.title("🎓 AI Course Recommendation System")
    st.markdown("Get personalized course recommendations based on your background, interests, and career goals!")
    
    # Initialize system
    if 'recommender' not in st.session_state:
        st.session_state.recommender = CourseRecommendationSystem()
        st.session_state.current_recommendations = []
        st.session_state.user_profile_key = ""
    
    # Sidebar for user profile
    with st.sidebar:
        st.header("👤 Your Profile")
        
        background = st.text_input(
            "Current Background", 
            placeholder="e.g., Final-year CS student",
            help="Describe your current educational/professional background"
        )
        
        interests = st.text_input(
            "Interests", 
            placeholder="e.g., AI and Data Science",
            help="What topics or technologies interest you?"
        )
        
        goals = st.text_input(
            "Career Goals", 
            placeholder="e.g., Become an ML Engineer",
            help="What are your career aspirations?"
        )
        
        skills = st.text_input(
            "Skills (Optional)", 
            placeholder="e.g., Python:Intermediate, Math:Beginner",
            help="Optional: Rate your current skill levels"
        )
        
        if st.button("🔍 Get Recommendations", type="primary"):
            if background or interests or goals:
                # Create user profile key
                user_profile = f"{background}|{interests}|{goals}|{skills}"
                st.session_state.user_profile_key = user_profile
                
                # Get recommendations
                with st.spinner("Finding the best courses for you..."):
                    recommendations = st.session_state.recommender.get_recommendations(user_profile)
                st.session_state.current_recommendations = recommendations
                st.success("Recommendations updated!")
            else:
                st.error("Please fill in at least one field to get recommendations.")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📚 Recommended Courses")
        
        if st.session_state.current_recommendations:
            for i, course in enumerate(st.session_state.current_recommendations):
                with st.expander(f"#{i+1}: {course['title']}", expanded=(i < 2)):
                    st.write(f"**Provider:** {course['provider']}")
                    st.write(f"**Level:** {course['skill_level'].title()}")
                    st.write(f"**Duration:** {course['duration']}")
                    st.write(f"**Rating:** {'⭐' * int(course['rating'])} ({course['rating']}/5)")
                    st.write(f"**Match Score:** {course['similarity_score']:.3f}")
                    
                    st.write("**Description:**")
                    st.write(course['description'])
                    
                    st.write("**Tags:**")
                    # Display tags in a more compact way
                    tags_html = " ".join([f"<span style='background-color: #f0f2f6; padding: 4px 8px; border-radius: 12px; margin: 4px; display: inline-block; font-size: 0.8em;'>{tag}</span>" for tag in course['tags']])
                    st.markdown(tags_html, unsafe_allow_html=True)
                    
                    # Feedback buttons
                    feedback_col1, feedback_col2 = st.columns(2)
                    with feedback_col1:
                        if st.button(f"👍 Like", key=f"like_{course['id']}"):
                            if st.session_state.user_profile_key:
                                st.session_state.recommender.record_feedback(
                                    st.session_state.user_profile_key, 
                                    course['id'], 
                                    'like'
                                )
                                st.success("Feedback recorded! This will improve future recommendations.")
                    
                    with feedback_col2:
                        if st.button(f"👎 Dislike", key=f"dislike_{course['id']}"):
                            if st.session_state.user_profile_key:
                                st.session_state.recommender.record_feedback(
                                    st.session_state.user_profile_key, 
                                    course['id'], 
                                    'dislike'
                                )
                                st.success("Feedback recorded! This will improve future recommendations.")
        else:
            st.info("👈 Fill in your profile in the sidebar to get personalized course recommendations!")
    
    with col2:
        st.header("🤖 Learning Path Assistant")
        st.markdown("Ask questions about your learning journey!")
        
        question = st.text_area(
            "Ask a question:",
            placeholder="e.g., Should I learn Python or R for data science?",
            height=100
        )
        
        if st.button("💬 Ask"):
            if question:
                with st.spinner("Thinking..."):
                    answer = st.session_state.recommender.answer_learning_question(
                        question, 
                        st.session_state.user_profile_key
                    )
                st.write("**Answer:**")
                st.write(answer)
            else:
                st.error("Please enter a question.")
        
        # Show some example questions
        st.markdown("**Example questions:**")
        example_questions = [
            "Should I learn Python or R for data science?",
            "What are the steps to become an ML engineer?",
            "How do I become a data scientist?",
            "What's the best web development learning path?"
        ]
        
        for eq in example_questions:
            if st.button(f"💡 {eq}", key=f"example_{hash(eq)}"):
                with st.spinner("Thinking..."):
                    answer = st.session_state.recommender.answer_learning_question(
                        eq, 
                        st.session_state.user_profile_key
                    )
                st.write("**Answer:**")
                st.write(answer)
    
    # Footer
    st.markdown("---")
    st.markdown("*Built with Streamlit and TF-IDF*")

if __name__ == "__main__":
    main()