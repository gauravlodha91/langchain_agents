import json
import numpy as np
from sentence_transformers import SentenceTransformer
import streamlit as st
from typing import List, Dict, Any
import pandas as pd
from datetime import datetime

# Course Dataset
COURSES = [
    {
        "id": 1,
        "title": "Machine Learning Specialization",
        "provider": "Coursera",
        "description": "Learn machine learning fundamentals including supervised learning, unsupervised learning, and neural networks. Covers linear regression, logistic regression, decision trees, and deep learning basics.",
        "skill_level": "intermediate",
        "tags": ["machine-learning", "python", "data-science", "neural-networks"],
        "duration": "6 months",
        "rating": 4.8
    },
    {
        "id": 2,
        "title": "Python for Data Science and Machine Learning",
        "provider": "Udemy",
        "description": "Complete Python programming course for data science. Learn pandas, numpy, matplotlib, seaborn, scikit-learn, and machine learning algorithms.",
        "skill_level": "beginner",
        "tags": ["python", "data-science", "pandas", "numpy", "beginner"],
        "duration": "3 months",
        "rating": 4.6
    },
    {
        "id": 3,
        "title": "Deep Learning Specialization",
        "provider": "Coursera",
        "description": "Advanced deep learning course covering neural networks, CNNs, RNNs, LSTMs, and transformer models. Includes practical projects with TensorFlow.",
        "skill_level": "advanced",
        "tags": ["deep-learning", "tensorflow", "cnn", "rnn", "advanced"],
        "duration": "4 months",
        "rating": 4.9
    },
    {
        "id": 4,
        "title": "Data Structures and Algorithms",
        "provider": "edX",
        "description": "Comprehensive course on data structures and algorithms. Covers arrays, linked lists, trees, graphs, sorting, searching, and dynamic programming.",
        "skill_level": "intermediate",
        "tags": ["algorithms", "data-structures", "programming", "problem-solving"],
        "duration": "5 months",
        "rating": 4.7
    },
    {
        "id": 5,
        "title": "Full Stack Web Development",
        "provider": "Udemy",
        "description": "Learn full stack web development with HTML, CSS, JavaScript, React, Node.js, Express, and MongoDB. Build complete web applications.",
        "skill_level": "beginner",
        "tags": ["web-development", "javascript", "react", "nodejs", "fullstack"],
        "duration": "6 months",
        "rating": 4.5
    },
    {
        "id": 6,
        "title": "AWS Cloud Practitioner",
        "provider": "AWS",
        "description": "Introduction to Amazon Web Services cloud computing. Learn AWS core services, security, pricing, and cloud architecture fundamentals.",
        "skill_level": "beginner",
        "tags": ["cloud", "aws", "devops", "infrastructure", "beginner"],
        "duration": "2 months",
        "rating": 4.4
    },
    {
        "id": 7,
        "title": "Natural Language Processing",
        "provider": "Coursera",
        "description": "Learn NLP techniques including text preprocessing, sentiment analysis, named entity recognition, and language models. Uses Python and NLTK.",
        "skill_level": "intermediate",
        "tags": ["nlp", "text-processing", "python", "nltk", "linguistics"],
        "duration": "4 months",
        "rating": 4.6
    },
    {
        "id": 8,
        "title": "Computer Vision Fundamentals",
        "provider": "edX",
        "description": "Introduction to computer vision concepts including image processing, feature detection, object recognition, and deep learning for vision.",
        "skill_level": "intermediate",
        "tags": ["computer-vision", "image-processing", "opencv", "deep-learning"],
        "duration": "3 months",
        "rating": 4.5
    },
    {
        "id": 9,
        "title": "SQL for Data Analysis",
        "provider": "Udemy",
        "description": "Master SQL for data analysis. Learn queries, joins, aggregations, window functions, and database design for data science applications.",
        "skill_level": "beginner",
        "tags": ["sql", "database", "data-analysis", "queries", "beginner"],
        "duration": "2 months",
        "rating": 4.7
    },
    {
        "id": 10,
        "title": "DevOps Engineering",
        "provider": "Coursera",
        "description": "Learn DevOps practices including CI/CD, containerization with Docker, Kubernetes, infrastructure as code, and monitoring.",
        "skill_level": "intermediate",
        "tags": ["devops", "docker", "kubernetes", "ci-cd", "automation"],
        "duration": "5 months",
        "rating": 4.6
    }
]

class CourseRecommendationSystem:
    def __init__(self):
        # Initialize sentence transformer model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.courses = COURSES
        self.user_feedback = {}  # Store user feedback
        self.user_preferences = {}  # Store learned preferences
        
        # Pre-compute course embeddings
        self._compute_course_embeddings()
        
    def _compute_course_embeddings(self):
        """Pre-compute embeddings for all courses"""
        course_texts = []
        for course in self.courses:
            # Combine title, description, and tags for embedding
            text = f"{course['title']} {course['description']} {' '.join(course['tags'])}"
            course_texts.append(text)
        
        self.course_embeddings = self.model.encode(course_texts)
    
    def create_user_profile_embedding(self, background: str, interests: str, goals: str, skills: str = ""):
        """Create user profile embedding from input text"""
        profile_text = f"Background: {background}. Interests: {interests}. Goals: {goals}. Skills: {skills}"
        return self.model.encode([profile_text])[0]
    
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
        similarities = np.dot(self.course_embeddings, user_embedding)
        
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
        """Simple rule-based Q&A for learning path questions"""
        question_lower = question.lower()
        
        # Simple knowledge base responses
        if "python" in question_lower and "data science" in question_lower:
            return """For data science, Python is generally recommended over R because:
            1. More versatile and general-purpose
            2. Better for machine learning (scikit-learn, TensorFlow)
            3. Stronger ecosystem and community
            4. Better for deployment and production systems
            
            Recommended learning path: Start with Python basics → pandas/numpy → matplotlib → scikit-learn → advanced ML libraries"""
        
        elif "ml engineer" in question_lower or "machine learning engineer" in question_lower:
            return """Steps to become an ML Engineer:
            1. Master programming (Python/SQL)
            2. Learn statistics and linear algebra
            3. Understand ML algorithms and frameworks
            4. Practice with real datasets and projects
            5. Learn MLOps (deployment, monitoring, CI/CD)
            6. Build a portfolio with end-to-end projects
            7. Gain experience with cloud platforms (AWS/GCP/Azure)"""
        
        elif "data scientist" in question_lower:
            return """Data Science learning path:
            1. Statistics and probability fundamentals
            2. Programming (Python/R and SQL)
            3. Data manipulation (pandas, numpy)
            4. Visualization (matplotlib, seaborn)
            5. Machine learning algorithms
            6. Domain expertise in your area of interest
            7. Communication and storytelling skills"""
        
        elif "web development" in question_lower:
            return """Web Development learning path:
            1. HTML/CSS fundamentals
            2. JavaScript programming
            3. Frontend framework (React/Vue/Angular)
            4. Backend development (Node.js/Python/Java)
            5. Database management (SQL/NoSQL)
            6. Version control (Git)
            7. Deployment and DevOps basics"""
        
        else:
            return f"I'd be happy to help with your learning path question: '{question}'. Could you be more specific about the technology or career path you're interested in? I can provide guidance on data science, machine learning, web development, and more."

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
                    tag_cols = st.columns(min(len(course['tags']), 4))
                    for j, tag in enumerate(course['tags']):
                        with tag_cols[j % 4]:
                            st.badge(tag)
                    
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
                answer = st.session_state.recommender.answer_learning_question(
                    eq, 
                    st.session_state.user_profile_key
                )
                st.write("**Answer:**")
                st.write(answer)
    
    # Footer
    st.markdown("---")
    st.markdown("*Built with Streamlit and SentenceTransformers*")

if __name__ == "__main__":
    main()