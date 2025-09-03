import json
import numpy as np
import streamlit as st
from typing import List, Dict, Any
import pandas as pd
from datetime import datetime
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import cohere
from dotenv import load_dotenv
import re
import sqlite3
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

load_dotenv()

# Enhanced Course Dataset with better categorization
COURSES = [
    {"id": 1, "title": "Machine Learning Specialization", "provider": "Coursera", "description": "Learn machine learning fundamentals including supervised learning, unsupervised learning, and neural networks. Covers linear regression, logistic regression, decision trees, and deep learning basics.", "skill_level": "intermediate", "tags": ["machine-learning", "python", "data-science", "neural-networks"], "duration": "6 months", "rating": 4.8, "cost": "paid", "category": "Data Science"},
    {"id": 2, "title": "Python for Data Science and Machine Learning", "provider": "Udemy", "description": "Complete Python programming course for data science. Learn pandas, numpy, matplotlib, seaborn, scikit-learn, and machine learning algorithms.", "skill_level": "beginner", "tags": ["python", "data-science", "pandas", "numpy", "beginner"], "duration": "3 months", "rating": 4.6, "cost": "paid", "category": "Programming"},
    {"id": 3, "title": "Deep Learning Specialization", "provider": "Coursera", "description": "Advanced deep learning course covering neural networks, CNNs, RNNs, LSTMs, and transformer models. Includes practical projects with TensorFlow.", "skill_level": "advanced", "tags": ["deep-learning", "tensorflow", "cnn", "rnn", "advanced"], "duration": "4 months", "rating": 4.9, "cost": "paid", "category": "Data Science"},
    {"id": 4, "title": "Data Structures and Algorithms", "provider": "edX", "description": "Comprehensive course on data structures and algorithms. Covers arrays, linked lists, trees, graphs, sorting, searching, and dynamic programming.", "skill_level": "intermediate", "tags": ["algorithms", "data-structures", "programming", "problem-solving"], "duration": "5 months", "rating": 4.7, "cost": "free", "category": "Programming"},
    {"id": 5, "title": "Full Stack Web Development", "provider": "Udemy", "description": "Learn full stack web development with HTML, CSS, JavaScript, React, Node.js, Express, and MongoDB. Build complete web applications.", "skill_level": "beginner", "tags": ["web-development", "javascript", "react", "nodejs", "fullstack"], "duration": "6 months", "rating": 4.5, "cost": "paid", "category": "Web Development"},
    {"id": 6, "title": "AWS Cloud Practitioner", "provider": "AWS", "description": "Introduction to Amazon Web Services cloud computing. Learn AWS core services, security, pricing, and cloud architecture fundamentals.", "skill_level": "beginner", "tags": ["cloud", "aws", "devops", "infrastructure", "beginner"], "duration": "2 months", "rating": 4.4, "cost": "free", "category": "Cloud Computing"},
    {"id": 7, "title": "Natural Language Processing", "provider": "Coursera", "description": "Learn NLP techniques including text preprocessing, sentiment analysis, named entity recognition, and language models. Uses Python and NLTK.", "skill_level": "intermediate", "tags": ["nlp", "text-processing", "python", "nltk", "linguistics"], "duration": "4 months", "rating": 4.6, "cost": "paid", "category": "Data Science"},
    {"id": 8, "title": "Computer Vision Fundamentals", "provider": "edX", "description": "Introduction to computer vision concepts including image processing, feature detection, object recognition, and deep learning for vision.", "skill_level": "intermediate", "tags": ["computer-vision", "image-processing", "opencv", "deep-learning"], "duration": "3 months", "rating": 4.5, "cost": "free", "category": "Data Science"},
    {"id": 9, "title": "SQL for Data Analysis", "provider": "Udemy", "description": "Master SQL for data analysis. Learn queries, joins, aggregations, window functions, and database design for data science applications.", "skill_level": "beginner", "tags": ["sql", "database", "data-analysis", "queries", "beginner"], "duration": "2 months", "rating": 4.7, "cost": "paid", "category": "Data Science"},
    {"id": 10, "title": "DevOps Engineering", "provider": "Coursera", "description": "Learn DevOps practices including CI/CD, containerization with Docker, Kubernetes, infrastructure as code, and monitoring.", "skill_level": "intermediate", "tags": ["devops", "docker", "kubernetes", "ci-cd", "automation"], "duration": "5 months", "rating": 4.6, "cost": "paid", "category": "DevOps"},
    {"id": 11, "title": "React Native Mobile Development", "provider": "Udemy", "description": "Build iOS and Android apps using React Native. Learn navigation, state management, API integration, and app deployment.", "skill_level": "intermediate", "tags": ["react-native", "mobile-development", "javascript", "ios", "android"], "duration": "4 months", "rating": 4.3, "cost": "paid", "category": "Mobile Development"},
    {"id": 12, "title": "Cybersecurity Fundamentals", "provider": "Coursera", "description": "Introduction to cybersecurity concepts including network security, cryptography, risk assessment, and security frameworks.", "skill_level": "beginner", "tags": ["cybersecurity", "networking", "cryptography", "security", "beginner"], "duration": "3 months", "rating": 4.5, "cost": "free", "category": "Cybersecurity"},
    {"id": 13, "title": "Blockchain Development", "provider": "Udemy", "description": "Learn blockchain technology and smart contract development using Ethereum, Solidity, and Web3.js.", "skill_level": "intermediate", "tags": ["blockchain", "ethereum", "solidity", "web3", "cryptocurrency"], "duration": "5 months", "rating": 4.4, "cost": "paid", "category": "Blockchain"},
    {"id": 14, "title": "UI/UX Design Masterclass", "provider": "Coursera", "description": "Comprehensive UI/UX design course covering user research, wireframing, prototyping, and design systems using Figma.", "skill_level": "beginner", "tags": ["ui-ux", "design", "figma", "prototyping", "user-research"], "duration": "4 months", "rating": 4.7, "cost": "paid", "category": "Design"},
    {"id": 15, "title": "Android App Development with Kotlin", "provider": "Google", "description": "Learn Android development using Kotlin. Covers activities, fragments, databases, API integration, and Material Design.", "skill_level": "intermediate", "tags": ["android", "kotlin", "mobile-development", "material-design"], "duration": "6 months", "rating": 4.6, "cost": "free", "category": "Mobile Development"},
    {"id": 16, "title": "iOS Development with Swift", "provider": "Apple", "description": "Build iOS applications using Swift and SwiftUI. Learn iOS frameworks, app architecture, and App Store deployment.", "skill_level": "intermediate", "tags": ["ios", "swift", "swiftui", "mobile-development", "xcode"], "duration": "6 months", "rating": 4.8, "cost": "free", "category": "Mobile Development"},
    {"id": 17, "title": "Google Cloud Platform Fundamentals", "provider": "Google", "description": "Introduction to GCP services including Compute Engine, Cloud Storage, BigQuery, and Kubernetes Engine.", "skill_level": "beginner", "tags": ["gcp", "cloud", "kubernetes", "bigquery", "infrastructure"], "duration": "3 months", "rating": 4.5, "cost": "free", "category": "Cloud Computing"},
    {"id": 18, "title": "Data Engineering with Apache Spark", "provider": "edX", "description": "Learn big data processing with Apache Spark. Covers RDDs, DataFrames, Spark SQL, and streaming data processing.", "skill_level": "advanced", "tags": ["spark", "big-data", "data-engineering", "scala", "streaming"], "duration": "4 months", "rating": 4.7, "cost": "free", "category": "Data Engineering"},
    {"id": 19, "title": "Digital Marketing Analytics", "provider": "Coursera", "description": "Learn digital marketing analytics using Google Analytics, social media metrics, and marketing attribution models.", "skill_level": "beginner", "tags": ["marketing", "analytics", "google-analytics", "social-media", "business"], "duration": "3 months", "rating": 4.4, "cost": "paid", "category": "Marketing"},
    {"id": 20, "title": "Java Programming Masterclass", "provider": "Udemy", "description": "Complete Java programming course covering OOP, collections, multithreading, and enterprise Java development.", "skill_level": "beginner", "tags": ["java", "programming", "oop", "multithreading", "enterprise"], "duration": "8 months", "rating": 4.6, "cost": "paid", "category": "Programming"},
    {"id": 21, "title": "Game Development with Unity", "provider": "Unity", "description": "Learn game development using Unity engine. Covers 2D/3D games, physics, animations, and mobile game development.", "skill_level": "intermediate", "tags": ["unity", "game-development", "c-sharp", "3d", "mobile-games"], "duration": "5 months", "rating": 4.5, "cost": "free", "category": "Game Development"},
    {"id": 22, "title": "Kubernetes Administration", "provider": "Linux Foundation", "description": "Learn Kubernetes container orchestration. Covers cluster management, networking, storage, and security.", "skill_level": "advanced", "tags": ["kubernetes", "containers", "devops", "orchestration", "docker"], "duration": "4 months", "rating": 4.8, "cost": "paid", "category": "DevOps"},
    {"id": 23, "title": "Python Web Development with Django", "provider": "Udemy", "description": "Build web applications using Django framework. Covers models, views, templates, authentication, and deployment.", "skill_level": "intermediate", "tags": ["django", "python", "web-development", "backend", "mvc"], "duration": "5 months", "rating": 4.7, "cost": "paid", "category": "Web Development"},
    {"id": 24, "title": "Business Intelligence with Power BI", "provider": "Microsoft", "description": "Learn business intelligence and data visualization using Microsoft Power BI. Create dashboards and reports.", "skill_level": "beginner", "tags": ["power-bi", "business-intelligence", "data-visualization", "microsoft", "dashboards"], "duration": "3 months", "rating": 4.4, "cost": "free", "category": "Business Intelligence"},
    {"id": 25, "title": "Artificial Intelligence Ethics", "provider": "edX", "description": "Explore ethical considerations in AI development including bias, fairness, transparency, and responsible AI practices.", "skill_level": "intermediate", "tags": ["ai-ethics", "responsible-ai", "bias", "fairness", "philosophy"], "duration": "2 months", "rating": 4.6, "cost": "free", "category": "AI Ethics"},
    {"id": 26, "title": "Salesforce Administration", "provider": "Salesforce", "description": "Learn Salesforce platform administration including user management, workflows, reports, and customization.", "skill_level": "beginner", "tags": ["salesforce", "crm", "administration", "workflows", "business"], "duration": "4 months", "rating": 4.5, "cost": "free", "category": "Business Software"},
    {"id": 27, "title": "C++ Programming for Competitive Programming", "provider": "Coursera", "description": "Master C++ for competitive programming. Covers STL, algorithms, data structures, and optimization techniques.", "skill_level": "advanced", "tags": ["cpp", "competitive-programming", "algorithms", "stl", "optimization"], "duration": "6 months", "rating": 4.8, "cost": "paid", "category": "Programming"},
    {"id": 28, "title": "Tableau Data Visualization", "provider": "Tableau", "description": "Create interactive data visualizations and dashboards using Tableau. Learn advanced charting and storytelling with data.", "skill_level": "intermediate", "tags": ["tableau", "data-visualization", "dashboards", "business-intelligence", "analytics"], "duration": "3 months", "rating": 4.6, "cost": "paid", "category": "Business Intelligence"},
    {"id": 29, "title": "Microsoft Azure Fundamentals", "provider": "Microsoft", "description": "Introduction to Microsoft Azure cloud services including virtual machines, storage, networking, and security.", "skill_level": "beginner", "tags": ["azure", "cloud", "microsoft", "virtual-machines", "security"], "duration": "2 months", "rating": 4.5, "cost": "free", "category": "Cloud Computing"},
    {"id": 30, "title": "Network Security and Penetration Testing", "provider": "Udemy", "description": "Learn ethical hacking and penetration testing techniques. Covers vulnerability assessment and security tools.", "skill_level": "advanced", "tags": ["penetration-testing", "ethical-hacking", "network-security", "vulnerability", "kali-linux"], "duration": "6 months", "rating": 4.7, "cost": "paid", "category": "Cybersecurity"},
    {"id": 31, "title": "MongoDB Database Development", "provider": "MongoDB", "description": "Learn NoSQL database development with MongoDB. Covers document modeling, queries, indexing, and aggregation.", "skill_level": "intermediate", "tags": ["mongodb", "nosql", "database", "aggregation", "indexing"], "duration": "3 months", "rating": 4.4, "cost": "free", "category": "Database"},
    {"id": 32, "title": "Rust Programming Language", "provider": "Rust Foundation", "description": "Learn systems programming with Rust. Covers memory safety, concurrency, and performance optimization.", "skill_level": "advanced", "tags": ["rust", "systems-programming", "memory-safety", "concurrency", "performance"], "duration": "5 months", "rating": 4.8, "cost": "free", "category": "Programming"},
    {"id": 33, "title": "Vue.js Frontend Development", "provider": "Vue", "description": "Build modern web applications using Vue.js framework. Covers components, routing, state management, and testing.", "skill_level": "intermediate", "tags": ["vuejs", "frontend", "javascript", "spa", "components"], "duration": "4 months", "rating": 4.6, "cost": "free", "category": "Web Development"},
    {"id": 34, "title": "TensorFlow for Deep Learning", "provider": "TensorFlow", "description": "Deep learning with TensorFlow. Covers neural networks, CNN, RNN, transfer learning, and model deployment.", "skill_level": "advanced", "tags": ["tensorflow", "deep-learning", "neural-networks", "cnn", "transfer-learning"], "duration": "5 months", "rating": 4.9, "cost": "free", "category": "Data Science"},
    {"id": 35, "title": "Go Programming Language", "provider": "Google", "description": "Learn Go programming for backend development and microservices. Covers concurrency, web services, and cloud development.", "skill_level": "intermediate", "tags": ["golang", "backend", "microservices", "concurrency", "cloud"], "duration": "4 months", "rating": 4.7, "cost": "free", "category": "Programming"},
    {"id": 36, "title": "Flutter Mobile App Development", "provider": "Google", "description": "Build cross-platform mobile apps with Flutter and Dart. Learn widgets, state management, and app deployment.", "skill_level": "intermediate", "tags": ["flutter", "dart", "mobile-development", "cross-platform", "widgets"], "duration": "5 months", "rating": 4.5, "cost": "free", "category": "Mobile Development"},
    {"id": 37, "title": "Data Science with R", "provider": "Coursera", "description": "Learn data science using R programming. Covers statistics, data visualization with ggplot2, and machine learning.", "skill_level": "intermediate", "tags": ["r-programming", "data-science", "statistics", "ggplot2", "visualization"], "duration": "4 months", "rating": 4.6, "cost": "paid", "category": "Data Science"},
    {"id": 38, "title": "Microservices Architecture", "provider": "Udemy", "description": "Design and build microservices-based applications. Learn service decomposition, API design, and distributed systems.", "skill_level": "advanced", "tags": ["microservices", "architecture", "api-design", "distributed-systems", "scalability"], "duration": "6 months", "rating": 4.7, "cost": "paid", "category": "Software Architecture"},
    {"id": 39, "title": "GraphQL API Development", "provider": "Apollo", "description": "Build modern APIs with GraphQL. Learn schema design, resolvers, subscriptions, and client integration.", "skill_level": "intermediate", "tags": ["graphql", "api", "schema-design", "resolvers", "apollo"], "duration": "3 months", "rating": 4.4, "cost": "free", "category": "Web Development"},
    {"id": 40, "title": "Quantum Computing Fundamentals", "provider": "IBM", "description": "Introduction to quantum computing concepts, quantum algorithms, and programming with Qiskit.", "skill_level": "advanced", "tags": ["quantum-computing", "qiskit", "quantum-algorithms", "physics", "emerging-tech"], "duration": "4 months", "rating": 4.8, "cost": "free", "category": "Emerging Technology"}
]

class DatabaseManager:
    """Enhanced database manager with better performance"""
    
    def __init__(self, db_path: str = "course_recommendations.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database with optimized schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Drop existing tables to reset schema
        cursor.execute('DROP TABLE IF EXISTS user_preferences')
        cursor.execute('DROP TABLE IF EXISTS user_feedback')
        cursor.execute('DROP TABLE IF EXISTS user_sessions')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_profile_hash TEXT,
                course_id INTEGER,
                feedback TEXT,
                timestamp TEXT,
                session_id TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_preferences (
                user_profile_hash TEXT PRIMARY KEY,
                liked_tags TEXT,
                disliked_tags TEXT,
                preferred_providers TEXT,
                preferred_skill_levels TEXT,
                preferred_categories TEXT,
                last_updated TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_sessions (
                session_id TEXT PRIMARY KEY,
                user_profile_hash TEXT,
                created_at TEXT,
                last_activity TEXT,
                total_recommendations INTEGER DEFAULT 0
            )
        ''')
        
        # Create indices for better performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_feedback_user ON user_feedback(user_profile_hash)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_feedback_course ON user_feedback(course_id)')
        
        conn.commit()
        conn.close()
    
    def save_feedback(self, user_profile_hash: str, course_id: int, feedback: str, session_id: str = None):
        """Save user feedback with session tracking"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO user_feedback (user_profile_hash, course_id, feedback, timestamp, session_id)
            VALUES (?, ?, ?, ?, ?)
        ''', (user_profile_hash, course_id, feedback, datetime.now().isoformat(), session_id))
        
        conn.commit()
        conn.close()
    
    def get_user_preferences(self, user_profile_hash: str) -> Dict:
        """Retrieve enhanced user preferences"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT liked_tags, disliked_tags, preferred_providers, preferred_skill_levels, preferred_categories
            FROM user_preferences WHERE user_profile_hash = ?
        ''', (user_profile_hash,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                'liked_tags': json.loads(result[0]) if result[0] else [],
                'disliked_tags': json.loads(result[1]) if result[1] else [],
                'preferred_providers': json.loads(result[2]) if result[2] else [],
                'preferred_skill_levels': json.loads(result[3]) if result[3] else [],
                'preferred_categories': json.loads(result[4]) if result[4] else []
            }
        return {
            'liked_tags': [], 'disliked_tags': [], 'preferred_providers': [], 
            'preferred_skill_levels': [], 'preferred_categories': []
        }
    
    def update_user_preferences(self, user_profile_hash: str, preferences: Dict):
        """Update enhanced user preferences"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO user_preferences 
            (user_profile_hash, liked_tags, disliked_tags, preferred_providers, 
             preferred_skill_levels, preferred_categories, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            user_profile_hash,
            json.dumps(preferences['liked_tags']),
            json.dumps(preferences['disliked_tags']),
            json.dumps(preferences['preferred_providers']),
            json.dumps(preferences['preferred_skill_levels']),
            json.dumps(preferences.get('preferred_categories', [])),
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def get_user_stats(self, user_profile_hash: str) -> Dict:
        """Get user statistics for dashboard"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                COUNT(*) as total_feedback,
                COUNT(CASE WHEN feedback = 'like' THEN 1 END) as likes,
                COUNT(CASE WHEN feedback = 'dislike' THEN 1 END) as dislikes
            FROM user_feedback 
            WHERE user_profile_hash = ?
        ''', (user_profile_hash,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                'total_feedback': result[0],
                'likes': result[1],
                'dislikes': result[2]
            }
        return {'total_feedback': 0, 'likes': 0, 'dislikes': 0}

class EnhancedCourseRecommendationSystem:
    def __init__(self):
        self.courses = COURSES
        self.db_manager = DatabaseManager()
        self.categories = list(set(course['category'] for course in COURSES))
        
        # Initialize sentence transformer with error handling
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self._compute_course_embeddings()
            st.success("✅ AI models loaded successfully!")
        except Exception as e:
            st.error(f"❌ Error loading sentence transformer: {e}")
            self.model = None
        
        # Initialize Cohere client
        self.cohere_client = None
        cohere_key = os.getenv("COHERE_API_KEY")
        if cohere_key:
            try:
                self.cohere_client = cohere.Client(cohere_key)
                st.success("✅ Cohere API connected!")
            except Exception as e:
                st.warning(f"⚠️ Cohere initialization error: {e}")
        else:
            st.warning("⚠️ COHERE_API_KEY not found in environment variables")
        
        # Enhanced learning paths with more detailed progression
        self.learning_paths = {
            "data-scientist": {
                "title": "Data Scientist Path",
                "description": "Comprehensive path to become a data scientist",
                "courses": [
                    "Python for Data Science and Machine Learning",
                    "SQL for Data Analysis", 
                    "Data Science with R",
                    "Machine Learning Specialization",
                    "Deep Learning Specialization"
                ],
                "duration": "12-18 months",
                "difficulty": "Beginner to Advanced"
            },
            "ml-engineer": {
                "title": "ML Engineer Path",
                "description": "Technical path focusing on ML systems and deployment",
                "courses": [
                    "Python for Data Science and Machine Learning",
                    "Machine Learning Specialization",
                    "TensorFlow for Deep Learning",
                    "AWS Cloud Practitioner",
                    "DevOps Engineering"
                ],
                "duration": "10-15 months",
                "difficulty": "Intermediate to Advanced"
            },
            "web-developer": {
                "title": "Full Stack Developer Path", 
                "description": "Complete web development journey",
                "courses": [
                    "Full Stack Web Development",
                    "Vue.js Frontend Development",
                    "Python Web Development with Django",
                    "GraphQL API Development",
                    "DevOps Engineering"
                ],
                "duration": "8-12 months",
                "difficulty": "Beginner to Intermediate"
            },
            "mobile-developer": {
                "title": "Mobile Developer Path",
                "description": "Cross-platform mobile development",
                "courses": [
                    "React Native Mobile Development",
                    "Flutter Mobile App Development", 
                    "Android App Development with Kotlin",
                    "iOS Development with Swift"
                ],
                "duration": "8-10 months",
                "difficulty": "Intermediate"
            }
        }
    
    def _compute_course_embeddings(self):
        """Compute optimized embeddings for courses"""
        if not self.model:
            return
            
        course_texts = []
        for course in self.courses:
            # Enhanced text representation with category
            text = (
                f"Title: {course['title']} "
                f"Category: {course['category']} "
                f"Description: {course['description']} "
                f"Level: {course['skill_level']} "
                f"Tags: {' '.join(course['tags'])} "
                f"Provider: {course['provider']}"
            )
            course_texts.append(text)
        
        try:
            embeddings = self.model.encode(course_texts, show_progress_bar=True)
            # L2 normalization for better cosine similarity
            self.course_embeddings = embeddings / np.linalg.norm(embeddings, axis=1)[:, np.newaxis]
        except Exception as e:
            st.error(f"Error computing course embeddings: {e}")
            self.course_embeddings = None
    
    def create_user_profile_embedding(self, background: str, interests: str, goals: str, skills: str = ""):
        """Create enhanced user profile embedding"""
        if not self.model:
            return None
            
        profile_text = f"Background: {background}. Interests: {interests}. Goals: {goals}. Skills: {skills}"
        try:
            embedding = self.model.encode([profile_text])
            return embedding / np.linalg.norm(embedding)  # Normalize
        except Exception as e:
            st.error(f"Error creating user profile embedding: {e}")
            return None
    
    def get_user_profile_hash(self, background: str, interests: str, goals: str, skills: str = "") -> str:
        """Create a hash for user profile"""
        profile_text = f"{background}|{interests}|{goals}|{skills}"
        return str(abs(hash(profile_text)))[:12]  # Shorter hash
    
    def get_recommendations(self, background: str, interests: str, goals: str, skills: str = "", 
                          filters: Dict = None, top_k: int = 5) -> List[Dict]:
        """Enhanced recommendation system with better scoring"""
        
        user_profile_hash = self.get_user_profile_hash(background, interests, goals, skills)
        user_prefs = self.db_manager.get_user_preferences(user_profile_hash)
        
        # Get semantic recommendations
        semantic_recs = self._get_semantic_recommendations(
            background, interests, goals, skills, filters, top_k * 2, user_prefs
        ) if self.model and self.course_embeddings is not None else []
        
        # Get keyword-based recommendations  
        keyword_recs = self._get_enhanced_keyword_recommendations(
            background, interests, goals, skills, filters, top_k * 2, user_prefs
        )
        
        # Combine with intelligent weighting
        combined_recs = self._combine_recommendations(semantic_recs, keyword_recs, user_prefs)
        
        return combined_recs[:top_k]
    
    def _get_semantic_recommendations(self, background: str, interests: str, goals: str, 
                                   skills: str = "", filters: Dict = None, top_k: int = 10,
                                   user_prefs: Dict = None):
        """Enhanced semantic recommendations with preference boosting"""
        profile_text = f"Background: {background} Interests: {interests} Goals: {goals} Skills: {skills}"
        
        try:
            user_embedding = self.create_user_profile_embedding(background, interests, goals, skills)
            if user_embedding is None:
                return []
            
            similarities = cosine_similarity(user_embedding, self.course_embeddings).flatten()
            
            # Apply preference adjustments
            if user_prefs:
                similarities = self._apply_preference_adjustments(similarities, user_prefs)
            
            scored_courses = []
            for course, score in zip(self.courses, similarities):
                course_copy = course.copy()
                course_copy['similarity_score'] = float(score)
                course_copy['recommendation_type'] = 'semantic'
                course_copy['recommendation_reason'] = self._get_recommendation_reason(
                    course, background, interests, goals, user_prefs
                )
                scored_courses.append(course_copy)
            
            # Apply filters
            if filters:
                scored_courses = self._apply_filters(scored_courses, filters)
            
            return sorted(scored_courses, key=lambda x: x['similarity_score'], reverse=True)[:top_k]
            
        except Exception as e:
            st.error(f"Error in semantic recommendation: {e}")
            return []
    
    def _get_enhanced_keyword_recommendations(self, background: str, interests: str, goals: str, 
                                            skills: str = "", filters: Dict = None, top_k: int = 10,
                                            user_prefs: Dict = None) -> List[Dict]:
        """Enhanced keyword-based recommendations with category matching"""
        profile_keywords = self._extract_keywords(f"{background} {interests} {goals} {skills}")
        
        scored_courses = []
        for course in self.courses:
            score = 0
            
            # Enhanced scoring system
            # Tag matching (highest weight)
            for tag in course['tags']:
                if any(keyword.lower() in tag.lower() for keyword in profile_keywords):
                    score += 3
            
            # Category matching
            course_category = course['category'].lower()
            if any(keyword.lower() in course_category for keyword in profile_keywords):
                score += 2
            
            # Title matching
            title_lower = course['title'].lower()
            for keyword in profile_keywords:
                if keyword.lower() in title_lower:
                    score += 2
            
            # Description matching
            desc_lower = course['description'].lower()
            for keyword in profile_keywords:
                if keyword.lower() in desc_lower:
                    score += 1
            
            # Apply user preferences
            if user_prefs:
                # Boost liked tags
                for tag in course['tags']:
                    if tag in user_prefs.get('liked_tags', []):
                        score += 2
                    elif tag in user_prefs.get('disliked_tags', []):
                        score -= 1
                
                # Boost preferred categories
                if course['category'] in user_prefs.get('preferred_categories', []):
                    score += 1
            
            if score > 0:
                course_copy = course.copy()
                course_copy['similarity_score'] = min(score / 10.0, 1.0)  # Normalize to [0,1]
                course_copy['recommendation_type'] = 'keyword'
                course_copy['recommendation_reason'] = self._get_recommendation_reason(
                    course, background, interests, goals, user_prefs
                )
                scored_courses.append(course_copy)
        
        # Apply filters
        if filters:
            scored_courses = self._apply_filters(scored_courses, filters)
        
        return sorted(scored_courses, key=lambda x: x['similarity_score'], reverse=True)[:top_k]
    
    def _combine_recommendations(self, semantic_recs: List[Dict], keyword_recs: List[Dict], 
                               user_prefs: Dict) -> List[Dict]:
        """Intelligently combine semantic and keyword recommendations"""
        combined = {}
        
        # Add semantic recommendations with 0.7 weight
        for course in semantic_recs:
            course_id = course['id']
            combined[course_id] = course.copy()
            combined[course_id]['similarity_score'] *= 0.7
            combined[course_id]['semantic_score'] = course['similarity_score']
            combined[course_id]['keyword_score'] = 0.0
        
        # Add/merge keyword recommendations with 0.3 weight
        for course in keyword_recs:
            course_id = course['id']
            if course_id in combined:
                # Merge scores
                combined[course_id]['similarity_score'] += course['similarity_score'] * 0.3
                combined[course_id]['keyword_score'] = course['similarity_score']
                # Use better recommendation reason
                if len(course['recommendation_reason']) > len(combined[course_id]['recommendation_reason']):
                    combined[course_id]['recommendation_reason'] = course['recommendation_reason']
            else:
                course['similarity_score'] *= 0.3
                course['semantic_score'] = 0.0
                course['keyword_score'] = course['similarity_score'] / 0.3
                combined[course_id] = course
        
        # Convert to list and sort
        final_recs = list(combined.values())
        return sorted(final_recs, key=lambda x: x['similarity_score'], reverse=True)
    
    def _apply_preference_adjustments(self, similarities: np.ndarray, preferences: Dict) -> np.ndarray:
        """Apply user preference adjustments to similarity scores"""
        adjusted = similarities.copy()
        
        for i, course in enumerate(self.courses):
            multiplier = 1.0
            
            # Tag preferences
            for tag in course['tags']:
                if tag in preferences.get('liked_tags', []):
                    multiplier *= 1.3
                elif tag in preferences.get('disliked_tags', []):
                    multiplier *= 0.6
            
            # Provider preferences
            if course['provider'] in preferences.get('preferred_providers', []):
                multiplier *= 1.2
            
            # Skill level preferences
            if course['skill_level'] in preferences.get('preferred_skill_levels', []):
                multiplier *= 1.15
            
            # Category preferences
            if course['category'] in preferences.get('preferred_categories', []):
                multiplier *= 1.1
            
            adjusted[i] *= multiplier
        
        return adjusted
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Enhanced keyword extraction"""
        words = re.findall(r'\b\w+\b', text.lower())
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'can', 'may', 'might', 'must', 'shall', 'this', 'that',
            'these', 'those', 'my', 'your', 'his', 'her', 'its', 'our', 'their', 'me', 'you', 'him', 'her',
            'us', 'them', 'i', 'we', 'he', 'she', 'it', 'they'
        }
        # Include domain-specific keywords
        domain_keywords = {
            'ai', 'ml', 'data', 'web', 'mobile', 'cloud', 'security', 'programming', 'development',
            'science', 'analysis', 'learning', 'intelligence', 'artificial', 'machine', 'deep',
            'neural', 'network', 'algorithm', 'database', 'frontend', 'backend', 'fullstack'
        }
        
        keywords = []
        for word in words:
            if (len(word) > 2 and word not in stop_words) or word in domain_keywords:
                keywords.append(word)
        
        return keywords
    
    def _apply_filters(self, courses: List[Dict], filters: Dict) -> List[Dict]:
        """Enhanced filtering with category support"""
        if not filters:
            return courses
        
        filtered = courses.copy()
        
        if 'providers' in filters and filters['providers']:
            filtered = [c for c in filtered if c['provider'] in filters['providers']]
        
        if 'skill_levels' in filters and filters['skill_levels']:
            filtered = [c for c in filtered if c['skill_level'] in filters['skill_levels']]
        
        if 'categories' in filters and filters['categories']:
            filtered = [c for c in filtered if c['category'] in filters['categories']]
        
        if 'cost' in filters and filters['cost']:
            filtered = [c for c in filtered if c['cost'] in filters['cost']]
        
        if 'max_duration' in filters and filters['max_duration']:
            max_months = filters['max_duration']
            filtered = [c for c in filtered if self._extract_duration_months(c['duration']) <= max_months]
        
        if 'min_rating' in filters and filters['min_rating']:
            filtered = [c for c in filtered if c['rating'] >= filters['min_rating']]
        
        return filtered
    
    def _extract_duration_months(self, duration: str) -> int:
        """Extract duration in months from duration string"""
        months = re.findall(r'(\d+)\s*months?', duration.lower())
        return int(months[0]) if months else 12
    
    def _get_recommendation_reason(self, course: Dict, background: str, interests: str, 
                                 goals: str, user_prefs: Dict = None) -> str:
        """Generate detailed explanation for recommendation"""
        reasons = []
        
        profile_text = f"{background} {interests} {goals}".lower()
        course_tags = [tag.replace('-', ' ') for tag in course['tags']]
        
        # Check for keyword matches
        matching_tags = [tag for tag in course_tags if tag.lower() in profile_text]
        if matching_tags:
            reasons.append(f"Matches your interests: {', '.join(matching_tags[:2])}")
        
        # Check category alignment
        if course['category'].lower() in profile_text:
            reasons.append(f"Aligns with your {course['category'].lower()} focus")
        
        # Check skill level appropriateness
        if course['skill_level'] == 'beginner' and any(word in profile_text for word in ['beginner', 'new', 'start']):
            reasons.append("Perfect for beginners")
        elif course['skill_level'] == 'advanced' and any(word in profile_text for word in ['advanced', 'expert', 'senior', 'experienced']):
            reasons.append("Matches your advanced level")
        
        # Check user preferences
        if user_prefs:
            liked_tags = [tag for tag in course['tags'] if tag in user_prefs.get('liked_tags', [])]
            if liked_tags:
                reasons.append(f"Based on your preferences: {', '.join(liked_tags[:2])}")
        
        # Provider reputation
        if course['rating'] >= 4.7:
            reasons.append("Highly rated course")
        
        if not reasons:
            reasons.append(f"Strong relevance match for your {course['category'].lower()} goals")
        
        return ". ".join(reasons[:3])
    
    def record_feedback(self, background: str, interests: str, goals: str, skills: str, 
                       course_id: int, feedback: str):
        """Enhanced feedback recording with category learning"""
        user_profile_hash = self.get_user_profile_hash(background, interests, goals, skills)
        
        # Save feedback
        self.db_manager.save_feedback(user_profile_hash, course_id, feedback)
        
        # Update preferences
        course = next(c for c in self.courses if c['id'] == course_id)
        preferences = self.db_manager.get_user_preferences(user_profile_hash)
        
        if feedback == 'like':
            # Add tags
            for tag in course['tags']:
                if tag not in preferences['liked_tags']:
                    preferences['liked_tags'].append(tag)
                # Remove from disliked if present
                if tag in preferences['disliked_tags']:
                    preferences['disliked_tags'].remove(tag)
            
            # Add provider
            if course['provider'] not in preferences['preferred_providers']:
                preferences['preferred_providers'].append(course['provider'])
            
            # Add skill level
            if course['skill_level'] not in preferences['preferred_skill_levels']:
                preferences['preferred_skill_levels'].append(course['skill_level'])
            
            # Add category
            if course['category'] not in preferences['preferred_categories']:
                preferences['preferred_categories'].append(course['category'])
        
        elif feedback == 'dislike':
            # Add to disliked tags
            for tag in course['tags']:
                if tag not in preferences['disliked_tags']:
                    preferences['disliked_tags'].append(tag)
                # Remove from liked if present
                if tag in preferences['liked_tags']:
                    preferences['liked_tags'].remove(tag)
        
        # Limit list sizes to prevent overwhelming preferences
        preferences['liked_tags'] = preferences['liked_tags'][-20:]
        preferences['disliked_tags'] = preferences['disliked_tags'][-10:]
        preferences['preferred_providers'] = preferences['preferred_providers'][-10:]
        
        self.db_manager.update_user_preferences(user_profile_hash, preferences)
    
    def get_learning_path(self, career_goal: str) -> Dict:
        """Get enhanced learning path information"""
        goal_lower = career_goal.lower()
        
        for path_key, path_info in self.learning_paths.items():
            if (path_key.replace('-', ' ') in goal_lower or 
                any(word in goal_lower for word in path_key.split('-'))):
                return path_info
        
        return None
    
    def answer_learning_question(self, question: str, background: str = "", interests: str = "", 
                               goals: str = "", skills: str = "") -> str:
        """Enhanced Q&A with better context"""
        if not self.cohere_client:
            return "Cohere API not available. Please configure COHERE_API_KEY environment variable."
        
        # Create comprehensive context
        context_parts = []
        
        if background or interests or goals or skills:
            context_parts.append(f"User Profile - Background: {background}, Interests: {interests}, Goals: {goals}, Skills: {skills}")
        
        # Add learning paths
        context_parts.append("Available Learning Paths:")
        for path_key, path_info in self.learning_paths.items():
            context_parts.append(f"- {path_info['title']}: {path_info['description']} (Duration: {path_info['duration']})")
        
        # Add course categories
        context_parts.append(f"Available Course Categories: {', '.join(self.categories)}")
        
        # Add sample courses by category
        context_parts.append("Sample Courses by Category:")
        for category in self.categories[:5]:  # Limit for context size
            category_courses = [c for c in self.courses if c['category'] == category][:2]
            if category_courses:
                course_titles = [c['title'] for c in category_courses]
                context_parts.append(f"- {category}: {', '.join(course_titles)}")
        
        context = "\n".join(context_parts)
        
        # Enhanced prompt
        prompt = f"""You are an expert education and career advisor with deep knowledge of technology learning paths and course recommendations. 

Context about available courses and user:
{context}

User Question: {question}

Please provide a helpful, structured response that:
1. Directly addresses the user's question
2. Provides specific, actionable recommendations
3. Considers the user's background and goals if provided
4. Suggests concrete next steps
5. Explains the reasoning behind recommendations
6. Mentions specific courses from the available catalog when relevant

Keep your response practical, encouraging, and focused on helping the user make progress toward their learning goals.

Response:"""

        try:
            response = self.cohere_client.generate(
                model='command',
                prompt=prompt,
                max_tokens=600,
                temperature=0.7,
                k=0,
                p=0.9
            )
            
            return response.generations[0].text.strip()
            
        except Exception as e:
            return f"Error generating response: {str(e)}. Please verify your Cohere API key is valid."

# Enhanced Streamlit UI
def create_course_card(course: Dict, index: int) -> None:
    """Create an enhanced course card with better visual design"""
    
    # Color coding by category
    category_colors = {
        "Data Science": "#FF6B6B", "Programming": "#4ECDC4", "Web Development": "#45B7D1",
        "Mobile Development": "#96CEB4", "Cloud Computing": "#FFEAA7", "DevOps": "#DDA0DD",
        "Cybersecurity": "#FF7675", "Design": "#A8E6CF", "Business Intelligence": "#FFB347",
        "Blockchain": "#87CEEB", "Game Development": "#F0A3FF", "AI Ethics": "#C7CEEA"
    }
    
    color = category_colors.get(course['category'], "#E0E0E0")
    
    with st.container():
        st.markdown(f"""
        <div style="
            border-left: 5px solid {color};
            padding: 15px;
            margin: 10px 0;
            background-color: #f8f9fa;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        ">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h4 style="color: #2c3e50; margin: 0;">{course['title']}</h4>
                <span style="background-color: {color}; color: white; padding: 4px 8px; border-radius: 15px; font-size: 0.8em;">
                    {course['category']}
                </span>
            </div>
            <p style="color: #7f8c8d; margin: 5px 0 0 0; font-size: 0.9em;">
                {course['provider']} • {course['skill_level'].title()} • {course['duration']} • {'Free' if course['cost'] == 'free' else 'Paid'}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.write(course['description'])
            
            # Enhanced recommendation reason
            if 'recommendation_reason' in course:
                st.info(f"💡 **Why recommended:** {course['recommendation_reason']}")
        
        with col2:
            # Rating display
            stars = "⭐" * int(course['rating'])
            st.markdown(f"**Rating:** {stars} {course['rating']}/5")
            
            # Match score with color coding
            score = course['similarity_score']
            if score > 0.7:
                score_color = "green"
            elif score > 0.4:
                score_color = "orange" 
            else:
                score_color = "red"
            
            st.markdown(f"**Match Score:** <span style='color: {score_color};'>{score:.3f}</span>", 
                       unsafe_allow_html=True)
            
            # Show recommendation type breakdown if available
            if 'semantic_score' in course and 'keyword_score' in course:
                st.markdown("**Score Breakdown:**")
                st.markdown(f"- Semantic: {course['semantic_score']:.3f}")
                st.markdown(f"- Keyword: {course['keyword_score']:.3f}")
        
        # Tags
        st.markdown("**Skills you'll learn:**")
        tags_html = " ".join([
            f"<span style='background-color: #e3f2fd; color: #1565c0; padding: 3px 8px; border-radius: 12px; margin: 2px; display: inline-block; font-size: 0.8em;'>{tag.replace('-', ' ').title()}</span>" 
            for tag in course['tags']
        ])
        st.markdown(tags_html, unsafe_allow_html=True)
        
        st.markdown("---")

def create_dashboard_metrics(recommender, user_profile_hash: str) -> None:
    """Create user dashboard with statistics"""
    stats = recommender.db_manager.get_user_stats(user_profile_hash)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Feedback", stats['total_feedback'])
    with col2:
        st.metric("Courses Liked", stats['likes'])
    with col3:
        st.metric("Courses Disliked", stats['dislikes'])
    with col4:
        if stats['total_feedback'] > 0:
            satisfaction = (stats['likes'] / stats['total_feedback']) * 100
            st.metric("Satisfaction", f"{satisfaction:.0f}%")
        else:
            st.metric("Satisfaction", "N/A")

def create_learning_path_viz(path_info: Dict) -> None:
    """Create learning path visualization"""
    if not path_info:
        return
    
    st.subheader(f"🗺️ {path_info['title']}")
    st.write(path_info['description'])
    
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Duration:** {path_info['duration']}")
    with col2:
        st.info(f"**Difficulty:** {path_info['difficulty']}")
    
    st.write("**Recommended Course Sequence:**")
    for i, course_title in enumerate(path_info['courses'], 1):
        st.write(f"{i}. {course_title}")

def main():
    st.set_page_config(
        page_title="AI Course Recommendation System", 
        page_icon="🎓", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
        .main-header {
            text-align: center;
            padding: 2rem 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
            margin-bottom: 2rem;
        }
        .stSelectbox > div > div {
            background-color: #f0f2f6;
        }
        .metric-card {
            background: white;
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="main-header">
        <h1>🎓 AI Course Recommendation System</h1>
        <p>Get personalized course recommendations powered by advanced AI</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize system
    if 'recommender' not in st.session_state:
        with st.spinner("🚀 Initializing AI recommendation system..."):
            st.session_state.recommender = EnhancedCourseRecommendationSystem()
        st.session_state.current_recommendations = []
        st.session_state.user_profile = {"background": "", "interests": "", "goals": "", "skills": ""}
        st.session_state.show_advanced_filters = False
    
    # Enhanced sidebar
    with st.sidebar:
        st.markdown("### 👤 Your Learning Profile")
        
        # User profile inputs with better UX
        with st.expander("📝 Profile Details", expanded=True):
            background = st.text_area(
                "Current Background", 
                value=st.session_state.user_profile.get("background", ""),
                placeholder="e.g., Final-year Computer Science student with internship experience in web development",
                help="Describe your current educational/professional background",
                height=80
            )
            
            interests = st.text_area(
                "Areas of Interest", 
                value=st.session_state.user_profile.get("interests", ""),
                placeholder="e.g., Artificial Intelligence, Machine Learning, Data Science, Cloud Computing",
                help="What topics or technologies interest you most?",
                height=60
            )
            
            goals = st.text_area(
                "Career Goals", 
                value=st.session_state.user_profile.get("goals", ""),
                placeholder="e.g., Become a Machine Learning Engineer at a tech company within 2 years",
                help="What are your short and long-term career aspirations?",
                height=60
            )
            
            skills = st.text_input(
                "Current Skills (Optional)", 
                value=st.session_state.user_profile.get("skills", ""),
                placeholder="e.g., Python, SQL, Basic Statistics, HTML/CSS",
                help="List your current technical skills and experience level"
            )
        
        st.markdown("---")
        
        # Enhanced filters section
        st.markdown("### 🔍 Smart Filters")
        
        # Toggle for advanced filters
        st.session_state.show_advanced_filters = st.checkbox("Show Advanced Filters", 
                                                            value=st.session_state.show_advanced_filters)
        
        # Basic filters
        categories = st.multiselect(
            "Course Categories",
            options=st.session_state.recommender.categories,
            help="Filter by course categories"
        )
        
        skill_levels = st.multiselect(
            "Skill Levels",
            options=["beginner", "intermediate", "advanced"],
            help="Select your preferred difficulty levels"
        )
        
        cost_filter = st.multiselect(
            "Cost Preference",
            options=["free", "paid"],
            default=["free", "paid"],
            help="Filter by course cost"
        )
        
        # Advanced filters (collapsible)
        if st.session_state.show_advanced_filters:
            with st.expander("🔧 Advanced Filters", expanded=False):
                providers = st.multiselect(
                    "Preferred Providers",
                    options=sorted(list(set(course["provider"] for course in COURSES))),
                    help="Filter by course providers"
                )
                
                max_duration = st.slider(
                    "Max Duration (months)",
                    min_value=1,
                    max_value=12,
                    value=12,
                    help="Maximum acceptable course duration"
                )
                
                min_rating = st.slider(
                    "Minimum Rating",
                    min_value=3.0,
                    max_value=5.0,
                    value=4.0,
                    step=0.1,
                    help="Minimum course rating"
                )
        else:
            providers = []
            max_duration = 12
            min_rating = 4.0
        
        # Number of recommendations
        num_recommendations = st.slider(
            "Number of Recommendations",
            min_value=3,
            max_value=15,
            value=8,
            help="How many courses to recommend"
        )
        
        # Enhanced recommendation button
        recommend_button = st.button(
            "🔍 Get AI Recommendations", 
            type="primary", 
            use_container_width=True,
            help="Generate personalized course recommendations"
        )
        
        if recommend_button:
            if background or interests or goals:
                # Store user profile
                st.session_state.user_profile = {
                    "background": background,
                    "interests": interests,
                    "goals": goals,
                    "skills": skills
                }
                
                # Create filters dict
                filters = {}
                if categories:
                    filters["categories"] = categories
                if providers:
                    filters["providers"] = providers
                if skill_levels:
                    filters["skill_levels"] = skill_levels
                if cost_filter:
                    filters["cost"] = cost_filter
                if max_duration < 12:
                    filters["max_duration"] = max_duration
                if min_rating > 4.0:
                    filters["min_rating"] = min_rating
                
                # Get recommendations
                with st.spinner("🤖 AI is analyzing your profile and finding the best matches..."):
                    recommendations = st.session_state.recommender.get_recommendations(
                        background, interests, goals, skills, filters, num_recommendations
                    )
                st.session_state.current_recommendations = recommendations
                st.success(f"Found {len(recommendations)} personalized recommendations!")
                st.rerun()
            else:
                st.error("Please fill in at least one profile field to get recommendations.")
        
        # User dashboard
        if any(st.session_state.user_profile.values()):
            st.markdown("---")
            st.markdown("### 📊 Your Dashboard")
            user_hash = st.session_state.recommender.get_user_profile_hash(
                background, interests, goals, skills
            )
            create_dashboard_metrics(st.session_state.recommender, user_hash)
    
    # Main content area with improved layout
    if st.session_state.current_recommendations:
        # Show recommendations
        st.header("🎯 Your Personalized Recommendations")
        
        # Summary stats
        recs = st.session_state.current_recommendations
        avg_score = sum(r['similarity_score'] for r in recs) / len(recs)
        free_count = sum(1 for r in recs if r['cost'] == 'free')
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Recommendations", len(recs))
        with col2:
            st.metric("Avg Match Score", f"{avg_score:.3f}")
        with col3:
            st.metric("Free Courses", free_count)
        with col4:
            st.metric("Categories", len(set(r['category'] for r in recs)))
        
        st.markdown("---")
        
        # Display recommendations with enhanced cards
        for i, course in enumerate(recs):
            col1, col2 = st.columns([4, 1])
            
            with col1:
                create_course_card(course, i)
            
            with col2:
                st.markdown("**Actions**")
                
                like_key = f"like_{course['id']}_{i}"
                dislike_key = f"dislike_{course['id']}_{i}"
                
                if st.button("👍 Like", key=like_key, use_container_width=True, type="secondary"):
                    st.session_state.recommender.record_feedback(
                        background, interests, goals, skills, course['id'], 'like'
                    )
                    st.success("Feedback recorded! Future recommendations will be improved.")
                    time.sleep(1)
                    st.rerun()
                
                if st.button("👎 Dislike", key=dislike_key, use_container_width=True):
                    st.session_state.recommender.record_feedback(
                        background, interests, goals, skills, course['id'], 'dislike'
                    )
                    st.success("Feedback recorded! We'll avoid similar courses.")
                    time.sleep(1)
                    st.rerun()
                
                # Additional info
                st.markdown(f"**ID:** {course['id']}")
                if 'recommendation_type' in course:
                    st.markdown(f"**Type:** {course['recommendation_type'].title()}")
    
    else:
        # Welcome screen with sample content
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header("🚀 Welcome to AI-Powered Learning!")
            
            st.markdown("""
            **Get started by:**
            1. 📝 Fill out your learning profile in the sidebar
            2. 🎯 Set your preferences and filters
            3. 🔍 Click "Get AI Recommendations" to discover courses
            4. 👍👎 Provide feedback to improve future recommendations
            """)
            
            st.subheader("🌟 Popular Courses")
            
            # Show top-rated courses as examples
            popular_courses = sorted(COURSES, key=lambda x: x['rating'], reverse=True)[:3]
            
            for course in popular_courses:
                with st.container():
                    col_a, col_b = st.columns([3, 1])
                    with col_a:
                        st.write(f"**{course['title']}**")
                        st.write(f"*{course['provider']} • {course['category']} • {course['skill_level'].title()}*")
                        st.write(course['description'][:120] + "...")
                    with col_b:
                        st.write(f"⭐ {course['rating']}/5")
                        st.write(f"🆓" if course['cost'] == 'free' else "💰")
                    st.markdown("---")
        
        with col2:
            # Learning Path Assistant
            st.header("🤖 Learning Assistant")
            
            # Quick questions section
            st.subheader("💡 Quick Questions")
            quick_questions = [
                "Should I learn Python or R for data science?",
                "What's the best path to become an ML engineer?", 
                "How do I transition to web development?",
                "What skills do I need for cybersecurity?",
                "Which cloud platform should I start with?"
            ]
            
            for i, qq in enumerate(quick_questions):
                if st.button(f"❓ {qq}", key=f"quick_{i}", use_container_width=True):
                    with st.spinner("🤔 Thinking..."):
                        answer = st.session_state.recommender.answer_learning_question(
                            qq, background, interests, goals, skills
                        )
                    
                    with st.expander("💬 Assistant's Answer", expanded=True):
                        st.write(f"**Q:** {qq}")
                        st.write(f"**A:** {answer}")
            
            # Custom question section
            st.subheader("🗨️ Ask Anything")
            question = st.text_area(
                "Your question:",
                placeholder="e.g., What programming language should I learn first for my goals?",
                height=80,
                key="custom_question"
            )
            
            if st.button("🔮 Ask Assistant", use_container_width=True):
                if question.strip():
                    with st.spinner("🧠 Generating personalized response..."):
                        answer = st.session_state.recommender.answer_learning_question(
                            question, background, interests, goals, skills
                        )
                    
                    with st.expander("🎯 Personalized Answer", expanded=True):
                        st.write(f"**Your Question:** {question}")
                        st.write(f"**Assistant's Answer:** {answer}")
                else:
                    st.error("Please enter a question.")
            
            # Learning path suggestions
            if goals:
                st.subheader("🗺️ Suggested Learning Path")
                learning_path = st.session_state.recommender.get_learning_path(goals)
                if learning_path:
                    create_learning_path_viz(learning_path)
                else:
                    st.info("💡 Try being more specific about your career goals to see suggested learning paths (e.g., 'data scientist', 'web developer', 'ML engineer')")
    
    # Additional sections for enhanced functionality
    st.markdown("---")
    
    # Course statistics and insights
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Course Database Statistics")
        
        # Create category distribution chart
        category_counts = {}
        for course in COURSES:
            category = course['category']
            category_counts[category] = category_counts.get(category, 0) + 1
        
        if len(category_counts) > 0:
            fig = px.bar(
                x=list(category_counts.keys()),
                y=list(category_counts.values()),
                title="Courses by Category",
                labels={'x': 'Category', 'y': 'Number of Courses'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("⭐ Rating Distribution")
        
        # Rating distribution
        ratings = [course['rating'] for course in COURSES]
        fig = px.histogram(
            x=ratings,
            nbins=20,
            title="Course Ratings Distribution",
            labels={'x': 'Rating', 'y': 'Count'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Footer with enhanced statistics
    st.markdown("---")
    st.subheader("📈 Platform Statistics")
    
    footer_cols = st.columns(6)
    
    with footer_cols[0]:
        st.metric("Total Courses", len(COURSES))
    
    with footer_cols[1]:
        free_courses = len([c for c in COURSES if c['cost'] == 'free'])
        st.metric("Free Courses", free_courses)
    
    with footer_cols[2]:
        providers = len(set(course['provider'] for course in COURSES))
        st.metric("Providers", providers)
    
    with footer_cols[3]:
        categories_count = len(set(course['category'] for course in COURSES))
        st.metric("Categories", categories_count)
    
    with footer_cols[4]:
        avg_rating = sum(course['rating'] for course in COURSES) / len(COURSES)
        st.metric("Avg Rating", f"{avg_rating:.1f}⭐")
    
    with footer_cols[5]:
        avg_duration = sum(
            st.session_state.recommender._extract_duration_months(course['duration']) 
            for course in COURSES
        ) / len(COURSES)
        st.metric("Avg Duration", f"{avg_duration:.1f} months")
    
    # Technology stack info
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 20px; background-color: #f8f9fa; border-radius: 10px;">
        <p><strong>🛠️ Built with:</strong> Streamlit • SentenceTransformers • Cohere AI • SQLite • Plotly</p>
        <p><strong>🧠 AI Features:</strong> Semantic Search • Natural Language Q&A • Personalized Learning • Feedback Loop</p>
        <p><em>This system learns from your feedback to provide increasingly personalized recommendations</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    import time
    main()