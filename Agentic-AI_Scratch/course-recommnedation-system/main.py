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
from huggingface_hub import hf_hub_download  # Changed this import
from huggingface_hub import list_repo_files

load_dotenv()

# Course Dataset (35+ courses as required)
COURSES = [
    {"id": 1, "title": "Machine Learning Specialization", "provider": "Coursera", "description": "Learn machine learning fundamentals including supervised learning, unsupervised learning, and neural networks. Covers linear regression, logistic regression, decision trees, and deep learning basics.", "skill_level": "intermediate", "tags": ["machine-learning", "python", "data-science", "neural-networks"], "duration": "6 months", "rating": 4.8, "cost": "paid"},
    {"id": 2, "title": "Python for Data Science and Machine Learning", "provider": "Udemy", "description": "Complete Python programming course for data science. Learn pandas, numpy, matplotlib, seaborn, scikit-learn, and machine learning algorithms.", "skill_level": "beginner", "tags": ["python", "data-science", "pandas", "numpy", "beginner"], "duration": "3 months", "rating": 4.6, "cost": "paid"},
    {"id": 3, "title": "Deep Learning Specialization", "provider": "Coursera", "description": "Advanced deep learning course covering neural networks, CNNs, RNNs, LSTMs, and transformer models. Includes practical projects with TensorFlow.", "skill_level": "advanced", "tags": ["deep-learning", "tensorflow", "cnn", "rnn", "advanced"], "duration": "4 months", "rating": 4.9, "cost": "paid"},
    {"id": 4, "title": "Data Structures and Algorithms", "provider": "edX", "description": "Comprehensive course on data structures and algorithms. Covers arrays, linked lists, trees, graphs, sorting, searching, and dynamic programming.", "skill_level": "intermediate", "tags": ["algorithms", "data-structures", "programming", "problem-solving"], "duration": "5 months", "rating": 4.7, "cost": "free"},
    {"id": 5, "title": "Full Stack Web Development", "provider": "Udemy", "description": "Learn full stack web development with HTML, CSS, JavaScript, React, Node.js, Express, and MongoDB. Build complete web applications.", "skill_level": "beginner", "tags": ["web-development", "javascript", "react", "nodejs", "fullstack"], "duration": "6 months", "rating": 4.5, "cost": "paid"},
    {"id": 6, "title": "AWS Cloud Practitioner", "provider": "AWS", "description": "Introduction to Amazon Web Services cloud computing. Learn AWS core services, security, pricing, and cloud architecture fundamentals.", "skill_level": "beginner", "tags": ["cloud", "aws", "devops", "infrastructure", "beginner"], "duration": "2 months", "rating": 4.4, "cost": "free"},
    {"id": 7, "title": "Natural Language Processing", "provider": "Coursera", "description": "Learn NLP techniques including text preprocessing, sentiment analysis, named entity recognition, and language models. Uses Python and NLTK.", "skill_level": "intermediate", "tags": ["nlp", "text-processing", "python", "nltk", "linguistics"], "duration": "4 months", "rating": 4.6, "cost": "paid"},
    {"id": 8, "title": "Computer Vision Fundamentals", "provider": "edX", "description": "Introduction to computer vision concepts including image processing, feature detection, object recognition, and deep learning for vision.", "skill_level": "intermediate", "tags": ["computer-vision", "image-processing", "opencv", "deep-learning"], "duration": "3 months", "rating": 4.5, "cost": "free"},
    {"id": 9, "title": "SQL for Data Analysis", "provider": "Udemy", "description": "Master SQL for data analysis. Learn queries, joins, aggregations, window functions, and database design for data science applications.", "skill_level": "beginner", "tags": ["sql", "database", "data-analysis", "queries", "beginner"], "duration": "2 months", "rating": 4.7, "cost": "paid"},
    {"id": 10, "title": "DevOps Engineering", "provider": "Coursera", "description": "Learn DevOps practices including CI/CD, containerization with Docker, Kubernetes, infrastructure as code, and monitoring.", "skill_level": "intermediate", "tags": ["devops", "docker", "kubernetes", "ci-cd", "automation"], "duration": "5 months", "rating": 4.6, "cost": "paid"},
    {"id": 11, "title": "React Native Mobile Development", "provider": "Udemy", "description": "Build iOS and Android apps using React Native. Learn navigation, state management, API integration, and app deployment.", "skill_level": "intermediate", "tags": ["react-native", "mobile-development", "javascript", "ios", "android"], "duration": "4 months", "rating": 4.3, "cost": "paid"},
    {"id": 12, "title": "Cybersecurity Fundamentals", "provider": "Coursera", "description": "Introduction to cybersecurity concepts including network security, cryptography, risk assessment, and security frameworks.", "skill_level": "beginner", "tags": ["cybersecurity", "networking", "cryptography", "security", "beginner"], "duration": "3 months", "rating": 4.5, "cost": "free"},
    {"id": 13, "title": "Blockchain Development", "provider": "Udemy", "description": "Learn blockchain technology and smart contract development using Ethereum, Solidity, and Web3.js.", "skill_level": "intermediate", "tags": ["blockchain", "ethereum", "solidity", "web3", "cryptocurrency"], "duration": "5 months", "rating": 4.4, "cost": "paid"},
    {"id": 14, "title": "UI/UX Design Masterclass", "provider": "Coursera", "description": "Comprehensive UI/UX design course covering user research, wireframing, prototyping, and design systems using Figma.", "skill_level": "beginner", "tags": ["ui-ux", "design", "figma", "prototyping", "user-research"], "duration": "4 months", "rating": 4.7, "cost": "paid"},
    {"id": 15, "title": "Android App Development with Kotlin", "provider": "Google", "description": "Learn Android development using Kotlin. Covers activities, fragments, databases, API integration, and Material Design.", "skill_level": "intermediate", "tags": ["android", "kotlin", "mobile-development", "material-design"], "duration": "6 months", "rating": 4.6, "cost": "free"},
    {"id": 16, "title": "iOS Development with Swift", "provider": "Apple", "description": "Build iOS applications using Swift and SwiftUI. Learn iOS frameworks, app architecture, and App Store deployment.", "skill_level": "intermediate", "tags": ["ios", "swift", "swiftui", "mobile-development", "xcode"], "duration": "6 months", "rating": 4.8, "cost": "free"},
    {"id": 17, "title": "Google Cloud Platform Fundamentals", "provider": "Google", "description": "Introduction to GCP services including Compute Engine, Cloud Storage, BigQuery, and Kubernetes Engine.", "skill_level": "beginner", "tags": ["gcp", "cloud", "kubernetes", "bigquery", "infrastructure"], "duration": "3 months", "rating": 4.5, "cost": "free"},
    {"id": 18, "title": "Data Engineering with Apache Spark", "provider": "edX", "description": "Learn big data processing with Apache Spark. Covers RDDs, DataFrames, Spark SQL, and streaming data processing.", "skill_level": "advanced", "tags": ["spark", "big-data", "data-engineering", "scala", "streaming"], "duration": "4 months", "rating": 4.7, "cost": "free"},
    {"id": 19, "title": "Digital Marketing Analytics", "provider": "Coursera", "description": "Learn digital marketing analytics using Google Analytics, social media metrics, and marketing attribution models.", "skill_level": "beginner", "tags": ["marketing", "analytics", "google-analytics", "social-media", "business"], "duration": "3 months", "rating": 4.4, "cost": "paid"},
    {"id": 20, "title": "Java Programming Masterclass", "provider": "Udemy", "description": "Complete Java programming course covering OOP, collections, multithreading, and enterprise Java development.", "skill_level": "beginner", "tags": ["java", "programming", "oop", "multithreading", "enterprise"], "duration": "8 months", "rating": 4.6, "cost": "paid"},
    {"id": 21, "title": "Game Development with Unity", "provider": "Unity", "description": "Learn game development using Unity engine. Covers 2D/3D games, physics, animations, and mobile game development.", "skill_level": "intermediate", "tags": ["unity", "game-development", "c-sharp", "3d", "mobile-games"], "duration": "5 months", "rating": 4.5, "cost": "free"},
    {"id": 22, "title": "Kubernetes Administration", "provider": "Linux Foundation", "description": "Learn Kubernetes container orchestration. Covers cluster management, networking, storage, and security.", "skill_level": "advanced", "tags": ["kubernetes", "containers", "devops", "orchestration", "docker"], "duration": "4 months", "rating": 4.8, "cost": "paid"},
    {"id": 23, "title": "Python Web Development with Django", "provider": "Udemy", "description": "Build web applications using Django framework. Covers models, views, templates, authentication, and deployment.", "skill_level": "intermediate", "tags": ["django", "python", "web-development", "backend", "mvc"], "duration": "5 months", "rating": 4.7, "cost": "paid"},
    {"id": 24, "title": "Business Intelligence with Power BI", "provider": "Microsoft", "description": "Learn business intelligence and data visualization using Microsoft Power BI. Create dashboards and reports.", "skill_level": "beginner", "tags": ["power-bi", "business-intelligence", "data-visualization", "microsoft", "dashboards"], "duration": "3 months", "rating": 4.4, "cost": "free"},
    {"id": 25, "title": "Artificial Intelligence Ethics", "provider": "edX", "description": "Explore ethical considerations in AI development including bias, fairness, transparency, and responsible AI practices.", "skill_level": "intermediate", "tags": ["ai-ethics", "responsible-ai", "bias", "fairness", "philosophy"], "duration": "2 months", "rating": 4.6, "cost": "free"},
    {"id": 26, "title": "Salesforce Administration", "provider": "Salesforce", "description": "Learn Salesforce platform administration including user management, workflows, reports, and customization.", "skill_level": "beginner", "tags": ["salesforce", "crm", "administration", "workflows", "business"], "duration": "4 months", "rating": 4.5, "cost": "free"},
    {"id": 27, "title": "C++ Programming for Competitive Programming", "provider": "Coursera", "description": "Master C++ for competitive programming. Covers STL, algorithms, data structures, and optimization techniques.", "skill_level": "advanced", "tags": ["cpp", "competitive-programming", "algorithms", "stl", "optimization"], "duration": "6 months", "rating": 4.8, "cost": "paid"},
    {"id": 28, "title": "Tableau Data Visualization", "provider": "Tableau", "description": "Create interactive data visualizations and dashboards using Tableau. Learn advanced charting and storytelling with data.", "skill_level": "intermediate", "tags": ["tableau", "data-visualization", "dashboards", "business-intelligence", "analytics"], "duration": "3 months", "rating": 4.6, "cost": "paid"},
    {"id": 29, "title": "Microsoft Azure Fundamentals", "provider": "Microsoft", "description": "Introduction to Microsoft Azure cloud services including virtual machines, storage, networking, and security.", "skill_level": "beginner", "tags": ["azure", "cloud", "microsoft", "virtual-machines", "security"], "duration": "2 months", "rating": 4.5, "cost": "free"},
    {"id": 30, "title": "Network Security and Penetration Testing", "provider": "Udemy", "description": "Learn ethical hacking and penetration testing techniques. Covers vulnerability assessment and security tools.", "skill_level": "advanced", "tags": ["penetration-testing", "ethical-hacking", "network-security", "vulnerability", "kali-linux"], "duration": "6 months", "rating": 4.7, "cost": "paid"},
    {"id": 31, "title": "MongoDB Database Development", "provider": "MongoDB", "description": "Learn NoSQL database development with MongoDB. Covers document modeling, queries, indexing, and aggregation.", "skill_level": "intermediate", "tags": ["mongodb", "nosql", "database", "aggregation", "indexing"], "duration": "3 months", "rating": 4.4, "cost": "free"},
    {"id": 32, "title": "Rust Programming Language", "provider": "Rust Foundation", "description": "Learn systems programming with Rust. Covers memory safety, concurrency, and performance optimization.", "skill_level": "advanced", "tags": ["rust", "systems-programming", "memory-safety", "concurrency", "performance"], "duration": "5 months", "rating": 4.8, "cost": "free"},
    {"id": 33, "title": "Vue.js Frontend Development", "provider": "Vue", "description": "Build modern web applications using Vue.js framework. Covers components, routing, state management, and testing.", "skill_level": "intermediate", "tags": ["vuejs", "frontend", "javascript", "spa", "components"], "duration": "4 months", "rating": 4.6, "cost": "free"},
    {"id": 34, "title": "TensorFlow for Deep Learning", "provider": "TensorFlow", "description": "Deep learning with TensorFlow. Covers neural networks, CNN, RNN, transfer learning, and model deployment.", "skill_level": "advanced", "tags": ["tensorflow", "deep-learning", "neural-networks", "cnn", "transfer-learning"], "duration": "5 months", "rating": 4.9, "cost": "free"},
    {"id": 35, "title": "Go Programming Language", "provider": "Google", "description": "Learn Go programming for backend development and microservices. Covers concurrency, web services, and cloud development.", "skill_level": "intermediate", "tags": ["golang", "backend", "microservices", "concurrency", "cloud"], "duration": "4 months", "rating": 4.7, "cost": "free"},
    {"id": 36, "title": "Flutter Mobile App Development", "provider": "Google", "description": "Build cross-platform mobile apps with Flutter and Dart. Learn widgets, state management, and app deployment.", "skill_level": "intermediate", "tags": ["flutter", "dart", "mobile-development", "cross-platform", "widgets"], "duration": "5 months", "rating": 4.5, "cost": "free"},
    {"id": 37, "title": "Data Science with R", "provider": "Coursera", "description": "Learn data science using R programming. Covers statistics, data visualization with ggplot2, and machine learning.", "skill_level": "intermediate", "tags": ["r-programming", "data-science", "statistics", "ggplot2", "visualization"], "duration": "4 months", "rating": 4.6, "cost": "paid"},
    {"id": 38, "title": "Microservices Architecture", "provider": "Udemy", "description": "Design and build microservices-based applications. Learn service decomposition, API design, and distributed systems.", "skill_level": "advanced", "tags": ["microservices", "architecture", "api-design", "distributed-systems", "scalability"], "duration": "6 months", "rating": 4.7, "cost": "paid"},
    {"id": 39, "title": "GraphQL API Development", "provider": "Apollo", "description": "Build modern APIs with GraphQL. Learn schema design, resolvers, subscriptions, and client integration.", "skill_level": "intermediate", "tags": ["graphql", "api", "schema-design", "resolvers", "apollo"], "duration": "3 months", "rating": 4.4, "cost": "free"},
    {"id": 40, "title": "Quantum Computing Fundamentals", "provider": "IBM", "description": "Introduction to quantum computing concepts, quantum algorithms, and programming with Qiskit.", "skill_level": "advanced", "tags": ["quantum-computing", "qiskit", "quantum-algorithms", "physics", "emerging-tech"], "duration": "4 months", "rating": 4.8, "cost": "free"}
]

class DatabaseManager:
    """Handles SQLite database operations for persistent storage"""
    
    def __init__(self, db_path: str = "course_recommendations.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # User feedback table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_profile_hash TEXT,
                course_id INTEGER,
                feedback TEXT,
                timestamp TEXT
            )
        ''')
        
        # User preferences table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_preferences (
                user_profile_hash TEXT PRIMARY KEY,
                liked_tags TEXT,
                disliked_tags TEXT,
                preferred_providers TEXT,
                preferred_skill_levels TEXT,
                last_updated TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_feedback(self, user_profile_hash: str, course_id: int, feedback: str):
        """Save user feedback to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO user_feedback (user_profile_hash, course_id, feedback, timestamp)
            VALUES (?, ?, ?, ?)
        ''', (user_profile_hash, course_id, feedback, datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
    
    def get_user_preferences(self, user_profile_hash: str) -> Dict:
        """Retrieve user preferences from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT liked_tags, disliked_tags, preferred_providers, preferred_skill_levels
            FROM user_preferences WHERE user_profile_hash = ?
        ''', (user_profile_hash,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                'liked_tags': json.loads(result[0]) if result[0] else [],
                'disliked_tags': json.loads(result[1]) if result[1] else [],
                'preferred_providers': json.loads(result[2]) if result[2] else [],
                'preferred_skill_levels': json.loads(result[3]) if result[3] else []
            }
        return {'liked_tags': [], 'disliked_tags': [], 'preferred_providers': [], 'preferred_skill_levels': []}
    
    def update_user_preferences(self, user_profile_hash: str, preferences: Dict):
        """Update user preferences in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO user_preferences 
            (user_profile_hash, liked_tags, disliked_tags, preferred_providers, preferred_skill_levels, last_updated)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            user_profile_hash,
            json.dumps(preferences['liked_tags']),
            json.dumps(preferences['disliked_tags']),
            json.dumps(preferences['preferred_providers']),
            json.dumps(preferences['preferred_skill_levels']),
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()

class CourseRecommendationSystem:
    def __init__(self):
        self.courses = COURSES
        self.db_manager = DatabaseManager()
        
        # Initialize sentence transformer for embeddings
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self._compute_course_embeddings()
        except Exception as e:
            st.error(f"Error loading sentence transformer: {e}")
            self.model = None
        
        # Initialize Cohere client
        self.cohere_client = None
        if os.getenv("COHERE_API_KEY"):
            try:
                self.cohere_client = cohere.Client(os.getenv("COHERE_API_KEY"))
            except Exception as e:
                st.warning(f"Cohere initialization error: {e}")
        
        # Learning path knowledge base
        self.learning_paths = {
            "data-scientist": [
                "Python for Data Science and Machine Learning",
                "SQL for Data Analysis",
                "Machine Learning Specialization",
                "Data Visualization with Python",
                "Deep Learning Specialization"
            ],
            "ml-engineer": [
                "Python for Data Science and Machine Learning",
                "Machine Learning Specialization",
                "Deep Learning Specialization",
                "TensorFlow for Deep Learning",
                "AWS Cloud Practitioner"
            ],
            "web-developer": [
                "Full Stack Web Development",
                "React.js Frontend Development",
                "Node.js Backend Development",
                "Python Web Development with Django",
                "DevOps Engineering"
            ],
            "mobile-developer": [
                "React Native Mobile Development",
                "Flutter Mobile App Development",
                "Android App Development with Kotlin",
                "iOS Development with Swift"
            ]
        }
    
    def _compute_course_embeddings(self):
        """Compute embeddings for all courses using sentence transformers"""
        if not self.model:
            return
            
        course_texts = []
        for course in self.courses:
            # Create more comprehensive text representation
            text = (
                f"Title: {course['title']} "
                f"Description: {course['description']} "
                f"Level: {course['skill_level']} "
                f"Tags: {' '.join(course['tags'])} "
                f"Provider: {course['provider']}"
            )
            course_texts.append(text)
        
        try:
            # Normalize embeddings
            embeddings = self.model.encode(course_texts)
            # L2 normalization
            self.course_embeddings = embeddings / np.linalg.norm(embeddings, axis=1)[:, np.newaxis]
        except Exception as e:
            st.error(f"Error computing course embeddings: {e}")
            self.course_embeddings = None
    
    def create_user_profile_embedding(self, background: str, interests: str, goals: str, skills: str = ""):
        """Create user profile embedding using sentence transformers"""
        if not self.model:
            return None
            
        profile_text = f"Background: {background}. Interests: {interests}. Goals: {goals}. Skills: {skills}"
        try:
            return self.model.encode([profile_text])
        except Exception as e:
            st.error(f"Error creating user profile embedding: {e}")
            return None
    
    def get_user_profile_hash(self, background: str, interests: str, goals: str, skills: str = "") -> str:
        """Create a hash for user profile for database storage"""
        profile_text = f"{background}|{interests}|{goals}|{skills}"
        return str(hash(profile_text))
    
    def get_recommendations(self, background: str, interests: str, goals: str, skills: str = "", 
                          filters: Dict = None, top_k: int = 5) -> List[Dict]:
        """Get course recommendations using hybrid approach (semantic + keyword)"""
        
        # Get both semantic and keyword-based recommendations
        semantic_recommendations = self._get_semantic_recommendations(
            background, interests, goals, skills, filters, top_k
        ) if self.model and self.course_embeddings is not None else []
        
        keyword_recommendations = self._get_tag_based_recommendations(
            background, interests, goals, skills, filters, top_k
        )
        
        # Combine recommendations with weighted scores
        combined_recommendations = {}
        
        # Add semantic recommendations with 0.7 weight
        for course in semantic_recommendations:
            course_id = course['id']
            combined_recommendations[course_id] = {
                **course,
                'similarity_score': course['similarity_score'] * 0.7
            }
        
        # Add keyword recommendations with 0.3 weight
        for course in keyword_recommendations:
            course_id = course['id']
            if course_id in combined_recommendations:
                # If course exists in both, take weighted average
                combined_recommendations[course_id]['similarity_score'] += (
                    course['similarity_score'] * 0.3
                )
            else:
                course['similarity_score'] *= 0.3
                combined_recommendations[course_id] = course
        
        # Convert to list and sort by final score
        final_recommendations = list(combined_recommendations.values())
        final_recommendations.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        return final_recommendations[:top_k]

    def _get_semantic_recommendations(self, background: str, interests: str, goals: str, 
                                   skills: str = "", filters: Dict = None, top_k: int = 5):
        """Get recommendations using semantic similarity"""
        profile_text = (
            f"Background: {background} "
            f"Interests: {interests} "
            f"Goals: {goals} "
            f"Skills: {skills}"
        )
        
        try:
            # Create and normalize user embedding
            user_embedding = self.model.encode([profile_text])
            user_embedding = user_embedding / np.linalg.norm(user_embedding)
            
            # Calculate similarities
            similarities = cosine_similarity(user_embedding, self.course_embeddings).flatten()
            
            # Create scored courses with semantic similarity
            scored_courses = [
                {
                    **course,
                    'similarity_score': float(score),
                    'recommendation_reason': f"Semantic match score: {score:.2f}"
                }
                for course, score in zip(self.courses, similarities)
            ]
            
            # Apply filters if any
            if filters:
                scored_courses = self._apply_filters(scored_courses, filters)
            
            return scored_courses
            
        except Exception as e:
            st.error(f"Error in semantic recommendation: {e}")
            return []
    
    def _get_tag_based_recommendations(self, background: str, interests: str, goals: str, 
                                     skills: str = "", filters: Dict = None, top_k: int = 5) -> List[Dict]:
        """Fallback tag-based recommendation system"""
        profile_keywords = self._extract_keywords(f"{background} {interests} {goals} {skills}")
        
        scored_courses = []
        for course in self.courses:
            score = 0
            # Match with tags
            for tag in course['tags']:
                if any(keyword.lower() in tag.lower() for keyword in profile_keywords):
                    score += 2
            
            # Match with title and description
            course_text = f"{course['title']} {course['description']}".lower()
            for keyword in profile_keywords:
                if keyword.lower() in course_text:
                    score += 1
            
            if score > 0:
                course_copy = course.copy()
                course_copy['similarity_score'] = score / 10.0  # Normalize
                course_copy['recommendation_reason'] = self._get_recommendation_reason(course, background, interests, goals)
                scored_courses.append(course_copy)
        
        # Apply filters
        if filters:
            scored_courses = self._apply_filters(scored_courses, filters)
        
        # Sort by score and return top k
        scored_courses.sort(key=lambda x: x['similarity_score'], reverse=True)
        return scored_courses[:top_k]
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text"""
        # Simple keyword extraction
        words = re.findall(r'\b\w+\b', text.lower())
        # Filter out common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
        return [word for word in words if word not in stop_words and len(word) > 2]
    
    def _apply_filters(self, courses: List[Dict], filters: Dict) -> List[Dict]:
        """Apply filters to course list"""
        if not filters:
            return courses
        
        filtered = courses.copy()
        
        if 'providers' in filters and filters['providers']:
            filtered = [c for c in filtered if c['provider'] in filters['providers']]
        
        if 'skill_levels' in filters and filters['skill_levels']:
            filtered = [c for c in filtered if c['skill_level'] in filters['skill_levels']]
        
        if 'cost' in filters and filters['cost']:
            filtered = [c for c in filtered if c['cost'] in filters['cost']]
        
        if 'max_duration' in filters and filters['max_duration']:
            # Simple duration filtering (assumes duration is in months)
            max_months = filters['max_duration']
            filtered = [c for c in filtered if self._extract_duration_months(c['duration']) <= max_months]
        
        return filtered
    
    def _extract_duration_months(self, duration: str) -> int:
        """Extract duration in months from duration string"""
        # Simple extraction - assumes format like "6 months"
        months = re.findall(r'(\d+)\s*months?', duration.lower())
        return int(months[0]) if months else 12  # Default to 12 if can't parse
    
    def _apply_preference_adjustments(self, similarities: np.ndarray, preferences: Dict) -> np.ndarray:
        """Apply user preference adjustments to similarity scores"""
        adjusted_similarities = similarities.copy()
        
        for i, course in enumerate(self.courses):
            # Boost scores for liked tags
            for tag in course['tags']:
                if tag in preferences.get('liked_tags', []):
                    adjusted_similarities[i] *= 1.3
                elif tag in preferences.get('disliked_tags', []):
                    adjusted_similarities[i] *= 0.7
            
            # Boost for preferred providers
            if course['provider'] in preferences.get('preferred_providers', []):
                adjusted_similarities[i] *= 1.2
            
            # Boost for preferred skill levels
            if course['skill_level'] in preferences.get('preferred_skill_levels', []):
                adjusted_similarities[i] *= 1.1
        
        return adjusted_similarities
    
    def _get_recommendation_reason(self, course: Dict, background: str, interests: str, goals: str) -> str:
        """Generate explanation for why this course was recommended"""
        reasons = []
        
        # Check for keyword matches
        profile_text = f"{background} {interests} {goals}".lower()
        course_tags = [tag.replace('-', ' ') for tag in course['tags']]
        
        matching_tags = [tag for tag in course_tags if tag.lower() in profile_text]
        if matching_tags:
            reasons.append(f"Matches your interests in: {', '.join(matching_tags[:2])}")
        
        if course['skill_level'] == 'beginner' and 'beginner' in profile_text:
            reasons.append("Perfect for beginners")
        elif course['skill_level'] == 'advanced' and any(word in profile_text for word in ['advanced', 'expert', 'senior']):
            reasons.append("Matches your advanced skill level")
        
        if not reasons:
            reasons.append("High relevance match based on your profile")
        
        return ". ".join(reasons[:2])
    
    def record_feedback(self, background: str, interests: str, goals: str, skills: str, course_id: int, feedback: str):
        """Record user feedback and update preferences"""
        user_profile_hash = self.get_user_profile_hash(background, interests, goals, skills)
        
        # Save feedback to database
        self.db_manager.save_feedback(user_profile_hash, course_id, feedback)
        
        # Update preferences based on feedback
        course = next(c for c in self.courses if c['id'] == course_id)
        preferences = self.db_manager.get_user_preferences(user_profile_hash)
        
        if feedback == 'like':
            for tag in course['tags']:
                if tag not in preferences['liked_tags']:
                    preferences['liked_tags'].append(tag)
            
            if course['provider'] not in preferences['preferred_providers']:
                preferences['preferred_providers'].append(course['provider'])
            
            if course['skill_level'] not in preferences['preferred_skill_levels']:
                preferences['preferred_skill_levels'].append(course['skill_level'])
        
        elif feedback == 'dislike':
            for tag in course['tags']:
                if tag not in preferences['disliked_tags']:
                    preferences['disliked_tags'].append(tag)
                # Remove from liked if present
                if tag in preferences['liked_tags']:
                    preferences['liked_tags'].remove(tag)
        
        # Update preferences in database
        self.db_manager.update_user_preferences(user_profile_hash, preferences)
    
    def get_learning_path(self, career_goal: str) -> List[str]:
        """Get suggested learning path for a career goal"""
        goal_lower = career_goal.lower()
        
        for path_key, courses in self.learning_paths.items():
            if path_key.replace('-', ' ') in goal_lower or any(word in goal_lower for word in path_key.split('-')):
                return courses
        
        return []
    
    def answer_learning_question(self, question: str, background: str = "", interests: str = "", 
                               goals: str = "", skills: str = "") -> str:
        """
        Answer learning-related questions using Cohere LLM with context
        """
        if not self.cohere_client:
            return "Cohere API key not found. Please set the COHERE_API_KEY environment variable to use this feature."
        
        # Create context from user profile and knowledge base
        context_parts = []
        
        if background or interests or goals or skills:
            context_parts.append(f"User Profile - Background: {background}, Interests: {interests}, Goals: {goals}, Skills: {skills}")
        
        # Add relevant learning paths to context
        context_parts.append("Available Learning Paths:")
        for path, courses in self.learning_paths.items():
            context_parts.append(f"- {path.replace('-', ' ').title()}: {' → '.join(courses[:3])}")
        
        # Add sample course information
        context_parts.append("\nSample Available Courses:")
        for course in self.courses[:5]:
            context_parts.append(f"- {course['title']} ({course['provider']}, {course['skill_level']})")
        
        context = "\n".join(context_parts)
        
        # Create comprehensive prompt
        prompt = f"""You are an expert education and career advisor. Based on the following context and user question, provide helpful, practical guidance about learning paths and course recommendations.

Context:
{context}

User Question: {question}

Instructions:
1. Provide a clear, structured answer
2. Include specific recommendations when possible
3. Consider the user's background and goals if provided
4. Suggest concrete next steps
5. Keep the response practical and actionable
6. If recommending courses, mention why they're relevant

Answer:"""

        try:
            response = self.cohere_client.generate(
                model='command',
                prompt=prompt,
                max_tokens=500,
                temperature=0.7,
                k=0,
                p=0.9,
                stop_sequences=[]
            )
            
            return response.generations[0].text.strip()
            
        except Exception as e:
            return f"Error generating response: {str(e)}. Please check your Cohere API key and try again."

    def _add_recommendation_metadata(self, course: Dict, semantic_score: float, 
                                   keyword_score: float) -> Dict:
        """Add explanation of how the recommendation was made"""
        course = course.copy()
        course['recommendation_source'] = {
            'semantic_score': semantic_score,
            'keyword_score': keyword_score,
            'final_score': course['similarity_score']
        }
        course['recommendation_reason'] = (
            f"Semantic relevance: {semantic_score:.2f}, "
            f"Keyword relevance: {keyword_score:.2f}"
        )
        return course

# Streamlit UI
def main():
    st.set_page_config(page_title="AI Course Recommendation System", page_icon="🎓", layout="wide")
    
    st.title("🎓 AI Course Recommendation System")
    st.markdown("Get personalized course recommendations and learning path guidance powered by AI!")
    
    # Initialize system
    if 'recommender' not in st.session_state:
        with st.spinner("Initializing recommendation system..."):
            st.session_state.recommender = CourseRecommendationSystem()
        st.session_state.current_recommendations = []
        st.session_state.user_profile = {"background": "", "interests": "", "goals": "", "skills": ""}
    
    # Sidebar for user profile and filters
    with st.sidebar:
        st.header("👤 Your Profile")
        
        # User profile inputs
        background = st.text_area(
            "Current Background", 
            value=st.session_state.user_profile["background"],
            placeholder="e.g., Final-year Computer Science student with internship experience",
            help="Describe your current educational/professional background",
            height=80
        )
        
        interests = st.text_area(
            "Interests", 
            value=st.session_state.user_profile["interests"],
            placeholder="e.g., Artificial Intelligence, Machine Learning, Data Science",
            help="What topics or technologies interest you?",
            height=60
        )
        
        goals = st.text_area(
            "Career Goals", 
            value=st.session_state.user_profile["goals"],
            placeholder="e.g., Become a Machine Learning Engineer at a tech company",
            help="What are your career aspirations?",
            height=60
        )
        
        skills = st.text_input(
            "Current Skills (Optional)", 
            value=st.session_state.user_profile["skills"],
            placeholder="e.g., Python, SQL, Basic Statistics",
            help="List your current technical skills"
        )
        
        st.markdown("---")
        st.subheader("🔍 Filters")
        
        # Filters
        providers = st.multiselect(
            "Preferred Providers",
            options=list(set(course["provider"] for course in COURSES)),
            help="Filter by course providers"
        )
        
        skill_levels = st.multiselect(
            "Skill Levels",
            options=["beginner", "intermediate", "advanced"],
            help="Filter by difficulty level"
        )
        
        cost_filter = st.multiselect(
            "Cost Preference",
            options=["free", "paid"],
            help="Filter by course cost"
        )
        
        max_duration = st.slider(
            "Max Duration (months)",
            min_value=1,
            max_value=12,
            value=12,
            help="Maximum course duration in months"
        )
        
        num_recommendations = st.slider(
            "Number of Recommendations",
            min_value=3,
            max_value=10,
            value=5
        )
        
        # Get recommendations button
        if st.button("🔍 Get Recommendations", type="primary", use_container_width=True):
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
                if providers:
                    filters["providers"] = providers
                if skill_levels:
                    filters["skill_levels"] = skill_levels
                if cost_filter:
                    filters["cost"] = cost_filter
                if max_duration < 12:
                    filters["max_duration"] = max_duration
                
                # Get recommendations
                with st.spinner("Finding the best courses for you..."):
                    recommendations = st.session_state.recommender.get_recommendations(
                        background, interests, goals, skills, filters, num_recommendations
                    )
                st.session_state.current_recommendations = recommendations
                st.success("Recommendations updated!")
                st.rerun()
            else:
                st.error("Please fill in at least one field to get recommendations.")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📚 Recommended Courses")
        
        if st.session_state.current_recommendations:
            for i, course in enumerate(st.session_state.current_recommendations):
                with st.expander(f"#{i+1}: {course['title']}", expanded=(i < 2)):
                    # Course details in columns
                    info_col1, info_col2 = st.columns(2)
                    
                    with info_col1:
                        st.write(f"**Provider:** {course['provider']}")
                        st.write(f"**Level:** {course['skill_level'].title()}")
                        st.write(f"**Duration:** {course['duration']}")
                        st.write(f"**Cost:** {course['cost'].title()}")
                    
                    with info_col2:
                        st.write(f"**Rating:** {'⭐' * int(course['rating'])} ({course['rating']}/5)")
                        st.write(f"**Match Score:** {course['similarity_score']:.3f}")
                        if 'recommendation_reason' in course:
                            st.write(f"**Why recommended:** {course['recommendation_reason']}")
                        
                        if 'recommendation_source' in course:
                            st.write("**Recommendation Details:**")
                            st.write(f"- Semantic match: {course['recommendation_source']['semantic_score']:.2f}")
                            st.write(f"- Keyword match: {course['recommendation_source']['keyword_score']:.2f}")
                            st.write(f"- Final score: {course['recommendation_source']['final_score']:.2f}")
                    
                    st.write("**Description:**")
                    st.write(course['description'])
                    
                    st.write("**Skills you'll learn:**")
                    # Display tags as badges
                    tags_html = " ".join([
                        f"<span style='background-color: #e1f5fe; color: #01579b; padding: 4px 8px; border-radius: 12px; margin: 2px; display: inline-block; font-size: 0.8em; border: 1px solid #b3e5fc;'>{tag.replace('-', ' ').title()}</span>" 
                        for tag in course['tags']
                    ])
                    st.markdown(tags_html, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    # Feedback buttons
                    feedback_col1, feedback_col2, feedback_col3 = st.columns([1, 1, 2])
                    with feedback_col1:
                        if st.button(f"👍 Like", key=f"like_{course['id']}", use_container_width=True):
                            st.session_state.recommender.record_feedback(
                                background, interests, goals, skills, course['id'], 'like'
                            )
                            st.success("✅ Feedback recorded!")
                            st.rerun()
                    
                    with feedback_col2:
                        if st.button(f"👎 Dislike", key=f"dislike_{course['id']}", use_container_width=True):
                            st.session_state.recommender.record_feedback(
                                background, interests, goals, skills, course['id'], 'dislike'
                            )
                            st.success("✅ Feedback recorded!")
                            st.rerun()
                    
                    with feedback_col3:
                        st.write("")  # Spacer
        else:
            st.info("👈 Fill in your profile in the sidebar to get personalized course recommendations!")
            
            # Show sample courses
            st.subheader("🌟 Popular Courses")
            sample_courses = COURSES[:3]
            for course in sample_courses:
                with st.container():
                    st.write(f"**{course['title']}** - {course['provider']}")
                    st.write(f"Level: {course['skill_level'].title()} | Duration: {course['duration']} | Rating: {course['rating']}/5")
                    st.write(course['description'][:100] + "...")
                    st.markdown("---")
    
    with col2:
        st.header("🤖 Learning Path Assistant")
        st.markdown("Ask questions about your learning journey!")
        
        # Predefined quick questions
        st.subheader("🚀 Quick Questions")
        quick_questions = [
            "Should I learn Python or R for data science?",
            "What's the best path to become an ML engineer?",
            "How do I transition to web development?",
            "What skills do I need for cybersecurity?"
        ]
        
        for i, qq in enumerate(quick_questions):
            if st.button(f"💡 {qq}", key=f"quick_{i}", use_container_width=True):
                with st.spinner("Thinking..."):
                    answer = st.session_state.recommender.answer_learning_question(
                        qq, background, interests, goals, skills
                    )
                with st.container():
                    st.write("**Q:** " + qq)
                    st.write("**A:** " + answer)
                    st.markdown("---")
        
        st.subheader("💬 Custom Question")
        question = st.text_area(
            "Ask your question:",
            placeholder="e.g., What programming language should I learn first?",
            height=100
        )
        
        if st.button("🔮 Ask Assistant", use_container_width=True):
            if question:
                with st.spinner("Generating response..."):
                    answer = st.session_state.recommender.answer_learning_question(
                        question, background, interests, goals, skills
                    )
                with st.container():
                    st.write("**Your Question:**")
                    st.write(question)
                    st.write("**Assistant's Answer:**")
                    st.write(answer)
            else:
                st.error("Please enter a question.")
        
        # Show learning paths if goals are provided
        if goals:
            st.subheader("🗺️ Suggested Learning Path")
            learning_path = st.session_state.recommender.get_learning_path(goals)
            if learning_path:
                st.write("Based on your goals, here's a suggested course sequence:")
                for i, course_title in enumerate(learning_path, 1):
                    st.write(f"{i}. {course_title}")
            else:
                st.write("Enter more specific career goals to see suggested learning paths.")
    
    # Footer with statistics
    st.markdown("---")
    footer_col1, footer_col2, footer_col3, footer_col4 = st.columns(4)
    
    with footer_col1:
        st.metric("Total Courses", len(COURSES))
    
    with footer_col2:
        free_courses = len([c for c in COURSES if c['cost'] == 'free'])
        st.metric("Free Courses", free_courses)
    
    with footer_col3:
        providers = len(set(course['provider'] for course in COURSES))
        st.metric("Providers", providers)
    
    with footer_col4:
        avg_rating = sum(course['rating'] for course in COURSES) / len(COURSES)
        st.metric("Avg Rating", f"{avg_rating:.1f}⭐")
    
    st.markdown("*Built with Streamlit, SentenceTransformers, and Cohere AI*")

if __name__ == "__main__":
    main()