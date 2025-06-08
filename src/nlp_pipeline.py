import re
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy

class NLPPipeline:
    """
    Core NLP pipeline for processing resumes and job descriptions
    """
    
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.skills_keywords = self._load_skills_database()
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
        
        self.stop_words = set(stopwords.words('english'))
        
    def _load_skills_database(self) -> Dict[str, List[str]]:
        """Load predefined skills database with categories"""
        return {
            'programming': [
                'python', 'java', 'javascript', 'c++', 'c#', 'r', 'sql', 'html', 'css',
                'react', 'angular', 'vue', 'node.js', 'django', 'flask', 'spring'
            ],
            'data_science': [
                'machine learning', 'deep learning', 'data analysis', 'statistics',
                'pandas', 'numpy', 'scikit-learn', 'tensorflow', 'pytorch', 'keras',
                'data visualization', 'tableau', 'power bi'
            ],
            'cloud': [
                'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform',
                'jenkins', 'ci/cd', 'devops'
            ],
            'soft_skills': [
                'leadership', 'communication', 'teamwork', 'problem solving',
                'project management', 'agile', 'scrum'
            ],
            'databases': [
                'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch',
                'oracle', 'sql server'
            ]
        }
    
    def clean_text(self, text: str) -> str:
        """Clean and preprocess text"""
        if not isinstance(text, str):
            return ""
        
        # Remove extra whitespace and special characters
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\.\-\+#]', ' ', text)
        
        # Convert to lowercase
        text = text.lower().strip()
        
        return text
    
    def extract_skills(self, text: str) -> Dict[str, List[str]]:
        """Extract skills from text based on predefined categories"""
        text = self.clean_text(text)
        found_skills = {category: [] for category in self.skills_keywords}
        
        for category, skills in self.skills_keywords.items():
            for skill in skills:
                if skill.lower() in text:
                    found_skills[category].append(skill)
        
        return found_skills
    
    def extract_experience_years(self, text: str) -> int:
        """Extract years of experience from text"""
        text = self.clean_text(text)
        
        # Common patterns for experience
        patterns = [
            r'(\d+)\+?\s*years?\s*(?:of\s*)?experience',
            r'(\d+)\+?\s*yrs?\s*(?:of\s*)?experience',
            r'experience\s*(?:of\s*)?(\d+)\+?\s*years?',
            r'(\d+)\+?\s*years?\s*in'
        ]
        
        years = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            years.extend([int(match) for match in matches])
        
        return max(years) if years else 0
    
    def get_tfidf_features(self, texts: List[str]) -> np.ndarray:
        """Generate TF-IDF features for texts"""
        cleaned_texts = [self.clean_text(text) for text in texts]
        return self.tfidf_vectorizer.fit_transform(cleaned_texts)
    
    def get_sentence_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate sentence embeddings using SentenceTransformer"""
        cleaned_texts = [self.clean_text(text) for text in texts]
        return self.sentence_model.encode(cleaned_texts)
    
    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts"""
        embeddings = self.get_sentence_embeddings([text1, text2])
        
        # Calculate cosine similarity
        dot_product = np.dot(embeddings[0], embeddings[1])
        norm_a = np.linalg.norm(embeddings[0])
        norm_b = np.linalg.norm(embeddings[1])
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    def calculate_skills_overlap(self, resume_skills: Dict, job_skills: Dict) -> Dict[str, float]:
        """Calculate skills overlap between resume and job"""
        overlap_scores = {}
        
        for category in resume_skills:
            resume_set = set(resume_skills[category])
            job_set = set(job_skills[category])
            
            if len(job_set) == 0:
                overlap_scores[category] = 0.0
            else:
                intersection = len(resume_set.intersection(job_set))
                overlap_scores[category] = intersection / len(job_set)
        
        return overlap_scores
    
    def process_document(self, text: str, doc_type: str = 'resume') -> Dict:
        """Process a document (resume or job description) and extract features"""
        features = {
            'raw_text': text,
            'cleaned_text': self.clean_text(text),
            'skills': self.extract_skills(text),
            'experience_years': self.extract_experience_years(text),
            'text_length': len(text),
            'doc_type': doc_type
        }
        
        # Get embeddings
        features['embeddings'] = self.get_sentence_embeddings([text])[0]
        
        return features

# Example usage and testing
if __name__ == "__main__":
    # Initialize pipeline
    nlp = NLPPipeline()
    
    # Sample resume text
    sample_resume = """
    John Doe
    Senior Software Engineer
    5 years of experience in Python, JavaScript, and machine learning.
    Proficient in React, Django, AWS, and PostgreSQL.
    Strong leadership and communication skills.
    """
    
    # Sample job description
    sample_job = """
    We are looking for a Software Engineer with 3+ years of experience.
    Required skills: Python, JavaScript, React, AWS.
    Experience with machine learning is a plus.
    Strong problem-solving and teamwork skills required.
    """
    
    # Process documents
    resume_features = nlp.process_document(sample_resume, 'resume')
    job_features = nlp.process_document(sample_job, 'job')
    
    print("Resume Skills:", resume_features['skills'])
    print("Job Skills:", job_features['skills'])
    print("Text Similarity:", nlp.calculate_text_similarity(sample_resume, sample_job))
    print("Skills Overlap:", nlp.calculate_skills_overlap(
        resume_features['skills'], 
        job_features['skills']
    ))