import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from .nlp_pipeline import NLPPipeline

class JobMatcher:
    """
    Main job matching algorithm with explainable features
    """
    
    def __init__(self):
        self.nlp_pipeline = NLPPipeline()
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10
        )
        self.feature_names = []
        self.is_trained = False
        
        # Weights for different matching components
        self.weights = {
            'skills_overlap': 0.4,
            'text_similarity': 0.3,
            'experience_match': 0.2,
            'overall_compatibility': 0.1
        }
    
    def extract_matching_features(self, resume_features: Dict, job_features: Dict) -> np.ndarray:
        """Extract features for matching algorithm"""
        features = []
        
        # 1. Skills overlap scores
        skills_overlap = self.nlp_pipeline.calculate_skills_overlap(
            resume_features['skills'], 
            job_features['skills']
        )
        
        for category, score in skills_overlap.items():
            features.append(score)
        
        # 2. Text similarity
        text_similarity = cosine_similarity(
            [resume_features['embeddings']], 
            [job_features['embeddings']]
        )[0][0]
        features.append(text_similarity)
        
        # 3. Experience matching
        resume_exp = resume_features['experience_years']
        job_exp = job_features.get('experience_years', 0)
        
        # Experience score (1.0 if resume exp >= job exp, scaled down if less)
        if job_exp == 0:
            exp_score = 1.0
        else:
            exp_score = min(1.0, resume_exp / job_exp)
        features.append(exp_score)
        
        # 4. Text length compatibility (normalized)
        resume_len = resume_features['text_length']
        job_len = job_features['text_length']
        len_ratio = min(resume_len, job_len) / max(resume_len, job_len) if max(resume_len, job_len) > 0 else 0
        features.append(len_ratio)
        
        # 5. Overall skills count
        total_resume_skills = sum(len(skills) for skills in resume_features['skills'].values())
        total_job_skills = sum(len(skills) for skills in job_features['skills'].values())
        
        skills_ratio = min(total_resume_skills, total_job_skills) / max(total_resume_skills, total_job_skills) if max(total_resume_skills, total_job_skills) > 0 else 0
        features.append(skills_ratio)
        
        return np.array(features)
    
    def get_feature_names(self) -> List[str]:
        """Get feature names for explainability"""
        names = []
        
        # Skills categories
        for category in self.nlp_pipeline.skills_keywords.keys():
            names.append(f'skills_overlap_{category}')
        
        # Other features
        names.extend([
            'text_similarity',
            'experience_match',
            'text_length_compatibility',
            'skills_count_ratio'
        ])
        
        return names
    
    def calculate_compatibility_score(self, resume_features: Dict, job_features: Dict) -> Dict:
        """Calculate detailed compatibility score"""
        # Extract features
        features = self.extract_matching_features(resume_features, job_features)
        feature_names = self.get_feature_names()
        
        # Skills overlap component
        skills_overlap = self.nlp_pipeline.calculate_skills_overlap(
            resume_features['skills'], 
            job_features['skills']
        )
        skills_score = np.mean(list(skills_overlap.values()))
        
        # Text similarity component
        text_similarity = features[len(skills_overlap)]  # Text similarity is after skills features
        
        # Experience match component
        experience_score = features[len(skills_overlap) + 1]
        
        # Overall compatibility (using all features)
        if self.is_trained:
            # Use trained model probability
            scaled_features = self.scaler.transform([features])
            overall_score = self.model.predict_proba(scaled_features)[0][1]  # Probability of match
        else:
            # Use weighted average as fallback
            overall_score = (
                skills_score * self.weights['skills_overlap'] +
                text_similarity * self.weights['text_similarity'] +
                experience_score * self.weights['experience_match']
            )
        
        # Detailed breakdown
        compatibility = {
            'overall_score': overall_score,
            'skills_score': skills_score,
            'text_similarity': text_similarity,
            'experience_score': experience_score,
            'skills_breakdown': skills_overlap,
            'feature_values': dict(zip(feature_names, features)),
            'resume_skills_found': resume_features['skills'],
            'job_skills_required': job_features['skills']
        }
        
        return compatibility
    
    def train_model(self, training_data: List[Tuple[Dict, Dict, float]]):
        """Train the matching model with labeled data"""
        X = []
        y = []
        
        for resume_features, job_features, label in training_data:
            features = self.extract_matching_features(resume_features, job_features)
            X.append(features)
            y.append(1 if label > 0.7 else 0)  # Binary classification threshold
        
        X = np.array(X)
        y = np.array(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        self.feature_names = self.get_feature_names()
        self.is_trained = True
        
        print(f"Model trained on {len(training_data)} samples")
        print(f"Feature importance: {dict(zip(self.feature_names, self.model.feature_importances_))}")
    
    def match_resume_to_jobs(self, resume_text: str, job_descriptions: List[str]) -> List[Dict]:
        """Match a resume to multiple job descriptions"""
        # Process resume
        resume_features = self.nlp_pipeline.process_document(resume_text, 'resume')
        
        matches = []
        for i, job_text in enumerate(job_descriptions):
            # Process job description
            job_features = self.nlp_pipeline.process_document(job_text, 'job')
            
            # Calculate compatibility
            compatibility = self.calculate_compatibility_score(resume_features, job_features)
            
            match_result = {
                'job_index': i,
                'job_text': job_text,
                'compatibility': compatibility,
                'rank_score': compatibility['overall_score']
            }
            
            matches.append(match_result)
        
        # Sort by compatibility score
        matches.sort(key=lambda x: x['rank_score'], reverse=True)
        
        return matches
    
    def get_top_matching_features(self, resume_features: Dict, job_features: Dict, top_n: int = 5) -> List[Dict]:
        """Get top features contributing to the match"""
        features = self.extract_matching_features(resume_features, job_features)
        feature_names = self.get_feature_names()
        
        # Create feature importance based on values
        feature_importance = []
        for i, (name, value) in enumerate(zip(feature_names, features)):
            feature_importance.append({
                'feature': name,
                'value': value,
                'importance': value,  # For now, use value as importance
                'description': self._get_feature_description(name, value)
            })
        
        # Sort by importance
        feature_importance.sort(key=lambda x: x['importance'], reverse=True)
        
        return feature_importance[:top_n]
    
    def _get_feature_description(self, feature_name: str, value: float) -> str:
        """Generate human-readable description for features"""
        if 'skills_overlap' in feature_name:
            category = feature_name.replace('skills_overlap_', '')
            return f"{value:.2%} match in {category.replace('_', ' ')} skills"
        elif feature_name == 'text_similarity':
            return f"{value:.2%} overall text similarity"
        elif feature_name == 'experience_match':
            return f"{value:.2%} experience requirement match"
        elif feature_name == 'text_length_compatibility':
            return f"{value:.2%} document length compatibility"
        elif feature_name == 'skills_count_ratio':
            return f"{value:.2%} skills quantity match"
        else:
            return f"{feature_name}: {value:.3f}"
    
    def save_model(self, filepath: str):
        """Save trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'weights': self.weights
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model"""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.weights = model_data['weights']
        self.is_trained = True
        
        print(f"Model loaded from {filepath}")

# Example usage
if __name__ == "__main__":
    matcher = JobMatcher()
    
    # Sample data
    resume = """
    Jane Smith - Data Scientist
    3 years of experience in Python, machine learning, and data analysis.
    Proficient in pandas, scikit-learn, and data visualization.
    Experience with AWS and SQL databases.
    """
    
    jobs = [
        """
        Data Scientist Position
        Looking for 2+ years experience in Python and machine learning.
        Required: pandas, scikit-learn, SQL.
        Preferred: AWS, data visualization experience.
        """,
        """
        Software Engineer Role
        5+ years experience required in Java and web development.
        Must know Spring framework and REST APIs.
        Database experience preferred.
        """
    ]
    
    # Match resume to jobs
    matches = matcher.match_resume_to_jobs(resume, jobs)
    
    for match in matches:
        print(f"\nJob {match['job_index']}:")
        print(f"Overall Score: {match['compatibility']['overall_score']:.3f}")
        print(f"Skills Score: {match['compatibility']['skills_score']:.3f}")
        print(f"Text Similarity: {match['compatibility']['text_similarity']:.3f}")
        print("Skills Breakdown:", match['compatibility']['skills_breakdown'])