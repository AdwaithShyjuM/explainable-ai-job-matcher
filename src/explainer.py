import numpy as np
import pandas as pd
from typing import Dict, List, Any
import shap
import lime
from lime.lime_text import LimeTextExplainer
import matplotlib.pyplot as plt
import seaborn as sns
from .matcher import JobMatcher
from .nlp_pipeline import NLPPipeline

class ExplainableJobMatcher:
    """
    Explainable AI wrapper for job matching with SHAP and LIME interpretations
    """
    
    def __init__(self):
        self.matcher = JobMatcher()
        self.explainer_shap = None
        self.explainer_lime = LimeTextExplainer(class_names=['Not Match', 'Match'])
        
    def prepare_shap_explainer(self, background_data: List[Tuple[str, str]]):
        """Prepare SHAP explainer with background data"""
        # Identify gaps
        gaps = []
        skills_breakdown = compatibility['skills_breakdown']
        for category, score in skills_breakdown.items():
            if score < 0.5 and len(job_features['skills'][category]) > 0:
                gaps.append(f"limited {category.replace('_', ' ')} skills")
        
        if compatibility['experience_score'] < 0.7:
            gaps.append("experience requirements")
        
        # Build summary
        summary = f"This is a {match_quality} match (score: {score:.2f}). "
        
        if strengths:
            summary += f"Key strengths include: {', '.join(strengths)}. "
        
        if gaps:
            summary += f"Areas for improvement: {', '.join(gaps)}. "
        
        # Add specific skills found
        found_skills = []
        for category, skills in resume_features['skills'].items():
            if skills and category in job_features['skills'] and job_features['skills'][category]:
                matching_skills = set(skills) & set(job_features['skills'][category])
                if matching_skills:
                    found_skills.extend(matching_skills)
        
        if found_skills:
            summary += f"Matching skills found: {', '.join(found_skills[:5])}{'...' if len(found_skills) > 5 else ''}."
        
        return summary
    
    def create_explanation_visualization(self, explanation: Dict, save_path: str = None) -> plt.Figure:
        """Create visualization for the explanation"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Job Match Explanation Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Overall Score Gauge
        ax1 = axes[0, 0]
        score = explanation['overall_score']
        colors = ['red', 'orange', 'yellow', 'lightgreen', 'green']
        color_idx = min(int(score * 5), 4)
        
        ax1.pie([score, 1-score], colors=[colors[color_idx], 'lightgray'], 
                startangle=90, counterclock=False)
        ax1.add_artist(plt.Circle((0,0), 0.6, color='white'))
        ax1.text(0, 0, f'{score:.1%}', ha='center', va='center', fontsize=20, fontweight='bold')
        ax1.set_title('Overall Match Score')
        
        # 2. Skills Breakdown
        ax2 = axes[0, 1]
        skills_data = explanation['basic_analysis']['skills_breakdown']
        categories = list(skills_data.keys())
        scores = list(skills_data.values())
        
        bars = ax2.bar(categories, scores, color='skyblue', alpha=0.7)
        ax2.set_title('Skills Category Breakdown')
        ax2.set_ylabel('Match Score')
        ax2.set_ylim(0, 1)
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.2f}', ha='center', va='bottom')
        
        # 3. Feature Importance (if SHAP data available)
        ax3 = axes[1, 0]
        if 'shap_analysis' in explanation and 'error' not in explanation['shap_analysis']:
            shap_data = explanation['shap_analysis']
            top_features = shap_data['top_positive_features'][:6]
            
            if top_features:
                feature_names = [f['feature'].replace('_', ' ')[:15] for f in top_features]
                shap_values = [f['shap_value'] for f in top_features]
                
                bars = ax3.barh(feature_names, shap_values, color='lightcoral')
                ax3.set_title('Top Contributing Features (SHAP)')
                ax3.set_xlabel('SHAP Value')
        else:
            # Fallback to basic feature importance
            basic_features = explanation['detailed_features']
            top_features = sorted(basic_features.items(), key=lambda x: x[1], reverse=True)[:6]
            
            feature_names = [name.replace('_', ' ')[:15] for name, _ in top_features]
            values = [value for _, value in top_features]
            
            bars = ax3.barh(feature_names, values, color='lightcoral')
            ax3.set_title('Top Feature Values')
            ax3.set_xlabel('Feature Value')
        
        # 4. Component Scores
        ax4 = axes[1, 1]
        components = ['Skills Match', 'Text Similarity', 'Experience Match']
        component_scores = [
            explanation['basic_analysis']['skills_match'],
            explanation['basic_analysis']['text_similarity'],
            explanation['basic_analysis']['experience_match']
        ]
        
        wedges, texts, autotexts = ax4.pie(component_scores, labels=components, autopct='%1.1f%%',
                                          colors=['lightblue', 'lightgreen', 'lightyellow'])
        ax4.set_title('Score Components')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def export_explanation_report(self, explanation: Dict, resume_text: str, 
                                job_text: str, output_path: str):
        """Export detailed explanation report"""
        report = []
        report.append("# Job Match Explanation Report\n")
        report.append(f"**Overall Match Score: {explanation['overall_score']:.1%}**\n")
        
        # Human readable summary
        report.append("## Executive Summary")
        report.append(explanation['human_readable_summary'])
        report.append("")
        
        # Detailed breakdown
        report.append("## Detailed Analysis")
        basic = explanation['basic_analysis']
        report.append(f"- **Skills Match:** {basic['skills_match']:.1%}")
        report.append(f"- **Text Similarity:** {basic['text_similarity']:.1%}")
        report.append(f"- **Experience Match:** {basic['experience_match']:.1%}")
        report.append("")
        
        # Skills breakdown
        report.append("### Skills Category Analysis")
        for category, score in basic['skills_breakdown'].items():
            report.append(f"- **{category.replace('_', ' ').title()}:** {score:.1%}")
        report.append("")
        
        # SHAP analysis if available
        if 'shap_analysis' in explanation and 'error' not in explanation['shap_analysis']:
            report.append("### Feature Importance (SHAP Analysis)")
            shap_data = explanation['shap_analysis']
            
            if shap_data['top_positive_features']:
                report.append("**Top Contributing Features:**")
                for feature in shap_data['top_positive_features']:
                    report.append(f"- {feature['feature']}: {feature['shap_value']:.3f}")
                report.append("")
        
        # LIME analysis if available
        if 'lime_analysis' in explanation and 'error' not in explanation['lime_analysis']:
            report.append("### Text Analysis (LIME)")
            lime_data = explanation['lime_analysis']
            
            if lime_data['top_words']:
                report.append("**Key Words Contributing to Match:**")
                for word_data in lime_data['top_words'][:10]:
                    contrib = "+" if word_data['contribution'] == 'positive' else "-"
                    report.append(f"- {word_data['word']} ({contrib}{abs(word_data['importance']):.3f})")
                report.append("")
        
        # Recommendations
        report.append("## Recommendations")
        score = explanation['overall_score']
        if score >= 0.7:
            report.append("- **Action:** Proceed with interview process")
            report.append("- **Focus:** Validate technical skills in interview")
        elif score >= 0.5:
            report.append("- **Action:** Consider for phone screening")
            report.append("- **Focus:** Assess skill gaps and growth potential")
        else:
            report.append("- **Action:** Not recommended for current role")
            report.append("- **Consider:** Alternative positions or future opportunities")
        
        # Write report
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print(f"Explanation report saved to {output_path}")

# Example usage and testing
if __name__ == "__main__":
    # Initialize explainer
    explainer = ExplainableJobMatcher()
    
    # Sample data
    resume = """
    Alice Johnson - Machine Learning Engineer
    4 years of experience in Python, TensorFlow, and deep learning.
    Strong background in data science with pandas, scikit-learn, and SQL.
    Experience with AWS cloud services and Docker.
    Published research in computer vision and NLP.
    """
    
    job = """
    Senior ML Engineer Position
    We're seeking an experienced ML Engineer with 3+ years of experience.
    Required skills: Python, TensorFlow, scikit-learn, SQL.
    Preferred: AWS, Docker, research background.
    Strong problem-solving and communication skills required.
    """
    
    # Generate comprehensive explanation
    explanation = explainer.generate_comprehensive_explanation(resume, job)
    
    print("=== COMPREHENSIVE EXPLANATION ===")
    print(f"Overall Score: {explanation['overall_score']:.3f}")
    print(f"Summary: {explanation['human_readable_summary']}")
    print("\nSkills Breakdown:")
    for category, score in explanation['basic_analysis']['skills_breakdown'].items():
        print(f"  {category}: {score:.3f}")
    
    # Create visualization
    fig = explainer.create_explanation_visualization(explanation)
    plt.show()
    
    # Export report
    explainer.export_explanation_report(explanation, resume, job, "match_report.md") Process background data
        background_features = []
        for resume_text, job_text in background_data:
            resume_features = self.matcher.nlp_pipeline.process_document(resume_text, 'resume')
            job_features = self.matcher.nlp_pipeline.process_document(job_text, 'job')
            features = self.matcher.extract_matching_features(resume_features, job_features)
            background_features.append(features)
        
        background_features = np.array(background_features)
        
        # Create SHAP explainer
        if self.matcher.is_trained:
            # Use the trained model
            def model_predict(X):
                X_scaled = self.matcher.scaler.transform(X)
                return self.matcher.model.predict_proba(X_scaled)[:, 1]
            
            self.explainer_shap = shap.Explainer(model_predict, background_features)
        else:
            print("Warning: Model not trained. SHAP explanations will use feature-based scoring.")
    
    def explain_match_shap(self, resume_text: str, job_text: str) -> Dict:
        """Generate SHAP explanations for a resume-job match"""
        # Process documents
        resume_features = self.matcher.nlp_pipeline.process_document(resume_text, 'resume')
        job_features = self.matcher.nlp_pipeline.process_document(job_text, 'job')
        
        # Extract features
        features = self.matcher.extract_matching_features(resume_features, job_features)
        feature_names = self.matcher.get_feature_names()
        
        if self.explainer_shap is None:
            # Fallback to feature importance analysis
            compatibility = self.matcher.calculate_compatibility_score(resume_features, job_features)
            
            # Create manual importance scores
            shap_values = features * compatibility['overall_score']  # Scaled by overall score
            base_value = 0.5  # Neutral baseline
            
        else:
            # Use SHAP explainer
            shap_values = self.explainer_shap.shap_values(features.reshape(1, -1))[0]
            base_value = self.explainer_shap.expected_value
        
        explanation = {
            'shap_values': shap_values,
            'base_value': base_value,
            'feature_names': feature_names,
            'feature_values': features,
            'predicted_score': base_value + np.sum(shap_values),
            'top_positive_features': self._get_top_shap_features(shap_values, feature_names, features, positive=True),
            'top_negative_features': self._get_top_shap_features(shap_values, feature_names, features, positive=False)
        }
        
        return explanation
    
    def explain_match_lime(self, resume_text: str, job_text: str) -> Dict:
        """Generate LIME explanations for text-based matching"""
        
        def predict_fn(texts):
            """Prediction function for LIME"""
            predictions = []
            for text in texts:
                # Combine with job description for prediction
                combined_resume = self.matcher.nlp_pipeline.process_document(text, 'resume')
                job_features = self.matcher.nlp_pipeline.process_document(job_text, 'job')
                
                compatibility = self.matcher.calculate_compatibility_score(combined_resume, job_features)
                score = compatibility['overall_score']
                
                # Return probabilities for [Not Match, Match]
                predictions.append([1 - score, score])
            
            return np.array(predictions)
        
        # Generate LIME explanation
        explanation = self.explainer_lime.explain_instance(
            resume_text,
            predict_fn,
            num_features=10,
            num_samples=100
        )
        
        # Process explanation
        lime_explanation = {
            'explanation': explanation,
            'feature_importance': explanation.as_list(),
            'score': explanation.predict_proba[1],  # Probability of match
            'top_words': self._get_top_lime_words(explanation.as_list())
        }
        
        return lime_explanation
    
    def generate_comprehensive_explanation(self, resume_text: str, job_text: str) -> Dict:
        """Generate comprehensive explanation combining multiple methods"""
        
        # Basic compatibility analysis
        resume_features = self.matcher.nlp_pipeline.process_document(resume_text, 'resume')
        job_features = self.matcher.nlp_pipeline.process_document(job_text, 'job')
        compatibility = self.matcher.calculate_compatibility_score(resume_features, job_features)
        
        explanation = {
            'overall_score': compatibility['overall_score'],
            'basic_analysis': {
                'skills_match': compatibility['skills_score'],
                'text_similarity': compatibility['text_similarity'],
                'experience_match': compatibility['experience_score'],
                'skills_breakdown': compatibility['skills_breakdown']
            },
            'detailed_features': compatibility['feature_values'],
            'human_readable_summary': self._generate_human_summary(compatibility, resume_features, job_features)
        }
        
        # Add SHAP explanation if available
        try:
            shap_explanation = self.explain_match_shap(resume_text, job_text)
            explanation['shap_analysis'] = shap_explanation
        except Exception as e:
            explanation['shap_analysis'] = {'error': str(e)}
        
        # Add LIME explanation
        try:
            lime_explanation = self.explain_match_lime(resume_text, job_text)
            explanation['lime_analysis'] = lime_explanation
        except Exception as e:
            explanation['lime_analysis'] = {'error': str(e)}
        
        return explanation
    
    def _get_top_shap_features(self, shap_values: np.ndarray, feature_names: List[str], 
                              feature_values: np.ndarray, positive: bool = True, top_n: int = 5) -> List[Dict]:
        """Get top SHAP features (positive or negative)"""
        if positive:
            indices = np.argsort(shap_values)[-top_n:][::-1]
        else:
            indices = np.argsort(shap_values)[:top_n]
        
        top_features = []
        for idx in indices:
            if positive and shap_values[idx] <= 0:
                continue
            if not positive and shap_values[idx] >= 0:
                continue
                
            top_features.append({
                'feature': feature_names[idx],
                'shap_value': shap_values[idx],
                'feature_value': feature_values[idx],
                'contribution': 'positive' if shap_values[idx] > 0 else 'negative'
            })
        
        return top_features
    
    def _get_top_lime_words(self, lime_features: List[Tuple[str, float]], top_n: int = 10) -> List[Dict]:
        """Extract top words from LIME explanation"""
        # Sort by absolute importance
        sorted_features = sorted(lime_features, key=lambda x: abs(x[1]), reverse=True)
        
        top_words = []
        for word, importance in sorted_features[:top_n]:
            top_words.append({
                'word': word,
                'importance': importance,
                'contribution': 'positive' if importance > 0 else 'negative'
            })
        
        return top_words
    
    def _generate_human_summary(self, compatibility: Dict, resume_features: Dict, job_features: Dict) -> str:
        """Generate human-readable explanation summary"""
        score = compatibility['overall_score']
        skills_score = compatibility['skills_score']
        
        # Determine match quality
        if score >= 0.8:
            match_quality = "excellent"
        elif score >= 0.6:
            match_quality = "good"
        elif score >= 0.4:
            match_quality = "moderate"
        else:
            match_quality = "poor"
        
        # Identify strengths
        strengths = []
        if skills_score >= 0.7:
            strengths.append("strong skills alignment")
        if compatibility['text_similarity'] >= 0.7:
            strengths.append("high overall compatibility")
        if compatibility['experience_score'] >= 0.8:
            strengths.append("sufficient experience")
        
        #