from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import sys
import os
import json
from typing import Dict, List

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from explainer import ExplainableJobMatcher
from matcher import JobMatcher
from nlp_pipeline import NLPPipeline

app = Flask(__name__)
CORS(app)

# Initialize the explainable matcher
explainer = ExplainableJobMatcher()

@app.route('/')
def home():
    """Home page with API documentation"""
    return jsonify({
        "message": "Explainable AI Job Matcher API",
        "version": "1.0.0",
        "endpoints": {
            "/api/match": "POST - Match resume to job descriptions",
            "/api/explain": "POST - Get detailed explanation for a match",
            "/api/health": "GET - Health check",
            "/api/skills": "GET - Get available skills categories"
        }
    })

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "message": "Job Matcher API is running"
    })

@app.route('/api/skills')
def get_skills():
    """Get available skills categories"""
    return jsonify({
        "skills_categories": explainer.matcher.nlp_pipeline.skills_keywords
    })

@app.route('/api/match', methods=['POST'])
def match_jobs():
    """
    Match a resume to multiple job descriptions
    
    Expected JSON payload:
    {
        "resume": "resume text here",
        "jobs": ["job description 1", "job description 2", ...]
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        resume_text = data.get('resume', '')
        job_descriptions = data.get('jobs', [])
        
        if not resume_text:
            return jsonify({"error": "Resume text is required"}), 400
        
        if not job_descriptions:
            return jsonify({"error": "At least one job description is required"}), 400
        
        # Perform matching
        matches = explainer.matcher.match_resume_to_jobs(resume_text, job_descriptions)
        
        # Format response
        response = {
            "status": "success",
            "total_jobs": len(job_descriptions),
            "matches": []
        }
        
        for match in matches:
            match_data = {
                "job_index": match['job_index'],
                "overall_score": match['compatibility']['overall_score'],
                "skills_score": match['compatibility']['skills_score'],
                "text_similarity": match['compatibility']['text_similarity'],
                "experience_score": match['compatibility']['experience_score'],
                "skills_breakdown": match['compatibility']['skills_breakdown'],
                "rank": len(response["matches"]) + 1
            }
            response["matches"].append(match_data)
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/explain', methods=['POST'])
def explain_match():
    """
    Get detailed explanation for a resume-job match
    
    Expected JSON payload:
    {
        "resume": "resume text here",
        "job": "job description here",
        "include_visualization": false,
        "export_report": false
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        resume_text = data.get('resume', '')
        job_text = data.get('job', '')
        include_viz = data.get('include_visualization', False)
        export_report = data.get('export_report', False)
        
        if not resume_text or not job_text:
            return jsonify({"error": "Both resume and job description are required"}), 400
        
        # Generate explanation
        explanation = explainer.generate_comprehensive_explanation(resume_text, job_text)
        
        # Format response
        response = {
            "status": "success",
            "explanation": {
                "overall_score": explanation['overall_score'],
                "summary": explanation['human_readable_summary'],
                "basic_analysis": explanation['basic_analysis'],
                "detailed_features": explanation['detailed_features']
            }
        }
        
        # Add SHAP analysis if available
        if 'shap_analysis' in explanation and 'error' not in explanation['shap_analysis']:
            shap_data = explanation['shap_analysis']
            response["explanation"]["shap_analysis"] = {
                "predicted_score": shap_data['predicted_score'],
                "top_positive_features": shap_data['top_positive_features'],
                "top_negative_features": shap_data['top_negative_features']
            }
        
        # Add LIME analysis if available
        if 'lime_analysis' in explanation and 'error' not in explanation['lime_analysis']:
            lime_data = explanation['lime_analysis']
            response["explanation"]["lime_analysis"] = {
                "score": lime_data['score'],
                "top_words": lime_data['top_words']
            }
        
        # Generate visualization if requested
        if include_viz:
            try:
                import matplotlib
                matplotlib.use('Agg')  # Use non-interactive backend
                import matplotlib.pyplot as plt
                import base64
                import io
                
                fig = explainer.create_explanation_visualization(explanation)
                
                # Convert plot to base64 string
                img_buffer = io.BytesIO()
                plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
                img_buffer.seek(0)
                img_str = base64.b64encode(img_buffer.read()).decode()
                plt.close(fig)
                
                response["explanation"]["visualization"] = f"data:image/png;base64,{img_str}"
                
            except Exception as viz_error:
                response["explanation"]["visualization_error"] = str(viz_error)
        
        # Export report if requested
        if export_report:
            try:
                report_path = f"reports/match_report_{hash(resume_text + job_text) % 10000}.md"
                os.makedirs('reports', exist_ok=True)
                explainer.export_explanation_report(explanation, resume_text, job_text, report_path)
                response["explanation"]["report_path"] = report_path
            except Exception as report_error:
                response["explanation"]["report_error"] = str(report_error)
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/batch_match', methods=['POST'])
def batch_match():
    """
    Batch process multiple resumes against multiple jobs
    
    Expected JSON payload:
    {
        "resumes": [{"id": "1", "text": "resume 1"}, ...],
        "jobs": [{"id": "job1", "text": "job desc 1"}, ...],
        "top_n": 3
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        resumes = data.get('resumes', [])
        jobs = data.get('jobs', [])
        top_n = data.get('top_n', 3)
        
        if not resumes or not jobs:
            return jsonify({"error": "Both resumes and jobs are required"}), 400
        
        results = []
        
        for resume_data in resumes:
            resume_id = resume_data.get('id', 'unknown')
            resume_text = resume_data.get('text', '')
            
            if not resume_text:
                continue
            
            # Get job descriptions as list
            job_texts = [job.get('text', '') for job in jobs]
            job_ids = [job.get('id', f'job_{i}') for i, job in enumerate(jobs)]
            
            # Match resume to all jobs
            matches = explainer.matcher.match_resume_to_jobs(resume_text, job_texts)
            
            # Get top N matches
            top_matches = matches[:top_n]
            
            resume_result = {
                "resume_id": resume_id,
                "top_matches": []
            }
            
            for match in top_matches:
                job_idx = match['job_index']
                match_result = {
                    "job_id": job_ids[job_idx] if job_idx < len(job_ids) else f'job_{job_idx}',
                    "score": match['compatibility']['overall_score'],
                    "skills_score": match['compatibility']['skills_score'],
                    "rank": len(resume_result["top_matches"]) + 1
                }
                resume_result["top_matches"].append(match_result)
            
            results.append(resume_result)
        
        response = {
            "status": "success",
            "processed_resumes": len(results),
            "total_jobs": len(jobs),
            "results": results
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/analyze_text', methods=['POST'])
def analyze_text():
    """
    Analyze a single text document (resume or job description)
    
    Expected JSON payload:
    {
        "text": "document text here",
        "doc_type": "resume" or "job"
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        text = data.get('text', '')
        doc_type = data.get('doc_type', 'resume')
        
        if not text:
            return jsonify({"error": "Text is required"}), 400
        
        # Process document
        features = explainer.matcher.nlp_pipeline.process_document(text, doc_type)
        
        response = {
            "status": "success",
            "analysis": {
                "doc_type": doc_type,
                "text_length": features['text_length'],
                "experience_years": features['experience_years'],
                "skills_found": features['skills'],
                "total_skills": sum(len(skills) for skills in features['skills'].values())
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('reports', exist_ok=True)
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)