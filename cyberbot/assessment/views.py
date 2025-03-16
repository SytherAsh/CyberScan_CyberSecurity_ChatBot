from django.shortcuts import render
from django.http import JsonResponse
from .utils import main, assess_domain, load_cybersecurity_domains
from .scoring import calculate_risk_score, verify_compliance
from .visual import visualize_risk_assessment, generate_report
from .models import AssessmentResult, DomainAssessment

def run_assessment_view(request):
    file_paths = ["/path/to/pdf"]  # Hardcode or get from request
    json_path = "/path/to/brain.json"
    max_predefined = int(request.GET.get('max_predefined', 2))
    max_dynamic = int(request.GET.get('max_dynamic', 1))
    
    retrieval_qa, llm = main(file_paths, json_path)
    raw_domains = load_cybersecurity_domains(json_path)
    domains = {name: DomainData(**data) for name, data in raw_domains.items()}
    documents = load_documents(file_paths)
    extracted_data = analyze_document_content(documents, domains)
    previous_questions = []
    assessment_result = AssessmentResult(domains={name: DomainAssessment(domain_name=name) for name in domains.keys()})
    
    query_cache = {}
    def cached_invoke(query: str) -> str:
        if query not in query_cache:
            query_cache[query] = retrieval_qa.invoke({"query": query})['result'].lower()
        return query_cache[query]
    
    for domain_name, domain_data in domains.items():
        domain_assessment = assess_domain(domain_name, domain_data, previous_questions, extracted_data, retrieval_qa, llm, cached_invoke)
        assessment_result.domains[domain_name] = domain_assessment
    
    assessment_result = calculate_risk_score(assessment_result, extracted_data, domains, retrieval_qa, cached_invoke)
    assessment_result = verify_compliance(assessment_result, extracted_data, domains, retrieval_qa, cached_invoke)
    
    risk_scores = {k: {"score": v.risk_score, "details": v.risk_details} for k, v in assessment_result.domains.items()}
    compliance_results = {k: v.compliance for k, v in assessment_result.domains.items()}
    report = generate_report(risk_scores, compliance_results, extracted_data, assessment_result.overall_risk_score)
    
    return JsonResponse({
        "overall_risk_score": assessment_result.overall_risk_score,
        "domains": {k: v.dict() for k, v in assessment_result.domains.items()},
        "report": report
    })
