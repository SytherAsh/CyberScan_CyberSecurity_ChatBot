from django.shortcuts import render
from django.http import JsonResponse
import json

def home(request):
    return render(request, 'home.html')

def ask_question(request):
    # Load the PDF, perform vectorization and embedding, and ask the questions
    questions = [
        "What is your network security strategy?",
        "How do you manage identity and access?",
        "What is your incident response plan?"
        # Add the rest of the questions
    ]
    return JsonResponse({'questions': questions})

def submit_answer(request):
    # Get the answer from the request, score it, and return the result
    answer = request.POST.get('answer')
    domain = request.POST.get('domain')
    score = score_answer(answer, domain)  # Use your scoring function here
    return JsonResponse({'score': score})

def score_answer(answer, domain):
    # Implement your scoring logic (e.g., using TF-IDF, vectorization, etc.)
    # Return the calculated score
    return 8.5  # Example static score
