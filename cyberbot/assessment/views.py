from django.shortcuts import render
from django.http import HttpResponse
from django.conf import settings
from typing import List
import os
from django.shortcuts import redirect, reverse
import time
import logging
from .ml_utils import process_query

logger = logging.getLogger(__name__)

#from django.shortcuts import render
from .utils import process_pdf_files
from .ml_utils import initialize_models

def upload_pdf(request):
    if request.method == 'POST':
        pdf_file = request.FILES.get('pdf_file')
        if not pdf_file:
            logger.error("No PDF file uploaded")
            return HttpResponse("No file uploaded", status=400)
        
        if not pdf_file.name.endswith('.pdf'):
            logger.error("Uploaded file is not a PDF")
            return HttpResponse("Please upload a PDF file", status=400)
        
        try:
            upload_dir = settings.PDF_DIR
            if not os.path.exists(upload_dir):
                os.makedirs(upload_dir)
            
            file_path = os.path.join(upload_dir, pdf_file.name)
            with open(file_path, 'wb+') as destination:
                for chunk in pdf_file.chunks():
                    destination.write(chunk)
            
            logger.info(f"Successfully uploaded PDF: {pdf_file.name}")
            
            # Process the uploaded file
            pdf_files = process_pdf_files([file_path])
            
            # Initialize models with the uploaded file
            initialize_models()
            logger.info("Models initialized with uploaded PDF")
            
            # Redirect to query endpoint
            return redirect(reverse('query_api'))
            
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            return HttpResponse(f"Error: {str(e)}", status=500)
    
    return render(request, 'upload.html')
def query_view(request):
    """Handle user queries and display chat history."""
    # Initialize chat history in session
    if 'chat_history' not in request.session:
        request.session['chat_history'] = []

    if request.method == 'POST':
        query = request.POST.get('query', '').strip()
        if not query:
            logger.warning("Empty query received")
            return render(request, 'index.html', {'error': 'Please enter a query'})

        # Security: Validate input length
        if len(query) > 1000:
            logger.warning(f"Query too long: {len(query)} characters")
            return render(request, 'index.html', {'error': 'Query exceeds maximum length'})

        try:
            start_time = time.time()
            chat_history = request.session['chat_history']
            answer = process_query(query, chat_history)
            execution_time = time.time() - start_time

            # Update chat history (limit to 10 entries for performance)
            chat_history.append({"question": query, "answer": answer})
            request.session['chat_history'] = chat_history[-10:]
            logger.info(f"Query processed in {execution_time:.3f} seconds")

            return render(request, 'index.html', {
                'query': query,
                'answer': answer,
                'execution_time': f"{execution_time:.3f} seconds",
                'chat_history': chat_history
            })
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return render(request, 'index.html', {'error': 'An error occurred. Please try again.'})

    return render(request, 'index.html', {'chat_history': request.session['chat_history']})