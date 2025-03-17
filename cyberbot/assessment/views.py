from django.shortcuts import render
import time
import logging
from .ml_utils import process_query

logger = logging.getLogger(__name__)

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