import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from google.cloud import dialogflow_v2 as dialogflow
import os
from pathlib import Path

current_dir = Path(__file__).parent
credentials_path = current_dir / "credentials" / "alex-sirait-p9cn-cb8b06ed7633.json"

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(credentials_path)

@csrf_exempt
def chatbot_api(request):
    if request.method == "POST":
        try:
            body = json.loads(request.body.decode("utf-8"))
            message = body.get("message", "")
            
            if message == "":
                return JsonResponse({"error": "message is required"}, status=500)

            session_client = dialogflow.SessionsClient()
            session = session_client.session_path("alex-sirait-p9cn", "unique-session-id")

            text_input = dialogflow.TextInput(text=message, language_code="id")
            query_input = dialogflow.QueryInput(text=text_input)
            response = session_client.detect_intent(request={"session": session, "query_input": query_input})

            ai_reply = response.query_result.fulfillment_text

            return JsonResponse({"reply": ai_reply}, status=200)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    else:
        return JsonResponse({"error": "Invalid request method"}, status=405)
