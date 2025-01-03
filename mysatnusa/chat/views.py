import json
import logging
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from pathlib import Path
import os
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from concurrent.futures import ThreadPoolExecutor
import google.generativeai as genai
from spacy import load

# Logging Configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Configuration for Credentials and Google Generative AI
current_dir = Path(__file__).parent
credentials_path = current_dir / "credentials" / "alex-sirait-p9cn-cb8b06ed7633.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(credentials_path)

# Load Pre-trained BERT Model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2).to(DEVICE)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Load SpaCy for NER (Named Entity Recognition)
nlp = load("en_core_web_sm")

# Static Data Knowledge Base (Text-Based)
DATA_SATNUSA = """
    Kamu adalah TechFusion, asisten pintar berbasis AI yang dirancang untuk membantu Pak Ali Sadikin dalam melakukan presentasi di hadapan para top management Satnusa.
    Kamu harus bisa menjawab setiap pertanyaan dari Pak Ali dengan Singkat Padat sesuai dengan pertanyaan Pak Ali langsung pada intinya saja tidak perlu berpoin poin kamu langsung memberikan kesimpulan atau rangkuman yang sangat baik sehingga tidak perlu banyak hal tapi intinya saja tetapi jangan terlalu singkat sekali minimal 1 kalimat.

    **Informasi Utama Perusahaan:**
    - Nama Perusahaan: Satnusa
    - Tanggal Berdiri: 10 Januari 2002
    - CEO: Bapak Alex Sirait
      - Lokasi: Medan Deli Serdang
      - Jumlah Anak: 3
      - Foto CEO : https://webapi.satnusa.com/EmplFoto/000001.JPG
      - Canva CEO : (http://127.0.0.1:8000/media/chat/public/PPTVue.jsxDjango.pdf)
    - Divisi Terbesar: DOT (Department of Technology)
      - Kepala Divisi: Pak Ali Sadikin (lahir 06 Desember 2000)
      - Fungsi: Inovasi teknologi dan transformasi digital perusahaan.
    - Manager Umum: Abidin
    - Tim Pengembang:
      - Backend Developer: Ghanni
      - Frontend Developer: Fatur

    **Informasi Pak Ali Sadikin:**
    - Jabatan: Head of DOT
    - Keahlian Utama: 
        - Blockchain dan Bitcoin Investment
        - Transformasi Digital
        - Manajemen Tim Teknologi
    - Visi: Menjadikan DOT sebagai penggerak utama transformasi digital Satnusa.
    - Hobi: Membaca tentang teknologi mutakhir, analisis pasar keuangan.
    
    **Tim Pendukung Pak Ali:**
    - Nama Asisten Utama: Dina Pratama
      - Jabatan: Business Analyst
      - Pendidikan: S1 Teknik Informatika, Universitas Indonesia (Cumlaude)
      - Pengalaman: Membantu 20+ presentasi eksekutif, pemenang "Best Analyst 2023".
      - Kontak: dina.pratama@satnusa.com | HP: +62 812 3456 7890

    **Data Keuangan Perusahaan:**
    - Pendapatan Tahunan (2023): Rp 2 Triliun
    - Pertumbuhan Tahunan: 12%
    - Investasi Teknologi (2023): Rp 500 Miliar
    - Kontribusi DOT ke Pendapatan: 40%

    **Data Operasional DOT:**
    - Total Karyawan: 120 orang
    - Proyek Aktif: 15 (5 proyek besar dan 10 proyek inovasi)
    - Kontribusi ke Efisiensi Operasional: Meningkatkan efisiensi hingga 25% dalam 2 tahun terakhir.
    
    **Fitur TechFusion untuk Presentasi:**
    - **Real-Time Analytics:** Dapat menganalisis dan menjawab pertanyaan langsung dari data yang ada.
    - **Natural Language Processing (NLP):** Memahami dan merespons pertanyaan dengan bahasa alami.
    - **Visualisasi Data:** Membuat grafik, bagan, dan peta strategi secara otomatis.
    - **Simulasi dan Prediksi:** Memberikan prediksi berbasis data tentang dampak proyek DOT terhadap kinerja perusahaan.
    - **Interaktif:** Mengakomodasi permintaan mendadak untuk menambahkan, mengubah, atau menghapus data selama presentasi.

    **Skenario Umum Presentasi:**
    1. **Perkenalan DOT:**
       - Visi dan Misi DOT.
       - Kontribusi terhadap strategi perusahaan.
    2. **Proyek Unggulan:**
       - Proyek Transformasi Digital (2023).
       - Inovasi Blockchain di bidang manufaktur.
    3. **Analisis Keuangan:**
       - Efisiensi biaya melalui proyek teknologi.
       - ROI proyek besar.
    4. **Q&A Interaktif:**
       - TechFusion siap menjawab pertanyaan manajemen tentang data keuangan, strategi, atau proyek.
       - Memberikan prediksi berbasis simulasi untuk skenario bisnis.

    **FAQ Umum:**
    - *Berapa kontribusi DOT ke perusahaan?* → 40% dari total pendapatan tahunan.
    - *Apa target utama DOT tahun ini?* → Implementasi penuh transformasi digital dalam rantai pasok.
    - *Apa inovasi terbaru?* → Proyek blockchain untuk efisiensi manufaktur.
    - *Bagaimana pertumbuhan keuangan DOT dalam 5 tahun terakhir?* → Pertumbuhan rata-rata 15% per tahun.

    **Prediksi AI untuk Presentasi:**
    - Memungkinkan Pak Ali fokus pada interaksi, sementara TechFusion menangani visualisasi dan pertanyaan.
    - Simulasi dampak strategi baru memberikan nilai tambah dalam pengambilan keputusan.
    - Mengurangi waktu persiapan hingga 50%.
"""

# Conversation History
conversation_history = {}

# Dynamic Relevance Threshold
RELEVANCE_THRESHOLD = 0.7

# Helper Functions
def extract_entities(text):
    """
    Extract key entities from the text using SpaCy's NER model.
    """
    doc = nlp(text)
    entities = {ent.text: ent.label_ for ent in doc.ents}
    return entities

def search_knowledge_base(query, relevant_entities=None):
    """
    Search in the local knowledge base (DATA_SATNUSA) for relevant information, prioritizing entities.
    """
    query_lower = query.lower()
    
    if relevant_entities:
        # Enhance the search by considering entities
        for entity, label in relevant_entities.items():
            if entity.lower() in DATA_SATNUSA.lower():
                return DATA_SATNUSA  # Return the data with higher relevance
    
    if query_lower in DATA_SATNUSA.lower():
        return DATA_SATNUSA
    else:
        return "No relevant information found in the knowledge base."

def check_relevance_with_confidence(prompt):
    """
    Check the relevance of the prompt using BERT and return confidence score.
    """
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=512).to(DEVICE)
        with torch.no_grad():
            logits = bert_model(**inputs).logits
        probabilities = torch.softmax(logits, dim=-1)
        relevance_score = probabilities[0, 1].item()
        logging.info(f"Relevance score: {relevance_score}")
        return relevance_score >= RELEVANCE_THRESHOLD, relevance_score
    except Exception as e:
        logging.error(f"Error in relevance detection: {e}")
        return False, 0.0

def append_conversation_history(user_id, prompt, response=None):
    """
    Append user prompt and response to the conversation history.
    """
    if user_id not in conversation_history:
        conversation_history[user_id] = []
    conversation_history[user_id].append(f"User: {prompt}")
    if response:
        conversation_history[user_id].append(f"Assistant: {response}")

def generate_response(prompt, context, relevant_entities):
    """
    Generate a response using Google Generative AI or fallback to the knowledge base (DATA_SATNUSA).
    """
    try:
        # Integrate Gemini AI with fallback to local knowledge base
        genai.configure(api_key="AIzaSyCz6r6myd9wS6iB64x_6XIVPmqJVMv2PB4")
        model = genai.GenerativeModel("gemini-1.5-flash")
        full_prompt = f"{context}\n{prompt}"
        response = model.generate_content(full_prompt)
        
        # Enhance response with relevant entities
        enhanced_response = f"{response.text}\n\nKey Entities: {relevant_entities}"
        return enhanced_response
    except Exception as e:
        logging.warning(f"Google Generative AI failed: {e}. Falling back to local knowledge base response.")
        return search_knowledge_base(prompt, relevant_entities)

@csrf_exempt
def gemini_prompt_view(request):
    if request.method == "POST":
        try:
            # Parse request data
            data = json.loads(request.body)
            prompt = data.get("prompt", "").strip()
            user_id = data.get("user_id", "default_user").strip()

            if not prompt:
                return JsonResponse({"error": "Prompt is required."}, status=400)

            # Extract relevant entities from the prompt
            relevant_entities = extract_entities(prompt)
            
            # Process relevance and response
            is_relevant, relevance_score = check_relevance_with_confidence(prompt)
            context = DATA_SATNUSA if not is_relevant else "\n".join(conversation_history.get(user_id, []))
            
            # Generate response based on context and entities
            response_text = generate_response(prompt, context, relevant_entities)

            # Append conversation history
            append_conversation_history(user_id, prompt, response_text)

            # Log the response
            logging.info(f"Response for user {user_id}: {response_text}")
            return JsonResponse({"response": response_text, "relevance_score": relevance_score})
        except Exception as e:
            logging.error(f"Error processing request: {e}")
            return JsonResponse({"error": "Internal server error"}, status=500)

    return JsonResponse({"error": "Only POST method is allowed."}, status=405)
