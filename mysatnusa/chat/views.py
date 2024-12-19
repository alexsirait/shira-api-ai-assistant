import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from google.cloud import dialogflow_v2 as dialogflow
import os
from pathlib import Path
import google.generativeai as genai
from transformers import BertTokenizer, BertForSequenceClassification
import torch

current_dir = Path(__file__).parent
credentials_path = current_dir / "credentials" / "alex-sirait-p9cn-cb8b06ed7633.json"

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(credentials_path)

# Pre-trained BERT model for sentence classification
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# Define static data
DATA_SATNUSA = """
    kamu adalah TechFusion, asisten pintar berbasis AI yang dirancang untuk membantu Pak Ali Sadikin dalam melakukan presentasi di hadapan para top management Satnusa.
    Kamu memiliki kemampuan analisis data, visualisasi, dan interaksi secara real-time untuk menjawab semua pertanyaan yang relevan.

    **Informasi Utama Perusahaan:**
    - Nama Perusahaan: Satnusa
    - Tanggal Berdiri: 10 Januari 2002
    - CEO: Bapak Alex Sirait
      - Lokasi: Medan Deli Serdang
      - Jumlah Anak: 3
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

# Maintain a session history for each user (or session)
conversation_history = {}

# Define function to check relevance using BERT
def is_relevant_to_satnusa(prompt):
    try:
        # Tokenize and encode the prompt
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=512)
        
        # Get predictions from the model
        with torch.no_grad():
            logits = model(**inputs).logits
        prediction = torch.argmax(logits, dim=-1)
        
        # Return True if the prediction is relevant (class '1')
        return prediction.item() == 1
    except Exception as e:
        print(f"Error in relevance detection: {e}")
        return False

# Define the API endpoint for prompt handling
@csrf_exempt
def gemini_prompt_view(request):
    if request.method == "POST":
        try:
            # Parse the request body for the prompt
            data = json.loads(request.body)
            prompt = data.get("prompt", "")
            user_id = data.get("user_id", "default_user")  # Use user_id to identify the user

            if not prompt:
                return JsonResponse({"error": "Prompt is required."}, status=400)

            # Initialize conversation history for the user if not present
            if user_id not in conversation_history:
                conversation_history[user_id] = []

            # Append current prompt to the conversation history
            conversation_history[user_id].append(f"User: {prompt}")

            # Check if the prompt is relevant to DATA_SATNUSA
            is_satnusa_relevant = is_relevant_to_satnusa(prompt)

            # For general knowledge queries, handle it without DATA_SATNUSA context
            if not is_satnusa_relevant:
                # Create the conversation history string
                history = "\n".join(conversation_history[user_id])
                full_prompt = f"{history}\nAssistant:"

                # Directly generate the response for general knowledge
                genai.configure(api_key="AIzaSyCz6r6myd9wS6iB64x_6XIVPmqJVMv2PB4")
                model = genai.GenerativeModel("gemini-1.5-flash")
                response = model.generate_content(full_prompt)
                
                # Store the assistant's response in the conversation history
                conversation_history[user_id].append(f"Assistant: {response.text}")

                # Return the general knowledge response
                return JsonResponse({"response": response.text})

            # For Satnusa-related queries, add DATA_SATNUSA context
            conversation_history_text = '\n'.join(conversation_history[user_id])
            full_prompt = f"{DATA_SATNUSA}\n{conversation_history_text}\nAssistant:"

            # Call Gemini API to generate content based on the prompt
            genai.configure(api_key="AIzaSyCz6r6myd9wS6iB64x_6XIVPmqJVMv2PB4")
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(full_prompt)

            # Store the assistant's response in the conversation history
            conversation_history[user_id].append(f"Assistant: {response.text}")
            
            return JsonResponse({"response": response.text})
        
        except Exception as e:
            return JsonResponse({"error": f"An error occurred: {str(e)}"}, status=500)

    return JsonResponse({"error": "Invalid request method."}, status=405)