# model.py

import os
import openai
import tempfile
from pdf2image import convert_from_bytes
from PIL import Image
import whisper
from sentence_transformers import SentenceTransformer
import numpy as np
from pydub import AudioSegment

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

class MultiModalModel:
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.whisper_model = whisper.load_model("small")
    

    def transcribe_audio(self, audio_file):
        # Read uploaded MP3
        audio = AudioSegment.from_file(audio_file, format="mp3")

        # Save it as a temp WAV file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            audio.export(temp_audio.name, format="wav")
            result = self.whisper_model.transcribe(temp_audio.name)
        os.unlink(temp_audio.name)

        return result["text"]


    def extract_text_from_pdf(self, pdf_file):
        images = convert_from_bytes(pdf_file.read())
        text = ""
        for image in images:
            temp_image = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            try:
                image.save(temp_image.name)
                temp_image.close()  # <<< important: close it after saving!
                extracted_text = self.ocr_image(temp_image.name)
                text += extracted_text + "\n"
            finally:
                os.unlink(temp_image.name)


    def ocr_image(self, image_path):
        import pytesseract
        return pytesseract.image_to_string(Image.open(image_path))

    def transcribe_audio(self, audio_file):
        # Create temp file and write audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(audio_file.read())
            temp_audio_path = temp_audio.name  # Save the path

        # Now temp_audio is closed, safe to use
        result = self.whisper_model.transcribe(temp_audio_path)

        # After transcribing, safely delete the temp file
        os.unlink(temp_audio_path)

        return result["text"]


    def create_embeddings(self, texts):
        return self.embedding_model.encode(texts)

    def answer_query(self, pdf_file, audio_file, query):
        if pdf_text is None:
            pdf_text = ""  # or handle it differently based on your use case
        if audio_text is None:
            audio_text = ""  # or handle it differently based on your use case

        combined_text = pdf_text + "\n" + audio_text

        # Step 3: Create embeddings
        documents = combined_text.split("\n")
        document_embeddings = self.create_embeddings(documents)
        query_embedding = self.create_embeddings([query])[0]

        # Step 4: Find most similar chunk
        similarities = np.dot(document_embeddings, query_embedding)
        best_idx = np.argmax(similarities)
        best_chunk = documents[best_idx]

        # Step 5: Final answer generation
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Based on the following content: {best_chunk}\nAnswer the query: {query}"}
            ]
        )

        return response['choices'][0]['message']['content']

