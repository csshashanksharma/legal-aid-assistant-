from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import whisper
from preprocessing import preprocess_for_retrieval, detect_language

app = FastAPI()

# Load Whisper model once (base is fast; use "small", "medium", or "large" for accuracy)
model = whisper.load_model("base")

class Query(BaseModel):
    text: str

@app.post("/query/")
def get_query(user_query: Query):
    cleaned, normalized = preprocess_for_retrieval(user_query.text)
    lang = detect_language(user_query.text)
    return {
        "original_text": user_query.text,
        "cleaned_text": cleaned,
        "normalized_text": normalized,
        "language": lang
    }


@app.post("/voice/")
async def voice_input(file: UploadFile = File(...)):
    try:
        audio_path = f"temp_{file.filename}"
        with open(audio_path, "wb") as buffer:
            buffer.write(await file.read())

        # Transcribe
        result = model.transcribe(audio_path)
        transcribed_text = result["text"]

        # Preprocess transcribed text
        cleaned, normalized = preprocess_for_retrieval(transcribed_text)
        lang = detect_language(transcribed_text)

        return {
            "original_text": transcribed_text,
            "cleaned_text": cleaned,
            "normalized_text": normalized,
            "language": lang
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

