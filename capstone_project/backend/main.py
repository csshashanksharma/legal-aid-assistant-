from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import whisper

app = FastAPI()

# Load Whisper model once (base is fast; use "small", "medium", or "large" for accuracy)
model = whisper.load_model("base")

class Query(BaseModel):
    text: str

@app.post("/query/")
def get_query(user_query: Query):
    return {"received_query": user_query.text}

@app.post("/voice/")
async def voice_input(file: UploadFile = File(...)):
    try:
        audio_path = f"temp_{file.filename}"
        with open(audio_path, "wb") as buffer:
            buffer.write(await file.read())

        # Debug: print the saved file path
        print(f"Saved file: {audio_path}")

        # Transcribe
        result = model.transcribe(audio_path)
        return {"transcribed_text": result["text"]}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}
