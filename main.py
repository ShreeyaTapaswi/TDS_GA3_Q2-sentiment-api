import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY", "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"),
    base_url="https://api.groq.com/openai/v1"
)

class CommentRequest(BaseModel):
    comment: str

@app.post("/comment")
async def analyze_comment(request: CommentRequest):
    if not request.comment.strip():
        raise HTTPException(status_code=400, detail="Comment cannot be empty")
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a sentiment analysis tool. "
                        "Reply with ONLY valid JSON, no extra text, no markdown. "
                        "Format: {\"sentiment\": \"positive\", \"rating\": 5} "
                        "sentiment must be: positive, negative, or neutral. "
                        "rating must be integer 1-5. 5=very positive, 1=very negative."
                    )
                },
                {
                    "role": "user",
                    "content": request.comment
                }
            ],
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"API error: {str(e)}")