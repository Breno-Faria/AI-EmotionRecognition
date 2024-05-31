import base64
import os
import requests


# OPENAI_API_KEY="sk-proj-ALbND6wQU00lHOo6P7NHT3BlbkFJlDpnztRQjB9bKmI47KQJ"
#  peter
# f"Bearer {api_key}"
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
  
def getResponse(image_data_b64):
    api_key = os.getenv("OPENAI_API_KEY")

    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer sk-proj-ALbND6wQU00lHOo6P7NHT3BlbkFJlDpnztRQjB9bKmI47KQJ"
    }

    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "I need you to identify the emotion in the picture. Please do not say anything but the emotion itself. Your answer should be one word only, and that word is the emotion, nothing else. The possible emotions are neutral or engaged. The description for both are: Neutral: A student displaying a state of calm attentiveness. This is characterized by a relaxed facial expression, eyes directed towards the speaker or material, and an overall composed demeanor. This state indicates neither strong engagement nor disinterest, but rather a baseline level of focus. Engaged: A student demonstrating clear signs of active involvement and concentration. This is marked by direct eye contact with the teaching material or speaker, and facial expressions that indicate interest, such as an attentive gaze and a forward-leaning posture that suggests engagement with the content being presented. If you are unsure if they are neutral or engaged, have a slightly bias towards engaged. Remember, extremely small bias. Only if you are very unsure. Make sure the bias is not too big, and make sure to reply with only the emotion and nothing else (neutral or engaged)"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_data_b64}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 300
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    return response.json()

