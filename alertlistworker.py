import google.generativeai as genai
import os
import dotenv

dotenv.load_dotenv()
try:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
except KeyError:
    print("ERROR: GOOGLE_API_KEY environment variable not set.")
    print("Please set your API key from Google AI Studio to run this example.")
    print("e.g., export GOOGLE_API_KEY='YOUR_API_KEY'")
    exit()


model = genai.GenerativeModel('models/gemini-2.5-pro')

def generate_alert_messages(senario, threshold_dbfs):
    prompt = (
    "Generate 5 alert messages to keep the following place quiet for everyone: {senario}. "
    "The target is to reduce noise levels below {threshold_dbfs} dBFS. "
    "Each message should be concise, clear, and suitable for a text-to-speech system. "
    "Provide the messages in a numbered list format. "
    "Start the response directly with the list (e.g., '1. ...' not 'Here are...'). "
    "These messages will be used in the order you provide them. Please make it pesuasive and varied, avoiding repetition. "
    ) 
    message_for_api = []
    print(prompt.format(senario=senario, threshold_dbfs=threshold_dbfs))
    message_for_api.append({"role": "user", "parts": prompt.format(senario=senario, threshold_dbfs=threshold_dbfs)})
    
    response = model.generate_content(message_for_api)

    
    messages = response.text.strip().split('\n')
    return [msg.split('. ', 1)[1] if '. ' in msg else msg for msg in messages if msg]

if __name__ == "__main__":
    senario = "Coding office/workspace"
    messages = generate_alert_messages(senario)
    for i, msg in enumerate(messages, 1):
        print(f"{i}. {msg}")