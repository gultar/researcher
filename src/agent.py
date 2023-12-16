from openai import OpenAI
from dotenv import load_dotenv
from datetime import date
from typing import Optional

class Agent:
    def __init__(self, role="general", model="gpt-4-1106-preview", smaller_model="gpt-3.5-turbo-16k"):
        # Initialize an empty array for messages
        self.initial_instructions = "You are a helpful assistant."
        self.messages = [{"role": "system", "content": self.initial_instructions}]
        self.role = role
        self.model = model
        self.smaller_model = smaller_model
        
        load_dotenv()

        self.client = OpenAI(
        organization='org-89slefGLSVO0BZUSanFQus9v',
        )

        today = date.today()
        today = str(today)

    def log(self, text):
        return print(text)
    

    def record(self, prompt: str):
        # Add user's prompt to the messages
        self.messages.append({"role": "user", "content": prompt})
        
        self.log(prompt)
        # Use OpenAI API to get response
        response = self.client.chat.completions.create(
            model=self.model,  # Use the desired model
            messages=self.messages
        )

        # Extract reply from the response
        reply = response.choices[0].message.content

        # Add system's reply to the messages
        self.messages.append({"role": "assistant", "content": reply})

        return reply

    def complete(self, prompt: str, temperature: Optional[int]=0):
        # Add user's prompt to the messages
        messages = [{"role": "system", "content": self.initial_instructions},{"role": "user", "content": prompt}]

        self.log(prompt)
        # Use OpenAI API to get response
        response = self.client.chat.completions.create(
            model=self.model,  # Use the desired model
            messages=messages,
            temperature=temperature
        )

        # Extract reply from the response
        reply = response.choices[0].message.content

        return reply
    
    def cheaper_complete(self, prompt: str, temperature: Optional[int]=0):
        # Add user's prompt to the messages
        
        messages = [{"role": "system", "content": self.initial_instructions},{"role": "user", "content": prompt}]
        self.log(prompt)
        # Use OpenAI API to get response
        response = self.client.chat.completions.create(
            model=self.smaller_model,  # Use the desired model
            messages=messages
        )

        # Extract reply from the response
        reply = response.choices[0].message.content

        return reply

