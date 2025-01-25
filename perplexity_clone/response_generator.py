from google.generativeai import configure, GenerativeModel
from dotenv import load_dotenv
from os import getenv
from random import choices
from json import dumps
#if __name__ == "__main__":
load_dotenv()
configure(api_key=getenv("GEMINI_API_KEY"))

system_instruction = """
Input Format
You will be provided with:

A knowledge base in JSON format containing an array of documents, where each document has:

url: The source URL of the document
data: The document content in markdown format


A question that needs to be answered

Task
Answer the given question using ONLY the provided knowledge base documents. Your response must:

Be formatted in markdown
Include relevant quotes from the knowledge base when appropriate
Cite sources by including the URL(s) of the knowledge base documents used
Clearly state if the question cannot be answered using the provided knowledge base

Example Input
jsonCopy{
  "knowledge_base": [
    {
      "url": "https://example.com/doc1",
      "data": "# Solar System\nThe solar system consists of the Sun and all celestial objects that orbit it..."
    },
    {
      "url": "https://example.com/doc2",
      "data": "# Mars\nMars is the fourth planet from the Sun and is often called the Red Planet..."
    }
  ],
  "question": "Why is Mars red in color?"
}
Example Output
Mars gets its distinctive red color from iron oxide (rust) on its surface. As stated in our knowledge base:

Mars is often called the Red Planet...

Source: https://example.com/doc2
Response Rules

Only use information explicitly stated in the provided knowledge base
If you cannot find relevant information, respond with:
"I cannot answer this question based on the provided knowledge base."
If you find partial information, clearly indicate what aspects of the question you can and cannot answer
Always cite the source URLs for any information used
Maintain clear markdown formatting throughout your response
Use blockquotes (>) when directly quoting from the knowledge base
If multiple sources are used, cite all relevant URLs

Format Your Answer As

Direct answer to the question
Supporting quotes from knowledge base (if applicable)
Source citations
Any caveats or limitations about the answer

Remember: Never make assumptions or include information not present in the provided knowledge base.
"""

generation_config = {
    "temperature": 0,
    "top_p": 1.0,
    "top_k": 40,
    "max_output_tokens": 5000,
    "response_mime_type": "text/plain",
}
model = GenerativeModel(
    model_name="gemini-1.5-pro",
    generation_config=generation_config,
    system_instruction=system_instruction,
)

chat_session = model.start_chat(history=[])


def generate_response(question: str,data:list[dict[str,...]]) -> str:
    input = {"knowledge_base":data,question:question}
    return chat_session.send_message(dumps(input)).text