from duckduckgo_search import DDGS
from primp import Client
from bs4 import BeautifulSoup
from google.generativeai import configure, GenerativeModel
from dotenv import load_dotenv
from os import getenv
from random import choices
#if __name__ == "__main__":
load_dotenv()
configure(api_key=getenv("GEMINI_API_KEY"))

system_instruction = """
You are a specialized assistant designed to extract relevant information from HTML content. Your role is to find and extract ONLY the specific information that directly answers the user's question, without summarization or additional context.
Make sure that the extracted content is ALWAYS IN MARKDOWN FORMATAND NOT HTML
Core Function

Extract exact relevant content that directly answers the question
Convert to Markdown without altering the original content
Exclude any information that doesn't directly address the question
Maintain the original wording where possible

Extraction Rules
Direct Matching
Extract content that directly addresses the question precisly
Include complete sentences/paragraphs that contain the answer
Exclude surrounding context

Relevance Filtering

Remove content that:
Only provides background information
Offers additional examples not asked for
Explains related but not directly requested information
Gives supplementary details beyond the scope of the question

Content Boundaries
Extract only within the semantic boundaries of the answer
Stop extraction when topic shifts away from the question
Include full data points without breaking mid-explanation

Markdown Conversion Rules
Convert HTML tags to Markdown:
<h1>-<h6> → # (appropriate level)
<ul>/<ol> → Markdown lists
<strong>/<b> → **text**
<em>/<i> → *text*
<table> → Markdown tables
<a> → [text](url)



Response Format:
[Exact extracted content in Markdown]

Extraction Examples
Question: "What is the price of the product?"
Good Response:
The XYZ Model costs $499.99

Bad Response (Avoid):
The XYZ Model costs $499.99. It comes with a 2-year warranty and free shipping. This competitive price point makes it an excellent value compared to similar products.
(Bad because it includes irrelevant details about warranty and comparisons)

Error States
If HTML is invalid: Try your level best to extract the data
If no matching content: Just reply with "NO CONTENT FOUND"
"""
generation_config = {
    "temperature": 0,
    "top_p": 1.0,
    "top_k": 40,
    "max_output_tokens": 5000,
    "response_mime_type": "text/plain",
}

def get_url(queries:dict[str,...])->list[str]:
    search = DDGS()
    url = set()
    res = search.text(queries["primary_query"])
    for val in res:
        url.add(val["href"])
    for val in queries["alternative_queries"]:
        res = search.text(val)
        for val in res:
            url.add(val["href"])
    return choices(list(url),k=min(len(url),5))

def extract_html(urls:list[str]) -> list[dict[str,...]]:
    client = Client()
    htmls = []
    for url in urls:
        html = client.get(url)
        htmls.append({"url":url,"data":sanitize_html(html.text)})
    return htmls
def sanitize_html(html:str)->str:
    page = BeautifulSoup(html,"html.parser")
    body = page.find("main") if page.find("main") else page.find("body")
    if body is None:
        return "None"
    for script in body.find_all("script"):
        script.decompose()
    for style in body.find_all("style"):
        style.decompose()
    for header in body.find_all("header"):
        header.decompose()
    for footer in body.find_all("footer"):
        footer.decompose()
    for nav in body.find_all("nav"):
        nav.decompose()
    for svg in body.find_all("svg"):
        svg.decompose()
    # remove all images
    for img in body.find_all("img"):
        img.decompose()
    # remove all forms
    for form in body.find_all("form"):
        form.decompose()
    # remove all buttons
    for button in body.find_all("button"):
        button.decompose()
    # remove all inputs
    for input in body.find_all("input"):
        input.decompose()
    # remove all textareas
    for textarea in body.find_all("textarea"):
        textarea.decompose()
    # replace all anchor tags with their text
    for a in body.find_all("a"):
        if a is not None and a.get("href"):
            a.replace_with(a.get_text())
        else:
            a.decompose()
    for div in body.find_all("div"):
        div.attrs.pop("class",None)
    return str(body)

def extract_data(question:str,htmls:list[dict[str,...]])->list[str]:
    model = GenerativeModel(
    model_name="gemini-2.0-flash-exp",
    generation_config=generation_config,
    system_instruction=system_instruction,
    )
    chat_session = model.start_chat(history=[])
    prompt = """
    Question:-{question}
    HTML DATA:- {html}
    """
    data = []
    for val in htmls:
        ans = chat_session.send_message(prompt.format(question=question,html=val["data"]))
        data.append({"url":val["url"],"date":ans.text})
    return data
