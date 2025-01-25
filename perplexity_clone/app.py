import streamlit as st
from perplexity_clone.query_generator import generate_queries
from perplexity_clone.knowledge_generator import get_url,extract_html,extract_data
from perplexity_clone.response_generator import generate_response

def main():
    st.title("Perplexity Clone")
    question = st.text_input("Ask a question.")
    if question:
        with st.status("Generating Search Query"):
            query = generate_queries(question)
        with st.status("Fetching Websites"):
            urls = get_url(query)
        with st.status("Generating Knowledge Base"):
            htmls = extract_html(urls)
            data = extract_data(question,htmls)
        with st.status("Genrating Resposne"):
            response = generate_response(question,data)
        st.markdown(response)
if __name__ == "__main__":
    main()
