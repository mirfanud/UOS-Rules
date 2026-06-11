import streamlit as st

from qa import ask_question

st.set_page_config(
    page_title="UOS Rules Assistant",
    layout="wide"
)

st.title("🎓 UOS Rules Assistant")

question = st.chat_input(
    "Ask a question..."
)

if question:

    with st.spinner("Searching regulations..."):

        answer, docs = ask_question(question)

    st.markdown("## Answer")

    st.write(answer)

    st.markdown("## Sources")

    displayed = set()

    for doc in docs:

        source = doc.metadata["source"]

        if source not in displayed:

            displayed.add(source)

            with st.expander(source):

                st.write(doc.page_content[:1500])
