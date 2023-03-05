from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.llms import OpenAI
from pydub import AudioSegment
import streamlit as st
import openai

API_KEY = "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"
openai.api_key = API_KEY

@st.cache_data
def transcribe(audio_file):
    audio_file = AudioSegment.from_file(audio_file)
    quarter = 25 * 60 * 1000

    first_quarter = audio_file[:quarter]
    first_quarter.export("first_quarter.wav", format="wav")

    second_quarter = audio_file[quarter:quarter * 2]
    second_quarter.export("second_quarter.wav", format="wav")

    third_quarter = audio_file[quarter * 2:quarter * 3]
    third_quarter.export("third_quarter.wav", format="wav")

    fourth_quarter = audio_file[quarter * 3:]
    fourth_quarter.export("fourth_quarter.wav", format="wav")

    transcripts = []
    for filename in ["first_quarter.wav", "second_quarter.wav", "third_quarter.wav", "fourth_quarter.wav"]:
        audio_file = open(filename, "rb")
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
        transcripts.append(transcript["text"])

    transcript = " ".join(item for item in transcripts)
    return transcript

@st.cache_resource
def get_embeddings(transcript):
    from langchain.embeddings.openai import OpenAIEmbeddings
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.vectorstores import Chroma

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=50)
    texts = text_splitter.split_text(transcript)
    embeddings = OpenAIEmbeddings(openai_api_key=API_KEY)
    docsearch = Chroma.from_texts(texts, embeddings, metadatas=[{"source": str(i)} for i in range(len(texts))])
    return texts, docsearch

def extract_sources(output):
    import re

    response = output.split("\nSOURCES: ")[0]

    source_pattern = r"\d+"
    sources = re.findall(source_pattern, output)
    sources = [int(s) for s in sources]

    return response, sources

st.header("Audio transcription")
st.subheader("Upload an audio interview file for transcription")

audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

if audio_file is not None:
    st.spinner("Transcribing audio...")
    transcript = transcribe(audio_file)

    st.spinner("Getting embeddings...")
    texts, docsearch = get_embeddings(transcript)

    st.subheader("Ask questions about the interview")
    query = st.text_input(label="Enter a question:", value="What is the attitude to vaccination?")
    docs = docsearch.similarity_search(query)

    chain = load_qa_with_sources_chain(OpenAI(temperature=0, openai_api_key=API_KEY), chain_type="stuff")
    output = chain({"input_documents": docs, "question": query}, return_only_outputs=True)

    response, sources = extract_sources(output["output_text"])

    st.write(response)

    for source in sources:
        with st.container():
            with st.expander("Source"):
                st.write(texts[source])