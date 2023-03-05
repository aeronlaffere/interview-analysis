from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from pydub import AudioSegment
import streamlit as st
import openai

@st.cache_data
def transcribe(audio_file):
    filenames = segment(audio_file)
    transcripts = []
    for filename in filenames:
        audio_file = open(filename, "rb")
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
        transcripts.append(transcript["text"])

    transcript = " ".join(item for item in transcripts)
    return transcript

def segment(audio_file):
    format = "wav" if ".wav" in audio_file.name else "mp3"
    # Segment the audio into 15 minute chunks
    audio_file = AudioSegment.from_file(audio_file, format=format)
    duration = audio_file.duration_seconds / 60
    quarter = 15 * 60 * 1000
    filenames = []
    segment_number = 0
    while duration >= 15:
        segment = audio_file[segment_number * quarter:(segment_number + 1) * quarter]
        segment.export(f"segment_{segment_number}.{format}", format=format)
        filenames.append(f"segment_{segment_number}.{format}")
        duration -= 15
        segment_number += 1
    segment = audio_file[segment_number * quarter:]
    segment.export(f"segment_{segment_number}.{format}", format=format)
    filenames.append(f"segment_{segment_number}.{format}")
    return(filenames)

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

template = """Given the following extracted parts of a long interview and a question, create a final answer with references ("SOURCES"). 
If you don't know the answer, just say that you don't know. Don't try to make up an answer.
ALWAYS return a "SOURCES" part in your answer.
Note that the interview is on a medical topic. Tailor your answer to medical professionals.
Refer to interviews as "the interviewee" or "the interviewee said" or (if describing a question) "the interviewer asked".

QUESTION: {question}
=========
{summaries}
=========
FINAL ANSWER:"""
PROMPT = PromptTemplate(template=template, input_variables=["summaries", "question"])

st.set_page_config(
    page_title="Interview analysis",
    page_icon = "📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_KEY = st.sidebar.text_input("Enter your OpenAI API key:", value="", type="password")
openai.api_key = API_KEY

st.sidebar.write("You need an OpenAI API key to run this demo. You can get one [here](https://platform.openai.com/signup).")
st.sidebar.write("This application was developed by [@aeronjl](https://twitter.com/aeronjl) and is open source. You can find the source code [here](https://github.com/aeronlaffere/interview-analysis).")

st.header("Audio transcription")
st.subheader("Upload an audio interview file for transcription")

audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

if audio_file is not None:
    st.spinner("Transcribing audio...")
    transcript = transcribe(audio_file)

    st.download_button(
        label="Download transcript",
        data=transcript,
        file_name="transcript.txt",
        mime="text/plain"
    )
    
    st.spinner("Getting embeddings...")
    texts, docsearch = get_embeddings(transcript)

    st.subheader("Ask questions about the interview")
    query = st.text_input(label="Enter a question:", value="")

    if query.value != "":
        docs = docsearch.similarity_search(query)

        chain = load_qa_with_sources_chain(OpenAI(temperature=0, openai_api_key=API_KEY), chain_type="stuff", prompt=PROMPT)
        output = chain({"input_documents": docs, "question": query}, return_only_outputs=True)

        response, sources = extract_sources(output["output_text"])

        st.write(response)

        st.write("### Sources")

        for source in sources:
            with st.container():
                st.spinner("Locating sources...")
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                            {"role": "system", "content": """
                            You are a helpful assistant. You are an expert in medical science assisting other experts.
                            You will be given a sample from an interview about a medical topic. You should summarise what was said in the sample in two or three sentences.
                            It is very important that you keep the summary brief, no more than three sentences. Do not make things up. Accuracy is extremely important.
                            """},
                            {"role": "user", "content": texts[source]},
                        ]
                    )
                st.write(response["choices"][0]["message"]["content"])
                with st.expander("Source"):
                    st.write(texts[source])