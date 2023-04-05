import openai
import streamlit as st
import streamlit.components.v1 as components

from langchain.document_loaders import WebBaseLoader, PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# Set up page configuration
st.set_page_config(page_title="Communications 📣", page_icon=":mega:", layout="wide")

# Set up page header and subheader
st.title("Political Campaign AI-powered Communication Tool")
st.subheader("A custom app for generating communications")

# Set API key
openai.api_key = st.secrets["oai-key"]

# Description and information
st.write("This tool generates communications for political campaigns using OpenAI's GPT-3 service. "
         "Please enter as much information as you can, and GPT will handle the rest.\n\n"
         "Note: GPT-3 might generate incorrect information, so editing output is still necessary. "
         "This is a demo with limitations.")

# Create a tab selection
tabs = st.selectbox(
    'Which communication do you want to create? 📄',
    ('Email 📧', 'Press Release 📰', 'Social Media 📲', 'Speech Writing 🎙️'))

# Function to generate content using GPT
def generic_completion(prompt):
    completions = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.85
    )
    message = completions['choices'][0]['message']['content']
    return message.strip()

# Function to generate a tweet
def tweet(output):
    return generic_completion(
        "Generate a tweet summarizing the following text. "
        "Make it engaging and concise: " + output)

# Function to load PDFs and websites
def load_data(source_type, source_path):
    if source_type == "Website":
        loader = WebBaseLoader(source_path)
        data = loader.load()
    elif source_type == "PDF":
        loader = PyPDFLoader(source_path)
        data = loader.load_and_split()
    return data

# Function to find similar documents
def find_similar_documents(texts, query):
    embeddings = OpenAIEmbeddings()
    docsearch = Chroma.from_texts(texts, embeddings)
    similar_docs = docsearch.similarity_search(query)
    return similar_docs

# Email tab
if tabs == 'Email 📧':
    subject = st.text_input("Email subject:")
    recipient = st.text_input("Recipient:")
    details = st.text_area("Email details:")

    if st.button(label="Generate Email"):
        try:
            output = generic_completion("Generate a well-written and engaging email for a political campaign. "
                                        "The email is to be sent to " + recipient + " with the subject " + subject +
                                        ". The email should include the following details: " + details)
            st.write("```")
            st.write(output)
            st.write("```")
        except:
            st.write("An error occurred while processing your request.")

# Press Release tab
elif tabs == 'Press Release 📰':
    headline = st.text_input("Press release headline:")
    subheadline = st.text_input("Press release subheadline:")
    body = st.text_area("Press release content:")

    if st.button(label="Generate Press Release"):
        try:
            output = generic_completion("Generate a compelling press release for a political campaign. "
                                        "The headline is: " + headline + ", and the subheadline is: " + subheadline +
                                        ". The press release should include the following content: " + body)
            st.write("```")
            st.write(output)
            st.write("```")
        except:
            st.write("An error occurred while processing your request.")

# Social Media tab
elif tabs == 'Social Media 📲':
    st.subheader('Generate Social Media Content 🎉')
    
    platform = st.selectbox('Select Social Media Platform', ['Twitter 🐦', 'Facebook 👍', 'Instagram 📷', 'LinkedIn 🔗'])

    if platform == 'Twitter 🐦':
        st.subheader('Twitter Post')
        category = st.selectbox("Choose a topic:", ["Campaign Announcement 📢", "Policy Position 📚", "Event Invitation 🎟️", "Fundraising 💰"])
        if st.button(label="Generate Twitter Post"):
            prompt = "Generate an engaging tweet for a political campaign on the topic: " + category
            generated_tweet = generic_completion(prompt)
            st.write(generated_tweet)

    elif platform == 'Facebook 👍':
        st.subheader('Facebook Post')
        category = st.selectbox("Choose a topic:", ["Campaign Announcement 📢", "Policy Position 📚", "Event Invitation 🎟️", "Fundraising 💰"])
        if st.button(label="Generate Facebook Post"):
            prompt = "Generate an engaging Facebook post for a political campaign on the topic: " + category
            generated_fb_post = generic_completion(prompt)
            st.write(generated_fb_post)

    elif platform == 'Instagram 📷':
        st.subheader('Instagram Caption')
        category = st.selectbox("Choose a topic:", ["Campaign Announcement 📢", "Policy Position 📚", "Event Invitation 🎟️", "Fundraising 💰"])
        if st.button(label="Generate Instagram Caption"):
            prompt = "Generate a captivating Instagram caption for a political campaign on the topic: " + category
            generated_insta_caption = generic_completion(prompt)
            st.write(generated_insta_caption)

    elif platform == 'LinkedIn 🔗':
        st.subheader('LinkedIn Post')
        category = st.selectbox("Choose a topic:", ["Campaign Announcement 📢", "Policy Position 📚", "Event Invitation 🎟️", "Fundraising 💰"])
        li_post_input = st.text_area("Enter your LinkedIn post idea:")
        if st.button(label="Generate LinkedIn Post"):
            generated_li_post = gpt_generate_text(category + li_post_input)
            st.write(generated_li_post)


elif communication == 'Speech Writing':
    prompt_speech = st.text_area("What should the speech be about?")
    if st.button(label="Generate Speech"):
        try:
            st.write("```")
            output = generic_completion("Generate a speech for the political campaign related to the topic: " + prompt_speech)
            st.write(output)
            st.write("```")
        except:
            st.write("An error occurred while processing your request.")

                  # Additional section for loading PDFs and websites
st.subheader("Load PDFs and Websites")
source_type = st.selectbox("Select Source Type", ["Website", "PDF"])
source_path = st.text_input("Enter the URL or path of the source")

loaded_data = False

if st.button(label="Load Source"):
    try:
        data = load_data(source_type, source_path)
        loaded_data = True
        st.write("Data loaded successfully!")
    except:
        st.write("An error occurred while loading the data. Please check the source type and path.")

# Use the loaded data to generate communication posts
if loaded_data:
    similar_docs = find_similar_documents(data, category)
    
    if similar_docs:
        prompt += " using the following related information: " + " ".join(similar_docs)
        generated_communication = generic_completion(prompt)
        st.write(generated_communication)
