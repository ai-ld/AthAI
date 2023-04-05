import openai
import streamlit as st
import streamlit.components.v1 as components

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
    'Which communication do you want to create?',
    ('Email', 'Press Release', 'Social Media', 'Speech Writing'))

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

# Email tab
if tabs == 'Email':
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
elif tabs == 'Press Release':
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
elif tabs == 'Social Media':
    social_media = st.selectbox(
        'Which social media platform?',
        ('Twitter', 'Facebook', 'Instagram', 'LinkedIn'))
    post_content = st.text_area("Post content:")

    if st.button(label="Generate Social Media Post"):
        try:
            output = generic_completion("Generate an engaging and concise social media post for a political campaign "
                                        "to be published on " + social_media + ". The post should include the "
                                        "following content: " + post_content)
            st.write("```")
            st.write(output)
            st.write("```")
            if social_media == 'Twitter':
                components.html(
                    f"""
                        <
