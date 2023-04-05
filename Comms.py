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
    st.subheader('Generate Social Media Content')
    
    platform = st.selectbox('Select Social Media Platform', ['Twitter', 'Facebook', 'Instagram', 'LinkedIn'])

    if platform == 'Twitter':
        st.subheader('Twitter Post')
        tweet_input = st.text_area("Enter your tweet idea:", max_chars=280)
        if st.button(label="Generate Twitter Post"):
            generated_tweet = gpt_generate_text(tweet_input)
            st.write(generated_tweet)

    elif platform == 'Facebook':
        st.subheader('Facebook Post')
        fb_post_input = st.text_area("Enter your Facebook post idea:")
        if st.button(label="Generate Facebook Post"):
            generated_fb_post = gpt_generate_text(fb_post_input)
            st.write(generated_fb_post)

    elif platform == 'Instagram':
        st.subheader('Instagram Caption')
        insta_caption_input = st.text_area("Enter your Instagram photo description or theme:")
        if st.button(label="Generate Instagram Caption"):
            generated_insta_caption = gpt_generate_text(insta_caption_input)
            st.write(generated_insta_caption)

    elif platform == 'LinkedIn':
        st.subheader('LinkedIn Post')
        li_post_input = st.text_area("Enter your LinkedIn post idea:")
        if st.button(label="Generate LinkedIn Post"):
            generated_li_post = gpt_generate_text(li_post_input)
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

