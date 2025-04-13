import streamlit as st
import wikipedia
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import random
import re

# Load model
model = SentenceTransformer('all-MiniLM-L6-V2')

# Define intents
intents = [
     {"tag": "greeting", "patterns": ["Hi", "Hello", "Hey", "How are you?", "What's up?", "Good morning"],
     "responses": ["Hello! How can I assist you today?", "Hey there! Need any help?", "Hi! What can I do for you?", "I'm doing great, thanks for asking! How about you?"]},

    {"tag": "goodbye", "patterns": ["Bye", "See you later", "Goodbye", "Take care"],
     "responses": ["Goodbye! Have a great day!", "See you soon!", "Take care and stay safe!", "Bye! Feel free to chat anytime!"]},

    {"tag": "thanks", "patterns": ["Thank you", "Thanks", "Thanks a lot", "I appreciate it"],
     "responses": ["You're very welcome!", "Happy to help!", "Anytime! Let me know if you need anything else."]},

    {"tag": "about", "patterns": ["What can you do?", "Who are you?", "What is your purpose?", "Tell me about yourself"],
     "responses": ["I'm a chatbot created to assist you with various queries!", "I can answer questions, chat with you, and offer guidance.", "My purpose is to make conversations more engaging and helpful!"]},

    {"tag": "jokes", "patterns": ["Tell me a joke", "Make me laugh", "Do you know any jokes?"],
     "responses": ["Why don't programmers like nature? It has too many bugs!", "I told my computer I needed a break, and now it won’t stop sending me vacation ads!", "Why was the JavaScript developer sad? Because he didn’t ‘null’ his problems!"]},

    {"tag": "weather", "patterns": ["What’s the weather like?", "Is it going to rain today?", "Tell me the weather"],
     "responses": ["I can't check live weather, but you can try a weather app!", "I recommend checking a weather website for real-time updates."]},
         {"tag": "math", "patterns": ["What is 2 + 2?", "Solve 5 * 6", "What is 10 squared?"],
     "responses": ["2 + 2 is 4!", "5 * 6 is 30!", "10 squared is 100!"]},

    {"tag": "science", "patterns": ["What is gravity?", "Tell me about space", "How does the sun produce energy?"],
     "responses": ["Gravity is the force that attracts objects toward each other.", "Space is a vast, empty area beyond Earth's atmosphere.", "The sun produces energy through nuclear fusion!"]},

    {"tag": "history", "patterns": ["Who was the first president of the USA?", "Tell me about World War 2", "When was the moon landing?"],
     "responses": ["The first U.S. president was George Washington.", "World War 2 lasted from 1939 to 1945.", "The first moon landing was in 1969, by Apollo 11."]},

    {"tag": "geography", "patterns": ["What is the capital of France?", "Which is the largest ocean?", "Tell me about Mount Everest"],
     "responses": ["The capital of France is Paris.", "The Pacific Ocean is the largest ocean on Earth.", "Mount Everest is the highest mountain, located in Nepal/Tibet."]},

    {"tag": "technology", "patterns": ["What is AI?", "What is machine learning?", "What is the Internet?"],
     "responses": ["AI stands for Artificial Intelligence, allowing machines to mimic human intelligence.", "Machine learning is a branch of AI that enables systems to learn from data.", "The Internet is a global network that connects computers worldwide."]},
        {"tag": "time", "patterns": ["What time is it?", "Tell me the current time"],
     "responses": ["I don't have a clock, but you can check your phone!", "Try asking Google or checking your watch."]},

    {"tag": "date", "patterns": ["What’s today’s date?", "What day is it?"],
     "responses": ["You can check the date on your device!", "Today's date is easily available on your phone."]},

    {"tag": "motivation", "patterns": ["Give me some motivation", "I need inspiration", "Tell me a motivational quote"],
     "responses": ["Believe in yourself! You are capable of great things.", "The only limit to your success is your mindset!", "Success is not final, failure is not fatal—it is the courage to continue that counts."]},

    {"tag": "health", "patterns": ["How can I stay healthy?", "What are some healthy foods?", "Give me fitness tips"],
     "responses": ["Eat balanced meals, exercise regularly, and get enough sleep!", "Healthy foods include fruits, vegetables, nuts, and lean proteins.", "Stay active, drink plenty of water, and maintain a good sleep schedule."]},
    {"tag": "movies", "patterns": ["Recommend a movie", "What’s a good movie to watch?", "Give me a movie suggestion"],
     "responses": ["Try watching 'Inception' if you like sci-fi!", "If you enjoy action, 'John Wick' is a great choice!", "How about a comedy? 'The Office' is hilarious!"]},

    {"tag": "music", "patterns": ["Recommend a song", "What’s a good song?", "Who is your favorite musician?"],
     "responses": ["Try listening to 'Blinding Lights' by The Weeknd!", "If you like pop, check out Taylor Swift!", "For classical music, Beethoven is a legend!"]},

    {"tag": "sports", "patterns": ["What’s the latest football match result?", "Tell me about cricket", "Who won the NBA championship?"],
     "responses": ["I don't have live updates, but you can check ESPN!", "Cricket is a popular sport in countries like India and England.", "The latest NBA champion can be found on sports websites."]},
    {"tag": "coding", "patterns": ["How do I learn Python?", "What is JavaScript used for?", "Explain HTML"],
     "responses": ["You can start learning Python with online courses like Codecademy!", "JavaScript is used for making interactive web pages.", "HTML is the standard language for creating web pages."]},

    {"tag": "cybersecurity", "patterns": ["What is phishing?", "How can I stay safe online?", "Tell me about hacking"],
     "responses": ["Phishing is a type of cyber attack where attackers trick you into giving personal information.", "Stay safe online by using strong passwords and avoiding suspicious links.", "Hacking refers to gaining unauthorized access to systems. Ethical hacking is used for security testing."]},
    {"tag": "random_fact", "patterns": ["Tell me a fact", "Give me a random fact", "Surprise me!"],
     "responses": ["Did you know honey never spoils?", "A day on Venus is longer than a year on Venus!", "Octopuses have three hearts!"]},

    {"tag": "riddles", "patterns": ["Tell me a riddle", "Give me a puzzle", "Challenge me"],
     "responses": ["What has keys but can't open locks? A piano!", "The more you take, the more you leave behind. What is it? Footsteps!", "I speak without a mouth and hear without ears. I have nobody, but I come alive with the wind. What am I? An echo!"]},

    {"tag": "languages", "patterns": ["How do you say hello in Spanish?", "Translate hello to French"],
     "responses": ["Hello in Spanish is 'Hola'!", "In French, hello is 'Bonjour'!", "In German, hello is 'Hallo'!"]},
    {
        "tag": "wikipedia_query",
        "patterns": ["Tell me about", "What is", "Who is", "Explain", "I want to know about"],
        "responses": []
    }
]

# Embed patterns
patterns, tags = [], []
for intent in intents:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        tags.append(intent['tag'])
pattern_embeddings = model.encode(patterns)

# Wikipedia fetcher
def fetch_wikipedia_summary(query, sentences=3):
    try:
        return wikipedia.summary(query, sentences=sentences)
    except wikipedia.exceptions.DisambiguationError as e:
        return f"⚠️ Your query is ambiguous. Try: {', '.join(e.options[:5])}"
    except wikipedia.exceptions.PageError:
        return "❌ No matching Wikipedia page found."
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

def is_wikipedia_query(user_input):
    keywords = ["what", "who", "where", "when", "why", "how", "capital", "founder", "president", "invention", "explain", "define", "meaning"]
    user_input_lower = user_input.lower()
    return any(kw in user_input_lower for kw in keywords)

def refine_query(user_input):
    user_input = re.sub(r'[^\w\s]', '', user_input)
    user_input = re.sub(r'\b(tell me about|what is|who is|explain|define|meaning of|i want to know about)\b', '', user_input.lower())
    return user_input.strip().title()

def chatbot_response(user_input):
    input_embedding = model.encode([user_input])
    similarity = cosine_similarity(input_embedding, pattern_embeddings)
    best_match_index = similarity.argmax()
    best_match_score = similarity[0][best_match_index]
    predicted_tag = tags[best_match_index]

    if best_match_score > 0.6 and predicted_tag != "wikipedia_query":
        for intent in intents:
            if intent["tag"] == predicted_tag:
                return random.choice(intent["responses"])

    if is_wikipedia_query(user_input):
        refined = refine_query(user_input)
        return fetch_wikipedia_summary(refined)

    return "I'm not sure what you mean. Try rephrasing that."

# Streamlit UI
st.set_page_config(page_title="Wikipedia Chatbot", layout="centered")
st.title("ChatBot Assistant")
st.markdown("Ask me **anything educational**, or say hi!")

# Chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 👉 Display chat history ABOVE the input form
for speaker, message in st.session_state.chat_history:
    if speaker == "You":
        # Display user's message on the right
        st.markdown(f'<div style="text-align: right; color: white;"><strong>{speaker}:</strong> {message}</div>', unsafe_allow_html=True)
    else:
        # Display bot's message on the left
        st.markdown(f'<div style="text-align: left; color: white;"><strong>{speaker}:</strong> {message}</div>', unsafe_allow_html=True)

# Input form BELOW the chat history
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("You:", placeholder="Ask something like 'What is AI?'")
    submitted = st.form_submit_button("Send")

# Handle response and update chat
if submitted and user_input:
    bot_response = chatbot_response(user_input)
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Bot", bot_response))

