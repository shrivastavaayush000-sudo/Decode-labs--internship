"""
Project 1: Rule-Based AI Chatbot
DecodeLabs Industrial Training Kit - Artificial Intelligence (Batch 2026)

Meet "Byte", a friendly study-buddy bot for coding interns.

How it works (IPO model):
  INPUT    -> read the message, then clean it (lowercase, strip spaces, remove punctuation)
  PROCESS  -> match it to an intent using dictionaries (O(1) lookup, no long if-elif ladder)
  OUTPUT   -> print a reply, or a fallback if nothing matches
"""

import random
import string
from datetime import datetime

BOT_NAME = "Byte"

# ---------------------------------------------------------------
# 1. KNOWLEDGE BASE: intent -> list of possible replies
#    A reply can also be a function, so it is computed when used.
# ---------------------------------------------------------------
RESPONSES = {
    "greeting": [
        "Hey there! I'm Byte, your coding study buddy.",
        "Hello! Ready to talk code?",
        "Hi! Great to see you.",
    ],
    "how_are_you": [
        "Running at 100% and zero bugs (for now)!",
        "I'm doing great. Bots don't get tired. How about you?",
    ],
    "name": [
        f"I'm {BOT_NAME}, a rule-based chatbot built for DecodeLabs Project 1.",
    ],
    "help": [
        "Try: greet me, ask my name, ask the time or date, say 'joke', "
        "'tip', or 'my name is <your name>'. Type 'bye' to exit.",
    ],
    "joke": [
        "Why do programmers prefer dark mode? Because light attracts bugs!",
        "There are 10 kinds of people: those who understand binary and those who don't.",
        "A SQL query walks into a bar, sees two tables and asks: can I join you?",
    ],
    "tip": [
        "Tip: write small functions, test them one by one, then connect them.",
        "Tip: read error messages from the bottom up. The real cause is usually there.",
        "Tip: commit to GitHub often. Small commits tell your story.",
    ],
    "thanks": [
        "You're welcome!",
        "Anytime. Happy coding!",
    ],
    "time": [lambda: "It's " + datetime.now().strftime("%I:%M %p") + " right now."],
    "date": [lambda: "Today is " + datetime.now().strftime("%A, %d %B %Y") + "."],
}

# ---------------------------------------------------------------
# 2. KEYWORDS: intent -> words/phrases that trigger it
#    These get flipped into a fast lookup table below.
# ---------------------------------------------------------------
KEYWORDS = {
    "greeting": ["hi", "hello", "hey", "hola", "namaste", "yo"],
    "how_are_you": ["how are you", "how r u", "whats up", "sup"],
    "name": ["your name", "who are you", "what are you"],
    "help": ["help", "commands", "menu", "options"],
    "joke": ["joke", "funny", "laugh"],
    "tip": ["tip", "advice", "suggestion"],
    "thanks": ["thanks", "thank you", "thx", "thankyou"],
    "time": ["time", "clock"],
    "date": ["date", "today", "day"],
}

# phrase -> intent  (dictionary lookup instead of a long if-elif ladder)
LOOKUP = {
    phrase: intent
    for intent, phrases in KEYWORDS.items()
    for phrase in phrases
}

EXIT_COMMANDS = {"bye", "exit", "quit", "goodbye", "see you"}

FALLBACKS = [
    "Hmm, I don't know that one yet. Type 'help' to see what I can do.",
    "I'm still learning! Try asking for a joke or a coding tip.",
    "That's outside my rulebook. Type 'help' for ideas.",
]


# ---------------------------------------------------------------
# 3. HELPER FUNCTIONS
# ---------------------------------------------------------------
def sanitize(text: str) -> str:
    """Lowercase, remove punctuation, trim extra spaces."""
    no_punct = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(no_punct.lower().split())


def find_intent(clean: str):
    """Try the whole sentence first, then each word on its own."""
    if clean in LOOKUP:
        return LOOKUP[clean]
    for word in clean.split():
        if word in LOOKUP:
            return LOOKUP[word]
    return None


def build_reply(intent: str) -> str:
    reply = random.choice(RESPONSES[intent])
    return reply() if callable(reply) else reply


# ---------------------------------------------------------------
# 4. MAIN LOOP
# ---------------------------------------------------------------
def main() -> None:
    user_name = None
    print(f"{BOT_NAME}: Hi! I'm {BOT_NAME}. Type 'help' for ideas or 'bye' to exit.")

    while True:
        raw = input("You: ")
        clean = sanitize(raw)

        # Ignore empty input
        if not clean:
            print(f"{BOT_NAME}: Say something and I'll reply!")
            continue

        # Exit strategy
        if clean in EXIT_COMMANDS:
            who = f", {user_name}" if user_name else ""
            print(f"{BOT_NAME}: Goodbye{who}! Keep coding.")
            break

        # Memory: remember the user's name
        if clean.startswith("my name is "):
            user_name = clean[len("my name is "):].title()
            print(f"{BOT_NAME}: Nice to meet you, {user_name}!")
            continue

        # Match intent and reply (with fallback)
        intent = find_intent(clean)
        if intent:
            print(f"{BOT_NAME}: {build_reply(intent)}")
        else:
            print(f"{BOT_NAME}: {random.choice(FALLBACKS)}")


if __name__ == "__main__":
    main()
