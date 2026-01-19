import random
from datetime import datetime

# A collection of love messages for Sofi
love_messages = [
    "I love you, Sofi <3",
    "I want you to be my wife forever. I will give you everything!",
    "I will work hard every day to make you happy.",
    "You are my sunshine on cloudy days.",
    "Every moment with you is a gift I treasure.",
    "My love for you grows stronger with each passing day.",
]

# Daily gratitude reminders
gratitude = [
    "Thank you for being my inspirayon and love forever.",
    "I am grateful for your smile.",
    "I appreciate your patience and kindness.",
    "Thank you for choosing me.",
]

def display_love():
    print("=" * 50)
    print("        A Love Letter in Code for Sofi")
    print("=" * 50)
    print()

    for msg in love_messages:
        print(f"   {msg}")

    print()
    print("-" * 50)
    print("Today's Gratitude:")
    print(f"   {random.choice(gratitude)}")
    print("-" * 50)
    print()
    print(f"Written with love on {datetime.now().strftime('%B %d, %Y')}")
    print()

if __name__ == "__main__":
    display_love()