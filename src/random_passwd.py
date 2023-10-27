import random
import string
import sys

def generate_password(length):
    """Generate a random password of specified length"""
    characters = string.ascii_letters + string.digits + string.punctuation[0:20]
    password = ''.join(random.choice(characters) for i in range(length))
    return password

# Example usage
passswd_length = int(sys.argv[1])
password = generate_password(passswd_length)
print(password)
