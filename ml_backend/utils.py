import string
import random

def generate_string(length):

    letters = string.ascii_lowercase  # define the lower case string
    # define the condition for random.choice() method
    result = ''.join((random.choice(letters)) for x in range(length))
    return result
