from dotenv import load_dotenv
import os

load_dotenv()  # This line brings all environment variables from .env into os.environ
print(os.environ.get("OPENAI_API_KEY"))
print(os.environ.get("YELP_API_KEY"))
print(os.environ.get("GOOGLE_MAPS_API_KEY"))

