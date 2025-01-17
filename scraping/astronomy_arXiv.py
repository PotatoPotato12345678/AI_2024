import requests
import xml.etree.ElementTree as ET
import time
from pylatexenc.latex2text import LatexNodes2Text


# Constants
ARXIV_API_URL = "http://export.arxiv.org/api/query"
OUTPUT_FILE = "./original/astronomy.txt"

# Function to fetch abstracts from arXiv
def fetch_arxiv_abstracts(category="astro-ph", start=0, max_results=200):
    params = {
        "search_query": f"cat:{category}",
        "start": start,
        "max_results": max_results
    }
    response = requests.get(ARXIV_API_URL, params=params)
    response.raise_for_status()
    return response.text

# Function to parse the abstracts from XML response
def parse_abstracts(response_xml):
    abstracts = []
    root = ET.fromstring(response_xml)
    for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
        abstract = entry.find("{http://www.w3.org/2005/Atom}summary").text.strip()
        abstracts.append(abstract)
    return abstracts

# Main function to retrieve 600 astronomy abstracts
def get_astronomy_abstracts():
    total_abstracts = []
    total_results = 2000
    max_per_request = 2000

    for start in range(0, total_results, max_per_request):
        print(f"Fetching results {start} to {start + max_per_request}...")
        response_xml = fetch_arxiv_abstracts(start=start, max_results=max_per_request)
        abstracts = parse_abstracts(response_xml)
        total_abstracts.extend(abstracts)
        time.sleep(3)  # Sleep for 3 seconds to avoid rate limiting

    return total_abstracts

# Run and save results
astronomy_abstracts = get_astronomy_abstracts()
with open(OUTPUT_FILE, "w", encoding="utf-8") as original_file:
    for abstract in astronomy_abstracts:
        original_file.write(abstract + "\n\n")
        parsed_abs = LatexNodes2Text().latex_to_text(abstract)

print(f"Total Astronomy abstracts saved: {len(astronomy_abstracts)}")
