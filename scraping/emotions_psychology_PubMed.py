import requests
import time
from xml.etree import ElementTree
from pylatexenc.latex2text import LatexNodes2Text

# Constants
PUBMED_SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_FETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
OUTPUT_FILE = "./original/emotions.txt"
OUTPUT_PARSED_FILE = "./latex_parsed/emotions.txt"
API_KEY = "d2cc067dd37c4df5ad6992c386fb22691808"  # Replace with your PubMed API Key

# Function to search PubMed for PMIDs
def search_pubmed(query, start=0, max_results=200):
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retstart": start,
        "api_key": API_KEY
    }
    response = requests.get(PUBMED_SEARCH_URL, params=params)
    response.raise_for_status()
    return ElementTree.fromstring(response.content)

# Function to fetch abstracts for a list of PMIDs
def fetch_pubmed_abstracts(pmids):
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
        "rettype": "abstract",
        "api_key": API_KEY
    }
    response = requests.get(PUBMED_FETCH_URL, params=params)
    response.raise_for_status()
    return ElementTree.fromstring(response.content)

# Function to parse abstracts from PubMed
def parse_abstracts(pubmed_xml):
    abstracts = []
    for article in pubmed_xml.findall(".//Abstract"):
        abstract_text = " ".join(elem.text for elem in article.findall(".//AbstractText") if elem.text)
        if abstract_text:
            abstracts.append(abstract_text)
    return abstracts

# Main function to retrieve 600 emotions abstracts 1799
def get_psychology_abstracts():
    query = "Emotions[MeSH Terms] AND abstract[Title/Abstract]"
    total_abstracts = []
    total_results = 1706
    max_per_request = 300

    for start in range(0, total_results, max_per_request):
        print(f"Fetching PMIDs {start} to {start + max_per_request}...")
        search_results = search_pubmed(query, start=start, max_results=max_per_request)
        pmids = [id_elem.text for id_elem in search_results.findall(".//Id")]
        
        print(f"Fetching abstracts for {len(pmids)} articles...")
        fetch_results = fetch_pubmed_abstracts(pmids)
        abstracts = parse_abstracts(fetch_results)
        total_abstracts.extend(abstracts)
        time.sleep(3)  # Sleep for 3 seconds

    return total_abstracts

# Run and save results
psychology_abstracts = get_psychology_abstracts()
with open(OUTPUT_FILE, "w", encoding="utf-8") as original_file:
    with open(OUTPUT_PARSED_FILE, "w", encoding="utf-8") as parsed_file:
        for abstract in psychology_abstracts:
            original_file.write(abstract + "\n\n")
            parsed_abs = LatexNodes2Text().latex_to_text(abstract)
            parsed_file.write(parsed_abs + "\n\n")


print(f"Total emotions abstracts saved: {len(psychology_abstracts)}")
