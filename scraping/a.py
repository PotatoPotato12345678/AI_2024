import requests
from xml.etree import ElementTree

# PubMed API base URL for searching
PUBMED_SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"

# Function to get the total number of xxxx papers
def get_pubmed_paper_count(query="cognitive psychology[MeSH Terms] AND abstract[Title/Abstract]"):
    params = {
        "db": "pubmed",
        "term": query,  # Query term for sociology papers
        "retmax": 1,  # Limit results to 1 (just to get the total count)
        "api_key": "d2cc067dd37c4df5ad6992c386fb22691808"  # Replace with your NCBI API Key
    }
    response = requests.get(PUBMED_SEARCH_URL, params=params)
    response.raise_for_status()  # Raise an error if the request fails
    tree = ElementTree.fromstring(response.content)
    
    # Find the total number of results in the response
    count = tree.find(".//Count").text
    return int(count)

# Get and print the total number of xxxx papers in PubMed
total_sociology_papers = get_pubmed_paper_count()
print(f"Total number of cognitive psychology[MeSH Terms] papers in PubMed: {total_sociology_papers}")
