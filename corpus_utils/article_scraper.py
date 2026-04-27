from bs4 import BeautifulSoup as bs
import requests
import re
import sys


def join_text_elements(text_elements: list[str]) -> str:
    """
    Helper function to join text extracted from <p> tags and links.

    Will join the text with a whitespace, unless the proceeding text ends with a punctuation mark, in which case it will join without a whitespace.
    This is to handle cases like "Draftsim's" where the apostrophe is part of the word and should not be separated by a space. 

    :param text_elements: A list of strings extracted from the HTML content, including both regular text and link text.
    """
    joined_text = ""
    for i, element in enumerate(text_elements):
        next_element = text_elements[i + 1] if i < len(text_elements) - 1 else ""
        prev_element = text_elements[i - 1] if i > 0 else ""
        if (
            next_element.startswith("'")
            or next_element.startswith(",")
            or next_element.startswith(";")
            or next_element.startswith(".")
            or next_element.startswith(":")
            or next_element.startswith("!")
            or next_element.startswith("?")
            or next_element.startswith("‘")  # stylistic apostrophe because of course this is a thing
            or prev_element.endswith("-")  # handle cases like "non-Entity"
        ):
            joined_text += element
        else:
            joined_text += element   + " "
    
    return joined_text.strip()


def scrape_words_for_ner(url):
    # 1. Fetch the webpage
    # We use a User-Agent header because some sites block default Python requests
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        print(f"Failed to retrieve page. Status code: {response.status_code}")
        return []

    # 2. Parse the HTML using BeautifulSoup
    soup = bs(response.text, 'html.parser')

    # 3. Locate the main article content
    # Draftsim and most WordPress sites use 'entry-content' or the <article> tag.
    main_content = soup.find('article') 
    if not main_content:
        # Fallback if <article> isn't found
        main_content = soup.find('div', class_='entry-content')
        
    if not main_content:
        print("Could not find the main content container.")
        return []

    text_elements = main_content.find_all(['p'])
    links = [element.find_all(['a']) for element in text_elements]
    
    full_text = []
    for text_element in text_elements:
        text_element_text = []
        for child in text_element.contents:
            if child.name == 'a':
                # print(f"Link text: {child.get_text(strip=True)}")
                text_element_text.append(child.get_text(strip=True))
            elif isinstance(child, str):
                clean_text = child.strip()
                if clean_text:
                    # print(f"Text: {clean_text}")
                    text_element_text.append(clean_text)
        # print(f"Parsed text from element: {text_element_text}")
        joined_text = join_text_elements(text_element_text)
        # print(f"\nJoined text: {joined_text}")
        full_text.append(joined_text)
        # print("\n--- END OF TEXT ELEMENT ---\n\n")

    return "\n\n".join(full_text)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python article_scraper_and_tokenizer.py <URL>")
        sys.exit(1)

    url = sys.argv[1]
    words = scrape_words_for_ner(url)
    print(words)