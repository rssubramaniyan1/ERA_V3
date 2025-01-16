import os
import requests
from bs4 import BeautifulSoup

def download_unicode_files(url, save_directory):
    # Create the directory if it doesn't exist
    os.makedirs(save_directory, exist_ok=True)

    try:
        # Fetch the webpage content
        response = requests.get(url, timeout=30, verify=False)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Locate the table by its class or id
        table = soup.find('table', {'class': 'sortable', 'id': 'sortabletable'})
        if not table:
            print("No table found on the page.")
            return

        # Add check for tbody
        tbody = table.find('tbody')
        if not tbody:
            print("Table structure not as expected (no tbody found).")
            return

        # Iterate over the table rows (skip the header row)
        for row in tbody.find_all('tr'):
            # Add error handling for row structure
            try:
                columns = row.find_all('td')
                if len(columns) < 6:  # Check if row has enough columns
                    print(f"Skipping row: insufficient columns")
                    continue
                
                unicode_column = columns[5]
                unicode_links = unicode_column.find_all('a', href=True)

                for link in unicode_links:
                    href = link['href']
                    # Construct absolute URL if the link is relative
                    file_url = href if href.startswith('http') else f'https://www.projectmadurai.org{href}'
                    file_name = os.path.join(save_directory, os.path.basename(file_url))

                    # Add progress indicator
                    print(f"Downloading: {file_url}", end='')
                    file_response = requests.get(file_url, timeout=30, verify=False)
                    file_response.raise_for_status()

                    # Save the file
                    with open(file_name, 'wb') as file:
                        file.write(file_response.content)
                    print(" ✓")  # Checkmark to indicate success

            except (IndexError, AttributeError) as e:
                print(f"Error processing row: {e}")
                continue
            except requests.exceptions.RequestException as e:
                print(f"\nError downloading {file_url}: {e}")
                continue

    except requests.exceptions.RequestException as e:
        print(f"An error occurred while accessing the main page: {e}")

# Usage
if __name__ == "__main__":
    url = "https://www.projectmadurai.org/pmworks.html"
    save_directory = "/Users/ravishankarsubramaniyam/Desktop/BPE/unicode_files"
    download_unicode_files(url, save_directory)
