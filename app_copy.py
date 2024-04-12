import requests
from bs4 import BeautifulSoup
import re
from datetime import date
from flask import Flask, request
from apscheduler.schedulers.background import BackgroundScheduler
import json
from urllib.parse import quote
from main import  file_reader

today_date = date.today()
url = 'https://www.wcpc.us/'
probateurl = 'https://www.wcpc.us/probate-information.html'

app = Flask(__name__)
scheduler = BackgroundScheduler()

current_date_str = today_date.strftime(r"%Y-%m-%d")
json_file_path = "data.json"
json_allfile_path = "alldata.json"

try:
    with open(json_allfile_path, "r") as file:
        allexisting_data = json.load(file)
except:
    allexisting_data = []

try:
    with open(json_file_path, "r") as file:
        existing_data = json.load(file)
except:
    existing_data = {}


def is_duplicate(new_data):
    for item in allexisting_data:
        if item.get("anchor_text") and item["anchor_text"].strip() == new_data.get("anchor_text", "").strip():
            return True


# Function to scrape text and paragraphs from a URL
def scrap_text_and_paragraphs(newurl, anchor_text):
    anchor_text = re.sub(r'\s+', ' ', anchor_text)
    anchor_text = anchor_text.replace('\n', '').replace('\r', '').replace('\u2019', '').replace('\u201c', '').replace('\u201d', '').replace(r'\u\w+', '').replace('\u2013','').replace('\\','').replace('\"','')

    newurl = newurl.replace('..', '')
    pattern = r'http[s]?://'
    hash_pattern = r'^#'
    
    if ' ' in newurl:
        newurl = quote(newurl, safe='')

    match = re.search(pattern, newurl)

    component_url = newurl

    if match is None:
        newurl = url + newurl

    text_file_format = '\n' + f"Link: {newurl} , Text: {' '.join(anchor_text.split())}" + '\n\n\n\n' + '.......................................................................................................'
    
    if not newurl.endswith(('.bmp', '.pdf', 'https://www.facebook.com/wcpcus/', '.aspx', '.org', '.gov')) and \
        not newurl.startswith('https://youtu.be/'):

        response = requests.get(newurl)
        
        soup = BeautifulSoup(response.content, 'lxml')
        paragraphs = [tag.get_text() for tag in soup.find_all(['p'])]
        
        formatted_text_json = re.sub(r'\s+', ' ', '\n'.join(paragraphs))
        formatted_text_json = formatted_text_json.replace('\u2019', '').replace('\u201c', '').replace('\u201d', '').replace('\u2013','').replace('\\','').replace('\"','')
        formatted_text_json = re.sub(r'\\u\w+', '', formatted_text_json)

        paragraphs = '.\n'.join(paragraphs).replace("�", "").replace('..', '.').replace('�B�', '')
        paragraphs = paragraphs.replace('\u2019', '').replace('\u201c', '').replace('\u201d', '').replace('\u2013','').replace('\\','').replace('\"','')
        paragraphs = re.sub(r'\\u\w+', '', paragraphs)
        paragraphs = re.sub(r'\s{2,}', '\n', paragraphs)

        new_link_data = {"anchor_text": anchor_text,'anchor_link':newurl,"scripted_value": formatted_text_json}
        text_file_format = '\n' + f"Link: {newurl} , Text: {' '.join(anchor_text.split())}" + '\n\n' + paragraphs + '\n\n' + '.......................................................................................................'
        file_name = f'{current_date_str}.txt'
        return_file_format=''
        nested_href=''
        nested_anchor_text=''
       
        

        for link in soup.find_all('a'):
            nested_href = link.get('href')
            if nested_href:
                nested_href = nested_href.replace('..', '')
                nested_anchor_text = link.text
                nested_anchor_text = re.sub(r'\s+', ' ', nested_anchor_text)
                nested_anchor_text = nested_anchor_text.replace('\n', '').replace('\r', '').replace('\u2019', '').replace('\u201c', '').replace('\u201d', '').replace(r'\u\w+', '').replace('\u2013','').replace('\\','').replace('\"','')
          

                if ' ' in nested_href:
                    nested_href = quote(nested_href, safe='')
                    nested_href = nested_href 
                
                match_nested_link = re.search(pattern, nested_href)
                not_started_hash = re.match(hash_pattern, nested_href)

                if  match_nested_link is None and not not_started_hash:
                    nested_href = url + nested_href

                elif re.match(hash_pattern, nested_href):  
                    nested_href = url + component_url + nested_href

                     
                new_nested_link_data = {"anchor_text": nested_anchor_text,'anchor_link':nested_href,"parent_url":newurl,'scripted_value':''}
  
                if not is_duplicate(new_link_data):
                    allexisting_data.append(new_nested_link_data)
              

                if nested_href not in existing_data:
                   existing_data[nested_href] = new_nested_link_data

        
      
        
        if  newurl in existing_data:
               previous_data = existing_data[newurl]

               if (previous_data["scripted_value"] != formatted_text_json):
                  new_link_data['scripted_value'] = formatted_text_json
                  existing_data[newurl] = new_link_data 
                  return_file_format= text_file_format
                  with open(file_name, 'a', encoding='utf-8') as file:
                       file.write(text_file_format)
        
        else:
            existing_data[newurl] = new_link_data
            return_file_format= text_file_format
            with open(file_name, 'a', encoding='utf-8') as file:
                    file.write(text_file_format)


        if not is_duplicate(new_link_data):
            allexisting_data.append(new_link_data)


        with open(json_file_path, "w") as file:
            json.dump(existing_data, file, indent=2)

        

        seen_anchor_texts = set()
        unique_data = []

        for item in allexisting_data:
            anchor_text = item.get("anchor_text")

            if anchor_text not in seen_anchor_texts:
                seen_anchor_texts.add(anchor_text)
                item["scripted_value"] = ""
                item["anchor_text"] =  anchor_text.replace("(opens in new window)", "").replace("(opens in new window", "").strip()

                if anchor_text.lower() == "filing":
                            item["anchor_link"] = "https://www.wcpc.us/filing.html"

                unique_data.append(item)

        with open(json_allfile_path, "w") as file:
          json.dump(unique_data, file, indent=2)        

        return return_file_format                  
    else:
        return text_file_format
  
# Function to perform web scraping and train the chatbot
def scraping_fn(soup_objects):
    collected_content = set()
    for current_soup in soup_objects:
        for link in current_soup.find_all('a'):
            href = link['href']
            anchor_text = link.text
            scraped_content = scrap_text_and_paragraphs(href, anchor_text)
            if scraped_content:
               collected_content.add(scraped_content)
                        
    return ''.join(collected_content)

# Function to run the scheduler
def my_function():
    reqs = requests.get(url)
    soup = BeautifulSoup(reqs.text, 'lxml')
    probateres = requests.get(probateurl)
    soup1 = BeautifulSoup(probateres.text, 'lxml')
    soup_objects = [soup,soup1]
    new_data = scraping_fn(soup_objects)

    if new_data:
        with open('alldata.txt', 'a', encoding='utf-8') as file:
            file.write(new_data)
            file_reader()

if __name__ == "__main__":
    my_function()
    scheduler.add_job(id = 'checkig_for_site_update', func = my_function, trigger = 'interval', hours=23)
    scheduler.start()
