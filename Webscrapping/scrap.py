from selenium import webdriver
import time
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import WebDriverException
import pandas as pd

# video_url = "https://www.youtube.com/watch?v=X0tOpBuYasI&t=74s&pp=ygUJYmxhY2sgYWRh"


# data = []
# path = "C:\\Webdriver\\chromedriver-win64\\chromedriver.exe"
# service = Service(executable_path=path)

# try:
#     url = input("Enter the link: ")
#     # Basic URL validation could be added here
    
#     with webdriver.Chrome(service=service) as driver:
#         wait = WebDriverWait(driver, 10)
#         driver.get(url)

       
#         SCROLL_PAUSE_TIME = 5
#         SCROLL_ITERATIONS = 100
#         for _ in range(SCROLL_ITERATIONS):
#             wait.until(EC.visibility_of_element_located((By.TAG_NAME, "body"))).send_keys(Keys.END)
#             time.sleep(SCROLL_PAUSE_TIME)

#         # Extract comments
#         comments = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "#comment #content-text")))
#         for comment in comments:
#             data.append(comment.text)

#     df = pd.DataFrame(data, columns=['Comment'])

#     # Save DataFrame to CSV file
#     df.to_csv('comments.csv', index=False)
    
#     print("Scraped comments saved to 'comments.csv' Successfully")

# except WebDriverException as e:
#     print("An error occurred:", e)






# data = []   
# path = "C:\\Webdriver\\chromedriver-win64\\chromedriver.exe"
# service = Service(executable_path=path)

# try:
#     url = input("Enter the link: ")
    
#     with webdriver.Chrome(service=service) as driver:
#         wait = WebDriverWait(driver, 10)
#         driver.get(url)

#         SCROLL_PAUSE_TIME = 5
#         SCROLL_ITERATIONS = 100
#         for _ in range(SCROLL_ITERATIONS):
#             try:
#                 wait.until(EC.visibility_of_element_located((By.TAG_NAME, "body"))).send_keys(Keys.END)
#                 time.sleep(SCROLL_PAUSE_TIME)
#             except WebDriverException as e:
#                 print("Error while scrolling:", e)
#                 # Handle browser disconnects by refreshing the page and continuing scrolling
#                 driver.refresh()
#                 continue

#         # Extract comments
#         comments = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "#comment #content-text")))
#         for comment in comments:
#             data.append(comment.text)

#     df = pd.DataFrame(data, columns=['Comment'])

#     # Save DataFrame to CSV file
#     df.to_csv('comments.csv', index=False)
    
#     print("Scraped comments saved to 'comments.csv' Successfully")

# except WebDriverException as e:
#     print("An error occurred:", e)








import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import WebDriverException, NoSuchElementException
import time

data = []
path = "C:\\Webdriver\\chromedriver-win64\\chromedriver.exe"
service = Service(executable_path=path)

try:
    url = input("Enter the link: ")
    
    with webdriver.Chrome(service=service) as driver:
        wait = WebDriverWait(driver, 10)
        driver.get(url)

        SCROLL_PAUSE_TIME = 5
        SCROLL_ITERATIONS = 150
        for _ in range(SCROLL_ITERATIONS):
            try:
                wait.until(EC.visibility_of_element_located((By.TAG_NAME, "body"))).send_keys(Keys.END)
                time.sleep(SCROLL_PAUSE_TIME)
            except WebDriverException as e:
                print("Error while scrolling:", e)
                driver.refresh()
                continue

        # Extract comments
        comments = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "#comment")))
        for comment in comments:
            try:
                author = comment.find_element(By.CSS_SELECTOR, "#author-text").text
            except NoSuchElementException:
                author = ""
                
            try:
                like_count = comment.find_element(By.CSS_SELECTOR, "#vote-count-middle").text
            except NoSuchElementException:
                like_count = ""
                
            try:
                text = comment.find_element(By.CSS_SELECTOR, "#content-text").text
            except NoSuchElementException:
                text = ""
            
            data.append([author, like_count, text])

    df = pd.DataFrame(data, columns=['Author', 'Like Count', 'Comment'])

    # Save DataFrame to CSV file
    df.to_csv('comments.csv', index=False)
    
    print("Scraped comments saved to 'comments.csv' Successfully")

except WebDriverException as e:
    print("An error occurred:", e)

