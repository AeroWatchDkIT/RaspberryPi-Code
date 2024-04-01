
# tag_parser.py

from datetime import datetime
import json
import requests
import os
from requests.exceptions import ConnectionError, Timeout, RequestException

# Parse tag code into shelf ID, column, and row
class TagParser:
    # @staticmethod
    # def send_failed_tags():
    #     filename = "backup.txt"

    #     if os.stat(filename).st_size != 0:
    #         with open(filename, "r") as f:
    #             lines = f.readlines()

    #         while len(lines) > 0:
    #             line = lines.pop(0)
    #             json_data = json.dumps(json.loads(line))
    #             TagParser.send_tags(json_data=json_data)

    #         with open(filename, "w") as f:
    #             f.writelines(lines)

    @staticmethod
    def send_failed_tags():
        filename = "backup.txt"
        lines_to_keep = []

        with open(filename, "r") as f:
            lines = f.readlines()

        for line in lines:
            try:
                json_data = json.loads(line.strip())
                TagParser.send_tags(json_data=json.dumps(json_data))
            except RequestException:
                # If sending fails, keep the line for the next attempt
                lines_to_keep.append(line)

        # Rewrite the file with the lines that weren't successfully sent
        with open(filename, "w") as f:
            f.writelines(lines_to_keep)
           

    @staticmethod
    def parse_tag_code(tag_code):
        parts = tag_code.split('.')
        shelf_id = parts[0]
        column = parts[1]
        row = parts[2]
        return shelf_id, column, row

    # Construct JSON data for POST request
    @staticmethod
    def construct_json(shelf_id, pallet_id, user_id="U-0001", forklift_id="F-0012"):
        data = {
            "pallet": {
                "id": pallet_id,
                "state": 0,
                "location": "Warehouse B"
            },
            "shelf": {
                "id": shelf_id,
                "palletId": None,
                "location": "Warehouse D"
            },
            "timeOfInteraction": datetime.now().isoformat(),
            "action": "Pallet removed from shelf",
            "userId": user_id,
            "forkliftId": forklift_id
        }
        return json.dumps(data)

    # Send POST request to API
    # @staticmethod
    # def send_tags(shelf_tag=None, pallet_tag=None, json_data=None):
    #     if json_data is None:
    #         json_data = TagParser.construct_json(shelf_id=shelf_tag, pallet_id=pallet_tag)

    #     try:
    #         #url = "https://192.168.1.2:7128/Interactions/TwoCodes"
    #         url = "https://192.168.207.168:7128/Interactions/TwoCodes"
    #         #url = "https://86.41.123.214:7128/Interactions/TwoCodes"
    #         headers = {"Content-Type": "application/json"}
    #         print(f"Sending request to URL: {url}")
    #         response = requests.post(url, data=json_data, headers=headers, verify=False)

    #         if response.status_code != 200:
    #             print(f"Error: {response.status_code} - {response.text}")
    #             raise Exception("Failed to send data to server.")
            
    #         print(f"Success: {response.text}")
    #     except requests.exceptions.ConnectionError as e:
    #         print(f"Network error: {e}. Storing the request for later.")
    #         with open("backup.txt", "a") as f:
    #             f.write(json_data + '\n')
    #     except Exception as e:
    #         print(f"An error occurred: {e}. Storing the request for later.")
    #         with open("backup.txt", "a") as f:
    #             f.write(json_data + '\n')
    # @staticmethod
    # def send_tags(shelf_tag=None, pallet_tag=None, json_data=None):
    #     if json_data is None:
    #         json_data = TagParser.construct_json(shelf_id=shelf_tag, pallet_id=pallet_tag)

    #     try:
    #         url = "https://192.168.1.2:7128/Interactions/TwoCodes"
    #         headers = {"Content-Type": "application/json"}
    #         response = requests.post(url, data=json_data, headers=headers, verify=False)

    #         if response.status_code == 200:
    #             print("Success:", response.text)
    #         else:
    #             print("Error:", response.status_code, response.text)
    #     except (ConnectionError, Timeout, RequestException) as e:
    #         print(f"Network error occurred: {e}. Storing failed request in file.")
    #         with open("backup.txt", "a") as f:
    #             f.write(json_data + '\n')

    @staticmethod
    def send_tags(shelf_tag=None, pallet_tag=None, json_data=None):
        # Constructing JSON data
        if json_data is None:
            json_data = TagParser.construct_json(shelf_id=shelf_tag, pallet_id=pallet_tag)

        try:
            # URL of the backend endpoint
            #url = "https://192.168.207.168:7128/Interactions/TwoCodes" #pi4_college_redmi
            url = "https://192.168.1.23:7128/Interactions/TwoCodes" #pi5_home
            #url = "https://192.168.16.222:7128/Interactions/TwoCodes"
            headers = {"Content-Type": "application/json"}

            # Making the POST request
            #response = TagParser.post_data(url, json_data)

            response = requests.post(url, data=json_data,headers=headers, verify=False)

            # Processing the response
            if response.status_code == 200:  # Or another success code as per your API
                print("Success:", response.text)
            else:
                print("Error:", response.status_code, response.text)
        except:
            # TODO: Create single table database to be extra sure that data won't be lost on the PI
            print("Something went wrong. Storing failed request in file")
            f = open("backup.txt", "a")
            f.write(json_data)
            f.write("\n")
            f.close()