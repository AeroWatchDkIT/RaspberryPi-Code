# tag_parser.py

from datetime import datetime
import json
import requests

# Parse tag code into shelf ID, column, and row
class TagParser:
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
    @staticmethod
    def send_tags(shelf_tag, pallet_tag):
        # Constructing JSON data
        json_data = TagParser.construct_json(shelf_id=shelf_tag, pallet_id=pallet_tag)

        # URL of the backend endpoint
        url = "https://192.168.1.2:7128/Interactions/TwoCodes"
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