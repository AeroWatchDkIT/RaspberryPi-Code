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
                "state": 1,
                "location": "Warehouse D"
            },
            "shelf": {
                "id": shelf_id,
                "palletId": None,
                "location": "None"
            },
            "timeOfInteraction": datetime.now().isoformat(),
            "action": "Pallet removed from shelf",
            "userId": user_id,
            "forkliftId": forklift_id
        }
        return json.dumps(data)

    # Send POST request to API
    @staticmethod
    def post_data(url, json_data):
        headers = {"Content-Type": "application/json"}
        response = requests.post(url, data=json_data, headers=headers, verify=False)
        return response