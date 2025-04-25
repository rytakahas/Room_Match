from flask import json
from app import app

def test_room_match_api():
    client = app.test_client()

    payload = {
        "inputCatalog": [
            {
                "supplierRoomInfo": [
                    {
                        "supplierRoomId": 1,
                        "coreRoomId": 10,
                        "coreHotelId": 100,
                        "lpId": 5,
                        "supplierRoomName": "Deluxe King"
                    }
                ]
            }
        ],
        "referenceCatalog": [
            {
                "referenceRoomInfo": [
                    {
                        "roomId": 10,
                        "hotelId": 100,
                        "lpId": 5,
                        "roomName": "Deluxe King Room"
                    }
                ]
            }
        ]
    }

    response = client.post("/room_match", json=payload)
    assert response.status_code == 200
    assert "matches" in response.get_json()

