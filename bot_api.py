from flask import Flask, request, jsonify

app = Flask(__name__)

# Hardcoded chatbot responses
responses = {
    "get me all dell laptops": {
        "status": "success",
        "data": [
            {
                "id": 101,
                "name": "Dell Latitude 5400",
                "asset_tag": "ASSET-2023-001",
                "serial": "D5400-XYZ123",
                "purchase_date": "2023-06-15",
                "current_value": "900.00",
                "manufacturer": "Dell",
                "condition": "Good",
                "isCheckedOut": 1
            },
            {
                "id": 102,
                "name": "Dell XPS 15",
                "asset_tag": "ASSET-2023-002",
                "serial": "XPS15-ABC456",
                "purchase_date": "2022-11-20",
                "current_value": "1300.00",
                "manufacturer": "Dell",
                "condition": "Excellent",
                "isCheckedOut": 0
            }
        ],
        "message": "2 Dell laptops found."
    },
    "show me assets that are checked out": {
        "status": "success",
        "data": [
            {
                "id": 101,
                "name": "Dell Latitude 5400",
                "asset_tag": "ASSET-2023-001",
                "assigned_user": 23,
                "last_checkout": "2025-02-15 10:45:00",
                "expected_checkin": "2025-03-20"
            },
            {
                "id": 205,
                "name": "MacBook Pro 16",
                "asset_tag": "ASSET-2024-010",
                "assigned_user": 45,
                "last_checkout": "2025-03-10 09:30:00",
                "expected_checkin": "2025-04-05"
            }
        ],
        "message": "2 assets are currently checked out."
    },
    "find me the asset with serial xps15-abc456": {
        "status": "success",
        "data": {
            "id": 102,
            "name": "Dell XPS 15",
            "asset_tag": "ASSET-2023-002",
            "serial": "XPS15-ABC456",
            "purchase_date": "2022-11-20",
            "current_value": "1300.00",
            "manufacturer": "Dell",
            "condition": "Excellent",
            "isCheckedOut": 0,
            "location_id": 5,
            "manager_id": 12,
            "description": "Dell XPS 15-inch laptop with 32GB RAM and 1TB SSD."
        },
        "message": "Asset details retrieved successfully."
    },
    "show me all assets assigned to user 23": {
        "status": "success",
        "data": [
            {
                "id": 101,
                "name": "Dell Latitude 5400",
                "asset_tag": "ASSET-2023-001",
                "serial": "D5400-XYZ123",
                "condition": "Good",
                "isCheckedOut": 1
            },
            {
                "id": 302,
                "name": "Samsung Galaxy S22",
                "asset_tag": "ASSET-2024-020",
                "serial": "SGS22-DEF789",
                "condition": "Excellent",
                "isCheckedOut": 1
            }
        ],
        "message": "2 assets assigned to user 23."
    },
}

@app.route('/api/chatbot', methods=['POST'])
def chatbot():
    user_input = request.json.get("query", "").lower()
    
    response = responses.get(user_input, {"status": "error", "message": "I don't understand that request."})

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
