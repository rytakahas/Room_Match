from flask import Flask, request, jsonify
from matcher import match_supplier_to_reference

app = Flask(__name__)

@app.route("/room_match", methods=["POST"])
def room_match():
    data = request.get_json()
    input_catalog = data.get("inputCatalog", [])
    reference_catalog = data.get("referenceCatalog", [])

    # Flatten input and reference room lists
    supplier_rooms = input_catalog[0]['supplierRoomInfo']
    reference_rooms = reference_catalog[0]['referenceRoomInfo']

    results = match_supplier_to_reference(supplier_rooms, reference_rooms)
    return jsonify({"matches": results})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=True)
