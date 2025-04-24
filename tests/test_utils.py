import pandas as pd
from room_match.utils import normalize, fast_match, generate_matches

def test_normalize():
    assert normalize("Hôtel Supérieur") == "hotel superieur"

def test_fast_match_score():
    assert fast_match("Deluxe King", "Deluxe King Room") > 0.8

def test_generate_matches_hotel_id_check():
    df_rooms = pd.DataFrame([{
        "supplier_room_id": 1,
        "core_room_id": 10,
        "core_hotel_id": 100,
        "lp_id": 5,
        "supplier_room_name": "Deluxe King"
    }])

    df_ref = pd.DataFrame([{
        "room_id": 10,
        "hotel_id": 999,  # Should fail hotel match
        "lp_id": 5,
        "room_name": "Deluxe King Room"
    }])

    matches = generate_matches(df_rooms, df_ref)
    assert len(matches) == 0

