# tests/test_room_match.py

from matcher import normalize, embed_text, match_supplier_to_reference
import torch

def test_normalize():
    assert normalize("Hôtel Supérieur") == "hotel superieur"
    assert normalize("Deluxe Room!!") == "deluxe room"
    assert normalize(None) == ""

def test_embed_text_returns_tensor():
    emb = embed_text("Deluxe King Room")
    assert isinstance(emb, torch.Tensor)
    assert emb.ndim == 1  # Should be a 1D vector

def test_match_supplier_to_reference_basic():
    supplier_rooms = [
        {"supplierRoomId": "S1", "supplierRoomName": "Deluxe King Bed"}
    ]
    reference_rooms = [
        {"roomId": "R1", "roomName": "Deluxe King"},
        {"roomId": "R2", "roomName": "Budget Twin Room"}
    ]

    results = match_supplier_to_reference(supplier_rooms, reference_rooms)
    assert isinstance(results, list)
    assert len(results) == 2

    for result in results:
        assert "match_score" in result
        assert "cosine_sim" in result
        assert isinstance(result["match_score"], float)
        assert isinstance(result["cosine_sim"], float)
        assert result["supplierRoomId"] == "S1"
        assert result["refRoomId"] in ["R1", "R2"]

