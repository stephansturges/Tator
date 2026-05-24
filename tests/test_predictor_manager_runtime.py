import localinferenceapi as api


def test_predictor_capacity_reduction_unloads_disabled_slots(monkeypatch):
    manager = api.PredictorManager()
    unload_calls = []
    try:
        manager.capacity = 3
        manager.enabled_slots = set(manager.slot_order)
        manager.slots["next"].token = "next-token"
        manager.slots["next"].variant = "sam1"
        manager.slots["next"].image_name = "next.jpg"
        manager.token_index[("next-token", "sam1")] = manager.slots["next"]
        manager.image_index[("sam1", "next.jpg")] = manager.slots["next"]

        for slot_name in ("next", "previous"):
            slot = manager.slots[slot_name]

            def record_unload(slot=slot, name=slot_name):
                unload_calls.append(name)
                slot.clear()

            monkeypatch.setattr(slot, "unload", record_unload)

        manager.set_capacity(1)

        assert set(unload_calls) == {"next", "previous"}
        assert manager.get_capacity() == 1
        assert manager.enabled_slots == {"current"}
        assert ("next-token", "sam1") not in manager.token_index
        assert ("sam1", "next.jpg") not in manager.image_index
    finally:
        manager.stop()
