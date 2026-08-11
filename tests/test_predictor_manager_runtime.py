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


def test_predictor_manager_honors_configured_default_capacity(monkeypatch):
    monkeypatch.setattr(api, "DEFAULT_PREDICTOR_SLOTS", 1)
    manager = api.PredictorManager()
    try:
        assert manager.get_capacity() == 1
        assert manager.enabled_slots == {"current"}
    finally:
        manager.stop()


def test_predictor_status_reports_runtime_and_load_latency():
    manager = api.PredictorManager()
    try:
        slot = manager.slots["current"]
        slot.variant = "sam1"
        slot.last_load_ms = 123.5
        slot.backends["sam1"] = type("Backend", (), {"backend": "mlx", "device": "metal"})()

        current = next(item for item in manager.status() if item["slot"] == "current")

        assert current["backend"] == "mlx"
        assert current["device"] == "metal"
        assert current["load_ms"] == 123.5
    finally:
        manager.stop()


def test_repeated_preload_of_active_token_does_not_reencode():
    manager = api.PredictorManager()
    set_image_calls = []

    class Backend:
        backend = "mlx"
        device = "metal"

        def set_image(self, image):
            set_image_calls.append(image.shape)

        def unload(self):
            return None

    try:
        slot = manager.slots["current"]
        slot.backends["sam1"] = Backend()
        image = api.np.zeros((8, 9, 3), dtype=api.np.uint8)

        manager.set_slot("current", image, "same-token", "sam1", "image.jpg")
        manager.set_slot("current", image, "same-token", "sam1", "image.jpg")

        assert set_image_calls == [(8, 9, 3)]
    finally:
        manager.stop()
