from __future__ import annotations

import unittest

from PIL import Image

from utils.embedding_recipe import (
    make_embedding_crop_views,
    preprocess_crop_with_metadata,
)


class EmbeddingRecipeSpatialMetadataTests(unittest.TestCase):
    def test_canonical_letterbox_metadata_tracks_real_content(self) -> None:
        landscape = Image.new("RGB", (200, 100), "white")
        portrait = Image.new("RGB", (100, 200), "white")
        try:
            landscape_processed, landscape_meta = preprocess_crop_with_metadata(
                landscape,
                mode="canonical",
                canonical_size=224,
            )
            portrait_processed, portrait_meta = preprocess_crop_with_metadata(
                portrait,
                mode="canonical",
                canonical_size=224,
            )
        finally:
            landscape.close()
            portrait.close()

        self.assertEqual(landscape_processed.size, (224, 224))
        self.assertEqual(landscape_meta["content_rect"], [0, 56, 224, 168])
        self.assertEqual(portrait_processed.size, (224, 224))
        self.assertEqual(portrait_meta["content_rect"], [56, 0, 168, 224])
        landscape_processed.close()
        portrait_processed.close()

    def test_embedding_view_metadata_carries_source_transform(self) -> None:
        image = Image.new("RGB", (320, 180), "gray")
        try:
            views, primary_bounds, metadata = make_embedding_crop_views(
                image,
                [40, 30, 140, 90],
                crop_mode="tight",
                padding_ratio=0.0,
                preprocess_mode="canonical",
                canonical_size=224,
                view_mode="tight_context",
            )
        finally:
            image.close()

        try:
            self.assertEqual(primary_bounds, (40, 30, 140, 90))
            self.assertEqual(len(views), 2)
            self.assertEqual(metadata[0]["source_image_size"], [320, 180])
            self.assertEqual(metadata[0]["source_crop_xyxy"], [40, 30, 140, 90])
            self.assertEqual(metadata[0]["source_crop_size"], [100, 60])
            self.assertEqual(metadata[0]["processed_input_size"], [224, 224])
            left, top, right, bottom = metadata[0]["processed_content_rect"]
            self.assertEqual((left, right), (0, 224))
            self.assertGreater(top, 0)
            self.assertLess(bottom, 224)
        finally:
            for view in views:
                view.close()


if __name__ == "__main__":
    unittest.main()
