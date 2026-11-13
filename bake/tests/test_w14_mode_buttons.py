import json
import pathlib
import tempfile
import unittest

import numpy as np

from tests.test_w03_cuda import make_image
from tests.test_w04_ofx import HOST, PLUGIN
from tests.test_w11_match import scene, srgb8
from tests.test_w11_match_native import host_run, write_png


@unittest.skipUnless(HOST.exists() and PLUGIN.exists(), "plugin or host not built")
class ModeButtons(unittest.TestCase):
    def test_switch_and_restore_preserves_render_and_dropdowns(self):
        image = make_image("dwg_di", 61)
        with tempfile.TemporaryDirectory() as tmp:
            baseline, before, _ = host_run(image, tmp)
            result, after, _ = host_run(image, tmp, extra=["--ui-buttons", "1"])
            self.assertEqual(baseline.returncode, 0, baseline.stdout + baseline.stderr)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("mode_buttons_valid=1", result.stdout)
            np.testing.assert_array_equal(before, after)

    def test_native_rows_preserve_choice_values_render_and_dropdowns(self):
        image = make_image("dwg_di", 64)
        with tempfile.TemporaryDirectory() as tmp:
            baseline, before, _ = host_run(image, tmp)
            result, after, _ = host_run(image, tmp, extra=["--ui-rows", "1", "--ui-buttons", "1"])
            self.assertEqual(baseline.returncode, 0, baseline.stdout + baseline.stderr)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("mode_rows_valid=1", result.stdout)
            self.assertIn("mode_buttons_valid=1", result.stdout)
            self.assertIn("params=52 clips=2", result.stdout)
            np.testing.assert_array_equal(before, after)

    def test_current_option_does_not_add_an_undo_edit(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, _, _ = host_run(make_image("dwg_di", 62), tmp, extra=["--press", "glowQualityOption0"])
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertNotIn("param_edit begin=", result.stdout)

    def test_match_only_button_drives_create_grade(self):
        with tempfile.TemporaryDirectory() as tmp:
            reference = pathlib.Path(tmp) / "ref.png"
            write_png(reference, np.round(srgb8(scene(32, 64)) * 255).astype(np.uint8))
            result, output, state = host_run(make_image("dwg_di", 63), tmp,
                extra=["--press", "matchSourceOption1", "--match-create", str(reference)])
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("match_create_status=0", result.stdout)
            self.assertEqual(json.loads(state)["source"], "only")
            self.assertTrue(np.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
