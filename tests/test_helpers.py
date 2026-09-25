"""
Unit tests for the pure helpers in app.py.

These must NOT import app.py directly: the module loads the Whisper model at
import time (~1.6 GB, several seconds). The helpers under test are pure, so
they are extracted from the source file by AST and executed in a bare
namespace. That keeps the test suite runnable on CPU without a GPU or the model.

Run:
    python tests/test_helpers.py
"""
import ast
import json
import math
import os
import subprocess
import sys
import types
import unittest

import numpy as np
import soundfile as sf

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")

# Helpers that are pure Python (no torch/transformers/model needed).
WANTED = {
    "_timestamp_mode",
    "_segments",
    "_timestamp",
    "_format_time",
    "_to_srt",
    "_to_vtt",
    "_frame_rms_db",
    "_vad_keep_indices",
    "_map_trimmed_to_original",
    "_remap_segment_times",
    "_decode_budget",
    "_load_tokens",
    "_decode_ffmpeg",
}

SR = 16000


def load_pure_helpers():
    """Extract and exec the pure helper functions from app.py."""
    tree = ast.parse(open(APP_PATH, encoding="utf-8").read(), filename=APP_PATH)

    keep = []

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in WANTED:
            keep.append(node)

    found = {n.name for n in keep}
    missing = WANTED - found

    if missing:
        raise RuntimeError(f"helpers missing from app.py: {sorted(missing)}")

    # Module globals the helpers read. Imports are re-bound to the real modules
    # so an extracted helper behaves exactly as it does inside app.py.
    env = {
        "np": np,
        "json": json,
        "os": os,
        "math": math,
        "subprocess": subprocess,
        "sf": sf,
        "Any": object,
        "Dict": dict,
        "List": list,
        "Optional": object,
        "Tuple": tuple,
        "ASR_API_TOKENS": "",
        "AUTH_FILE": "/nonexistent-auth-file",
        "SR": SR,
        "ASR_MAX_NEW_TOKENS": 440,
        "ASR_MAX_TOTAL_TOKENS": 448,
        "ASR_VAD_ENABLED": True,
        "VAD_THRESHOLD_DB": -45.0,
        "VAD_MIN_SILENCE_MS": 500,
        "VAD_SPEECH_PAD_MS": 200,
        "VAD_JOIN_GAP_MS": 300,
    }

    mod = ast.Module(body=keep, type_ignores=[])
    exec(compile(mod, APP_PATH, "exec"), env)  # noqa: S102 - intentional for tests
    return types.SimpleNamespace(**{n: env[n] for n in found})


H = load_pure_helpers()


class TestTimestampMode(unittest.TestCase):
    def test_empty_defaults_to_true(self):
        self.assertIs(H._timestamp_mode(None), True)
        self.assertIs(H._timestamp_mode(""), True)

    def test_plain_word(self):
        self.assertEqual(H._timestamp_mode("word"), "word")

    def test_json_list_with_word(self):
        self.assertEqual(H._timestamp_mode('["word"]'), "word")
        self.assertEqual(H._timestamp_mode("[\"word\"]"), "word")

    def test_json_list_without_word(self):
        self.assertIs(H._timestamp_mode('["segment"]'), True)

    def test_json_string_word(self):
        self.assertEqual(H._timestamp_mode('"word"'), "word")

    def test_garbage_falls_back_to_true(self):
        self.assertIs(H._timestamp_mode("not-json-at-all"), True)


class TestFormatTime(unittest.TestCase):
    def test_srt_separator(self):
        self.assertEqual(H._format_time(0.0, True), "00:00:00,000")
        self.assertEqual(H._format_time(3661.5, True), "01:01:01,500")

    def test_vtt_separator(self):
        self.assertEqual(H._format_time(3661.5, False), "01:01:01.500")

    def test_negative_clamped(self):
        self.assertEqual(H._format_time(-5.0, True), "00:00:00,000")

    def test_exact_second_has_no_drift(self):
        self.assertEqual(H._format_time(10.0, True), "00:00:10,000")


class TestSegmentHelpers(unittest.TestCase):
    def test_segments_prefers_chunks(self):
        self.assertEqual(H._segments({"chunks": [1, 2], "segments": [3]}), [1, 2])

    def test_segments_falls_back(self):
        self.assertEqual(H._segments({"segments": [3]}), [3])

    def test_segments_empty(self):
        self.assertEqual(H._segments({}), [])

    def test_timestamp_tuple(self):
        self.assertEqual(H._timestamp({"timestamp": (1.0, 2.0)}), (1.0, 2.0))

    def test_timestamp_dict(self):
        self.assertEqual(H._timestamp({"timestamp": {"start": 1, "end": 2}}), (1.0, 2.0))

    def test_timestamp_missing_is_zero(self):
        self.assertEqual(H._timestamp({}), (0.0, 0.0))

    def test_timestamp_none_start(self):
        self.assertEqual(H._timestamp({"timestamp": (None, 2.0)}), (0.0, 2.0))

    def test_timestamp_open_ended(self):
        self.assertEqual(H._timestamp({"timestamp": (1.0, None)}), (1.0, 0.0))


class TestSrtVtt(unittest.TestCase):
    segs = [
        {"timestamp": (0.0, 1.5), "text": " hello"},
        {"timestamp": (1.5, 3.25), "text": " world "},
    ]

    def test_srt_structure(self):
        out = H._to_srt(self.segs)
        lines = out.splitlines()
        self.assertEqual(lines[0], "1")
        self.assertEqual(lines[1], "00:00:00,000 --> 00:00:01,500")
        self.assertEqual(lines[2], "hello")
        self.assertEqual(lines[4], "2")
        self.assertEqual(lines[5], "00:00:01,500 --> 00:00:03,250")

    def test_vtt_header(self):
        out = H._to_vtt(self.segs)
        self.assertTrue(out.startswith("WEBVTT"))
        self.assertIn("00:00:00.000 --> 00:00:01.500", out)

    def test_srt_empty(self):
        self.assertEqual(H._to_srt([]), "")


class TestDecodeBudget(unittest.TestCase):
    def test_scales_with_duration(self):
        short = H._decode_budget(5.0, False)
        long = H._decode_budget(60.0, False)
        self.assertGreater(long, short)

    def test_respects_model_limit(self):
        # 1000 s of audio must not ask for more than the 448-position decoder.
        self.assertLessEqual(H._decode_budget(1000.0, False), 448 - 8)

    def test_respects_configured_cap(self):
        self.assertLessEqual(H._decode_budget(60.0, False), 440)

    def test_timestamps_get_more_budget(self):
        self.assertGreater(H._decode_budget(30.0, True), H._decode_budget(30.0, False))

    def test_never_zero(self):
        self.assertGreaterEqual(H._decode_budget(0.0, False), 64)


class TestVad(unittest.TestCase):
    def _tone(self, seconds, amp=0.1):
        n = int(SR * seconds)
        t = np.arange(n) / SR
        return (amp * np.sin(2 * math.pi * 220 * t)).astype("float32")

    def _silence(self, seconds):
        return np.zeros(int(SR * seconds), dtype="float32")

    def test_short_audio_skipped(self):
        arr = np.concatenate([self._tone(0.5), self._silence(0.5)])
        self.assertIsNone(H._vad_keep_indices(arr))

    def test_pure_silence_skipped(self):
        # Never blank the output; let the model decide.
        self.assertIsNone(H._vad_keep_indices(self._silence(5.0)))

    def test_full_speech_untouched(self):
        self.assertIsNone(H._vad_keep_indices(self._tone(5.0)))

    def test_leading_trailing_silence_trimmed(self):
        arr = np.concatenate(
            [self._silence(2.0), self._tone(3.0), self._silence(2.0)]
        )
        kept, offsets = H._vad_keep_indices(arr)
        self.assertLess(kept.size, arr.size * 0.6)
        self.assertGreater(kept.size, arr.size * 0.3)

    def test_kept_span_covers_the_speech(self):
        speech = self._tone(3.0)
        arr = np.concatenate([self._silence(2.0), speech, self._silence(2.0)])
        trimmed, _ = H._vad_keep_indices(arr)
        # The speech must survive with most of its energy intact.
        self.assertGreater(float(np.sqrt(np.mean(trimmed**2))), 0.5 * 0.1)

    def test_gap_longer_than_merge_is_split(self):
        arr = np.concatenate([self._tone(2.0), self._silence(3.0), self._tone(2.0)])
        trimmed, mapping = H._vad_keep_indices(arr)
        self.assertEqual(mapping[0].size, 2)
        self.assertLess(trimmed.size, arr.size)

    def test_offsets_are_original_indices(self):
        """The mapping must place every kept region back on the original
        timeline, otherwise segment timestamps cannot be mapped."""
        arr = np.concatenate([self._silence(2.0), self._tone(3.0), self._silence(2.0)])
        trimmed, mapping = H._vad_keep_indices(arr)
        starts, lengths, orig_starts, orig_ends = mapping
        self.assertEqual(starts.size, lengths.size)
        self.assertEqual(starts.size, orig_starts.size)
        # The single speech region starts 2 s in, less the 200 ms speech pad.
        self.assertAlmostEqual(orig_starts[0], (2.0 - 0.2) * SR, delta=SR * 0.03)
        # Trimmed audio is shorter than the original.
        self.assertLess(trimmed.size, arr.size)


class TestVadJoinGap(unittest.TestCase):
    """Speech regions must not be spliced flush against each other: the decoder
    needs the phrase boundary or it enters a repetition loop."""

    def _two_regions(self):
        tone = (0.1 * np.sin(2 * math.pi * 220 * np.arange(3 * SR) / SR)).astype("float32")
        return np.concatenate([np.zeros(2 * SR, dtype="float32"), tone,
                               np.zeros(3 * SR, dtype="float32"), tone])

    def test_gap_inserted_between_regions(self):
        arr = self._two_regions()
        trimmed, mapping = H._vad_keep_indices(arr)
        starts, lengths, orig_starts, orig_ends = mapping
        self.assertEqual(starts.size, 2, "expected two kept regions")
        expected_gap = int(SR * 300 / 1000)
        between = starts[1] - (starts[0] + lengths[0])
        self.assertEqual(between, expected_gap)

    def test_trimmed_still_shorter_than_original(self):
        arr = self._two_regions()
        trimmed, _ = H._vad_keep_indices(arr)
        # 11 s original, ~6 s speech + 0.3 s gap.
        self.assertLess(trimmed.size, arr.size)
        self.assertGreater(trimmed.size, 6 * SR)

    def test_mapping_is_monotonic(self):
        arr = self._two_regions()
        _, mapping = H._vad_keep_indices(arr)
        starts, lengths, orig_starts, orig_ends = mapping
        self.assertTrue(np.all(orig_ends > orig_starts))
        self.assertTrue(np.all(np.diff(orig_starts) > 0))


class TestMapTrimmedToOriginal(unittest.TestCase):
    def _mapping(self):
        # Two regions: trimmed [0,1000) <- original [32000, 48000)
        #             trimmed [1300,2300) <- original [100000,116000)
        return (
            np.array([0.0, 1300.0]),
            np.array([1000.0, 1000.0]),
            np.array([32000.0, 100000.0]),
            np.array([48000.0, 116000.0]),
        )

    def test_first_region_maps_linearly(self):
        m = self._mapping()
        self.assertAlmostEqual(H._map_trimmed_to_original(0, m), 32000.0)
        self.assertAlmostEqual(H._map_trimmed_to_original(500, m), 40000.0)
        self.assertAlmostEqual(H._map_trimmed_to_original(1000, m), 48000.0)

    def test_second_region_maps_to_its_own_original_offset(self):
        m = self._mapping()
        self.assertAlmostEqual(H._map_trimmed_to_original(1300, m), 100000.0)
        self.assertAlmostEqual(H._map_trimmed_to_original(1800, m), 108000.0)

    def test_gap_position_clamps_to_region_end(self):
        m = self._mapping()
        # 1100 is inside the inserted gap (1000..1300).
        self.assertAlmostEqual(H._map_trimmed_to_original(1100, m), 48000.0)

    def test_negative_clamps_to_first_region(self):
        m = self._mapping()
        self.assertGreaterEqual(H._map_trimmed_to_original(-100, m), 0.0)

    def test_far_past_end_is_clamped(self):
        m = self._mapping()
        self.assertLessEqual(H._map_trimmed_to_original(10**7, m), 116000.0)


class TestRemapSegmentTimes(unittest.TestCase):
    """VAD trims silence, so decoded timestamps are relative to the trimmed
    audio. These must be mapped back or every srt/vtt cue after the first
    removed silence is wrong."""

    def _mapping_with_gap(self):
        # 2 s silence, 3 s speech, 2 s silence at 16 kHz.
        arr = np.concatenate(
            [
                np.zeros(2 * SR, dtype="float32"),
                (0.1 * np.sin(2 * math.pi * 220 * np.arange(3 * SR) / SR)).astype("float32"),
                np.zeros(2 * SR, dtype="float32"),
            ]
        )
        return H._vad_keep_indices(arr)

    def test_no_mapping_is_passthrough(self):
        segs = [{"timestamp": [0.0, 1.0], "text": "a"}]
        self.assertEqual(H._remap_segment_times(segs, None), segs)

    def test_zero_start_stays_zero(self):
        _, mapping = self._mapping_with_gap()
        segs = [{"timestamp": [0.0, 1.0], "text": "a"}]
        out = H._remap_segment_times(segs, mapping)
        self.assertEqual(out[0]["timestamp"][0], 0.0)

    def test_timestamps_move_forward(self):
        _, mapping = self._mapping_with_gap()
        segs = [{"timestamp": [0.5, 2.5], "text": "a"}]
        out = H._remap_segment_times(segs, mapping)
        # Trimming the leading 2 s must push every cue later in wall-clock time.
        self.assertGreater(out[0]["timestamp"][0], 0.5)
        self.assertGreater(out[0]["timestamp"][1], 2.5)

    def test_list_shape_preserved(self):
        _, mapping = self._mapping_with_gap()
        segs = [{"timestamp": [0.5, 2.5], "text": "a"}]
        out = H._remap_segment_times(segs, mapping)
        self.assertIsInstance(out[0]["timestamp"], list)
        self.assertEqual(len(out[0]["timestamp"]), 2)

    def test_far_past_end_is_clamped_to_original_duration(self):
        _, mapping = self._mapping_with_gap()
        segs = [{"timestamp": [0.0, 1e6], "text": "a"}]
        out = H._remap_segment_times(segs, mapping)
        self.assertLessEqual(out[0]["timestamp"][1], 7.0)

    def test_never_shrinks_a_timestamp(self):
        _, mapping = self._mapping_with_gap()
        segs = [{"timestamp": [1.0, 2.0], "text": "a"}]
        out = H._remap_segment_times(segs, mapping)
        self.assertGreaterEqual(out[0]["timestamp"][0], 1.0)
        self.assertGreaterEqual(out[0]["timestamp"][1], 2.0)

    def test_text_and_other_fields_preserved(self):
        _, mapping = self._mapping_with_gap()
        segs = [{"timestamp": [0.5, 2.5], "text": "hello", "id": 7}]
        out = H._remap_segment_times(segs, mapping)
        self.assertEqual(out[0]["text"], "hello")
        self.assertEqual(out[0]["id"], 7)


class TestFrameRms(unittest.TestCase):
    def test_silence_is_floor(self):
        db = H._frame_rms_db(np.zeros(SR, dtype="float32"), 320)
        self.assertEqual(db.size, SR // 320)
        self.assertTrue(np.all(db < -100))

    def test_loud_is_higher(self):
        quiet = H._frame_rms_db(np.full(SR, 0.01, dtype="float32"), 320)
        loud = H._frame_rms_db(np.full(SR, 0.5, dtype="float32"), 320)
        self.assertTrue(np.all(loud > quiet))


class TestAuthTokens(unittest.TestCase):
    def test_parses_comma_and_newline(self):
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("tokenA\ntokenB\n")
            path = f.name
        try:
            self.assertEqual(H._load_tokens(path), {"tokenA", "tokenB"})
        finally:
            os.unlink(path)


class TestFfmpegDecode(unittest.TestCase):
    """The ffmpeg fallback must accept a format soundfile rejects."""

    def test_mp3_roundtrip(self):
        import subprocess
        import tempfile

        n = SR * 2
        t = np.arange(n) / SR
        tone = (0.2 * np.sin(2 * math.pi * 440 * t)).astype("float32")

        with tempfile.TemporaryDirectory() as tmpdir:
            src = os.path.join(tmpdir, "tone.wav")
            mp3 = os.path.join(tmpdir, "tone.mp3")
            sf.write(src, tone, SR)
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", src, mp3], check=True
            )

            with open(mp3, "rb") as f:
                data = f.read()

            arr, sr = H._decode_ffmpeg(data)
            self.assertEqual(sr, SR)
            self.assertEqual(str(arr.dtype), "float32")
            self.assertAlmostEqual(arr.size / SR, 2.0, delta=0.2)
            self.assertGreater(float(np.abs(arr).max()), 0.05)

    def test_garbage_raises(self):
        with self.assertRaises(ValueError):
            H._decode_ffmpeg(b"this is definitely not audio data")


if __name__ == "__main__":
    unittest.main(verbosity=2)
