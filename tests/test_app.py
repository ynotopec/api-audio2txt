import asyncio
import unittest

import transformers


class _Model:
    def eval(self):
        return self

    def to(self, _device):
        return self


class _ModelLoader:
    @classmethod
    def from_pretrained(cls, *_args, **_kwargs):
        return _Model()


class _Processor:
    tokenizer = object()
    feature_extractor = object()


class _ProcessorLoader:
    @classmethod
    def from_pretrained(cls, *_args, **_kwargs):
        return _Processor()


PIPELINE_CALLS = []


def _pipeline_factory(*_args, **_kwargs):
    def infer(inputs, **_options):
        PIPELINE_CALLS.append(len(inputs))
        return [{"text": str(item["array"])} for item in inputs]

    return infer


# Patch heavyweight model creation before importing the application module.
transformers.AutoModelForSpeechSeq2Seq = _ModelLoader
transformers.AutoProcessor = _ProcessorLoader
transformers.pipeline = _pipeline_factory

import app  # noqa: E402


class AppTests(unittest.TestCase):
    def test_timestamp_modes(self):
        self.assertFalse(app._timestamp_mode(None, "json"))
        self.assertFalse(app._timestamp_mode(None, "text"))
        self.assertTrue(app._timestamp_mode(None, "srt"))
        self.assertEqual(app._timestamp_mode('["word"]', "json"), "word")

    def test_concurrent_requests_are_micro_batched(self):
        async def run_batch():
            results = await asyncio.gather(
                *[app._run_asr_async(i, "fr", False) for i in range(4)]
            )
            self.assertEqual(
                [result["text"] for result in results],
                ["0", "1", "2", "3"],
            )
            self.assertEqual(app.asr_queue.qsize(), 0)

        PIPELINE_CALLS.clear()
        asyncio.run(run_batch())
        self.assertEqual(PIPELINE_CALLS, [4])


if __name__ == "__main__":
    unittest.main()
