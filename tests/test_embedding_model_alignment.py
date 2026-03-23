import importlib.util
import os
import runpy
import sys
import threading
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_CONFIG_PATH = REPO_ROOT / "src" / "config.py"
INGESTION_CONFIG_PATH = REPO_ROOT / "scripts" / "config_ingestion.py"
INGESTION_UTILS_PATH = REPO_ROOT / "scripts" / "ingestion_utils.py"


def run_module_with_env(module_path: Path, env: dict):
    with (
        mock.patch.dict(os.environ, env, clear=True),
        mock.patch("dotenv.load_dotenv", return_value=False),
    ):
        return runpy.run_path(str(module_path))


@contextmanager
def loaded_ingestion_utils(env: dict):
    scripts_dir = str(REPO_ROOT / "scripts")
    added_sys_path = False
    previous_config_module = sys.modules.pop("config_ingestion", None)
    module_name = f"ingestion_utils_test_{uuid.uuid4().hex}"
    previous_test_module = sys.modules.pop(module_name, None)

    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
        added_sys_path = True

    try:
        with (
            mock.patch.dict(os.environ, env, clear=True),
            mock.patch("dotenv.load_dotenv", return_value=False),
        ):
            cfg_spec = importlib.util.spec_from_file_location("config_ingestion", str(INGESTION_CONFIG_PATH))
            if cfg_spec is None or cfg_spec.loader is None:
                raise RuntimeError("Failed to load config_ingestion spec")
            cfg_module = importlib.util.module_from_spec(cfg_spec)
            sys.modules["config_ingestion"] = cfg_module
            cfg_spec.loader.exec_module(cfg_module)

            util_spec = importlib.util.spec_from_file_location(module_name, str(INGESTION_UTILS_PATH))
            if util_spec is None or util_spec.loader is None:
                raise RuntimeError("Failed to load ingestion_utils spec")
            util_module = importlib.util.module_from_spec(util_spec)
            sys.modules[module_name] = util_module
            util_spec.loader.exec_module(util_module)
            yield util_module
    finally:
        sys.modules.pop("config_ingestion", None)
        sys.modules.pop(module_name, None)
        if previous_config_module is not None:
            sys.modules["config_ingestion"] = previous_config_module
        if previous_test_module is not None:
            sys.modules[module_name] = previous_test_module
        if added_sys_path:
            try:
                sys.path.remove(scripts_dir)
            except ValueError:
                pass


class EmbeddingModelAlignmentTests(unittest.TestCase):
    def test_runtime_defaults_to_non_batch_and_groq_default(self):
        cfg = run_module_with_env(SRC_CONFIG_PATH, {})
        self.assertEqual(cfg["LLM_PROVIDER"], "groq")
        self.assertEqual(cfg["EMBEDDING_MODEL"], "Qwen/Qwen3-Embedding-0.6B")

    def test_runtime_embedding_prefers_runtime_split_var_over_legacy(self):
        cfg = run_module_with_env(
            SRC_CONFIG_PATH,
            {
                "RUNTIME_EMBEDDING_MODEL": "runtime-model",
                "EMBEDDING_MODEL": "legacy-model",
            },
        )
        self.assertEqual(cfg["EMBEDDING_MODEL"], "runtime-model")

    def test_ingestion_defaults_to_batch_model(self):
        cfg = run_module_with_env(INGESTION_CONFIG_PATH, {})
        ingestion_config = cfg["IngestionConfig"]
        self.assertEqual(ingestion_config.EMBEDDING_MODEL, "Qwen/Qwen3-Embedding-0.6B-batch")

    def test_ingestion_embedding_prefers_split_var_over_legacy(self):
        cfg = run_module_with_env(
            INGESTION_CONFIG_PATH,
            {
                "INGESTION_EMBEDDING_MODEL": "ingestion-model",
                "EMBEDDING_MODEL": "legacy-model",
            },
        )
        ingestion_config = cfg["IngestionConfig"]
        self.assertEqual(ingestion_config.EMBEDDING_MODEL, "ingestion-model")

    def test_ingestion_embedding_provider_uses_deepinfra_base_url_override(self):
        with loaded_ingestion_utils(
            {
                "DEEPINFRA_API_KEY": "test-key",
                "DEEPINFRA_BASE_URL": "https://custom.deepinfra.local/v1/openai",
            }
        ) as ingestion_utils:
            ingestion_utils.IngestionConfig.EMBEDDING_PROVIDER = "deepinfra"
            ingestion_utils.IngestionConfig.EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B-batch"
            with mock.patch("openai.OpenAI") as openai_cls:
                ingestion_utils.EmbeddingProvider()

        kwargs = openai_cls.call_args.kwargs
        self.assertEqual(kwargs["base_url"], "https://custom.deepinfra.local/v1/openai")
        self.assertEqual(kwargs["api_key"], "test-key")

    def test_ingestion_embedding_provider_builds_one_client_per_key(self):
        with loaded_ingestion_utils(
            {
                "DEEPINFRA_API_KEYS": "key-a, key-b ,key-a",
                "DEEPINFRA_API_KEY": "key-c",
            }
        ) as ingestion_utils:
            ingestion_utils.IngestionConfig.EMBEDDING_PROVIDER = "deepinfra"
            ingestion_utils.IngestionConfig.EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B-batch"
            with mock.patch("openai.OpenAI") as openai_cls:
                provider = ingestion_utils.EmbeddingProvider()

        self.assertEqual(openai_cls.call_count, 3)
        self.assertEqual(
            [call.kwargs["api_key"] for call in openai_cls.call_args_list],
            ["key-a", "key-b", "key-c"],
        )
        self.assertEqual(provider.openai_client, provider.openai_clients[0])

    def test_ingestion_embedding_provider_fails_over_across_keys_and_preserves_order(self):
        class FakeEmbeddingItem:
            def __init__(self, index, embedding):
                self.index = index
                self.embedding = embedding

        class FakeEmbeddingResponse:
            def __init__(self, data):
                self.data = data

        first_client = mock.Mock()
        second_client = mock.Mock()
        first_client.embeddings.create.side_effect = Exception("429 Too Many Requests")
        second_client.embeddings.create.return_value = FakeEmbeddingResponse(
            [
                FakeEmbeddingItem(1, [2.0, 2.5]),
                FakeEmbeddingItem(0, [1.0, 1.5]),
            ]
        )

        with loaded_ingestion_utils(
            {
                "DEEPINFRA_API_KEYS": "key-a,key-b",
                "EMBEDDING_REQUEST_MAX_RETRIES": "2",
            }
        ) as ingestion_utils:
            ingestion_utils.IngestionConfig.EMBEDDING_PROVIDER = "deepinfra"
            ingestion_utils.IngestionConfig.EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B-batch"
            with (
                mock.patch("openai.OpenAI", side_effect=[first_client, second_client]),
                mock.patch.object(ingestion_utils.time, "sleep", return_value=None),
            ):
                provider = ingestion_utils.EmbeddingProvider()
                embeddings = provider.embed_batch(["alpha", "beta"])

        self.assertEqual(embeddings, [[1.0, 1.5], [2.0, 2.5]])
        self.assertEqual(first_client.embeddings.create.call_count, 1)
        self.assertEqual(second_client.embeddings.create.call_count, 1)

    def test_embedding_provider_shapes_batches_by_token_budget(self):
        class FakeEmbeddingItem:
            def __init__(self, index, embedding):
                self.index = index
                self.embedding = embedding

        class FakeEmbeddingResponse:
            def __init__(self, data):
                self.data = data

        def _make_response(input_texts):
            return FakeEmbeddingResponse(
                [FakeEmbeddingItem(i, [float(i)]) for i in range(len(input_texts))]
            )

        with loaded_ingestion_utils(
            {
                "DEEPINFRA_API_KEY": "test-key",
                "EMBEDDING_BATCH_SIZE": "64",
            }
        ) as ingestion_utils:
            ingestion_utils.IngestionConfig.EMBEDDING_PROVIDER = "deepinfra"
            ingestion_utils.IngestionConfig.EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B-batch"
            fake_client = mock.Mock()
            fake_client.embeddings.create.side_effect = (
                lambda model, input, encoding_format: _make_response(input)
            )
            with mock.patch("openai.OpenAI", return_value=fake_client):
                provider = ingestion_utils.EmbeddingProvider()
                provider._max_input_tokens_per_request = 100
                texts = [
                    " ".join(["w"] * 40),  # ~53 tokens
                    " ".join(["w"] * 40),  # ~53 tokens
                    " ".join(["w"] * 40),  # ~53 tokens
                ]
                shaped = provider._shape_embedding_request_batches(
                    texts, batch_size=ingestion_utils.IngestionConfig.EMBEDDING_BATCH_SIZE
                )
                provider.embed_batch(texts)

        self.assertEqual(len(shaped), 3)
        # 3 items should split into 3 calls because 2 texts exceed token budget.
        self.assertEqual(fake_client.embeddings.create.call_count, 3)

    def test_embedding_provider_honors_retry_after_on_rate_limit(self):
        class FakeResponse:
            def __init__(self, headers):
                self.headers = headers
                self.status_code = 429

        class RateLimitError(Exception):
            def __init__(self, message, headers):
                super().__init__(message)
                self.response = FakeResponse(headers)
                self.status_code = 429

        class FakeEmbeddingItem:
            def __init__(self, index, embedding):
                self.index = index
                self.embedding = embedding

        class FakeEmbeddingResponse:
            def __init__(self, data):
                self.data = data

        first_client = mock.Mock()
        second_client = mock.Mock()
        first_client.embeddings.create.side_effect = [
            RateLimitError("429 Too Many Requests", {"Retry-After": "9"}),
        ]
        second_client.embeddings.create.return_value = FakeEmbeddingResponse(
            [FakeEmbeddingItem(0, [1.0, 2.0])]
        )

        sleep_calls = []
        sleep_lock = threading.Lock()

        def _record_sleep(seconds):
            with sleep_lock:
                sleep_calls.append(seconds)

        with loaded_ingestion_utils(
            {
                "DEEPINFRA_API_KEYS": "key-a,key-b",
                "EMBEDDING_REQUEST_MAX_RETRIES": "2",
            }
        ) as ingestion_utils:
            ingestion_utils.IngestionConfig.EMBEDDING_PROVIDER = "deepinfra"
            ingestion_utils.IngestionConfig.EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B-batch"
            with (
                mock.patch("openai.OpenAI", side_effect=[first_client, second_client]),
                mock.patch.object(ingestion_utils.time, "sleep", side_effect=_record_sleep),
            ):
                provider = ingestion_utils.EmbeddingProvider()
                provider.embed_batch(["alpha"])

        self.assertTrue(any(call >= 9.0 for call in sleep_calls))

    def test_ingestion_embedding_provider_attempts_model_fallback_after_repeated_5xx(self):
        class ServerError(Exception):
            def __init__(self, message):
                super().__init__(message)
                self.status_code = 500

        class FakeEmbeddingItem:
            def __init__(self, index, embedding):
                self.index = index
                self.embedding = embedding

        class FakeEmbeddingResponse:
            def __init__(self, data):
                self.data = data

        calls = []

        def _make_side_effect(client_name):
            def _inner(model, input, encoding_format):
                calls.append((client_name, model, list(input)))
                if model == "model-primary":
                    raise ServerError("500 upstream error")
                return FakeEmbeddingResponse(
                    [
                        FakeEmbeddingItem(1, [2.0, 2.5]),
                        FakeEmbeddingItem(0, [1.0, 1.5]),
                    ]
                )
            return _inner

        first_client = mock.Mock()
        second_client = mock.Mock()
        first_client.embeddings.create.side_effect = _make_side_effect("key-a")
        second_client.embeddings.create.side_effect = _make_side_effect("key-b")

        with loaded_ingestion_utils(
            {
                "DEEPINFRA_API_KEYS": "key-a,key-b",
                "INGESTION_EMBEDDING_MODELS": "model-primary,model-fallback",
                "EMBEDDING_REQUEST_MAX_RETRIES": "4",
                "EMBEDDING_CLIENT_FAILURE_THRESHOLD": "2",
            }
        ) as ingestion_utils:
            ingestion_utils.IngestionConfig.EMBEDDING_PROVIDER = "deepinfra"
            ingestion_utils.IngestionConfig.EMBEDDING_MODEL = "legacy-single-model"
            with (
                mock.patch("openai.OpenAI", side_effect=[first_client, second_client]),
                mock.patch.object(ingestion_utils.time, "sleep", return_value=None),
            ):
                provider = ingestion_utils.EmbeddingProvider()
                embeddings = provider.embed_batch(["alpha", "beta"])

        self.assertEqual(embeddings, [[1.0, 1.5], [2.0, 2.5]])
        self.assertGreaterEqual(len(calls), 3)
        self.assertEqual(calls[0][1], "model-primary")
        self.assertEqual(calls[1][1], "model-primary")
        self.assertIn("model-fallback", [model for _client, model, _input in calls])

    def test_ingestion_embedding_provider_opens_circuit_for_unhealthy_route(self):
        class ServerError(Exception):
            def __init__(self, message):
                super().__init__(message)
                self.status_code = 500

        class FakeEmbeddingItem:
            def __init__(self, index, embedding):
                self.index = index
                self.embedding = embedding

        class FakeEmbeddingResponse:
            def __init__(self, data):
                self.data = data

        first_client = mock.Mock()
        second_client = mock.Mock()
        first_client.embeddings.create.side_effect = ServerError("500 first route unhealthy")
        second_client.embeddings.create.return_value = FakeEmbeddingResponse(
            [FakeEmbeddingItem(0, [1.0, 1.5])]
        )

        with loaded_ingestion_utils(
            {
                "DEEPINFRA_API_KEYS": "key-a,key-b",
                "EMBEDDING_REQUEST_MAX_RETRIES": "3",
                "EMBEDDING_CLIENT_FAILURE_THRESHOLD": "1",
                "EMBEDDING_CLIENT_COOLDOWN_SECONDS": "60",
            }
        ) as ingestion_utils:
            ingestion_utils.IngestionConfig.EMBEDDING_PROVIDER = "deepinfra"
            ingestion_utils.IngestionConfig.EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B-batch"
            with (
                mock.patch("openai.OpenAI", side_effect=[first_client, second_client]),
                mock.patch.object(ingestion_utils.time, "sleep", return_value=None),
            ):
                provider = ingestion_utils.EmbeddingProvider()
                provider.embed_batch(["alpha"])
                provider.embed_batch(["beta"])

        self.assertEqual(first_client.embeddings.create.call_count, 1)
        self.assertEqual(second_client.embeddings.create.call_count, 2)


if __name__ == "__main__":
    unittest.main()
