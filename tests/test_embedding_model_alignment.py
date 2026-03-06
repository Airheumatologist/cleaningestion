import importlib.util
import os
import runpy
import sys
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


if __name__ == "__main__":
    unittest.main()
