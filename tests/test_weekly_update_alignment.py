import importlib.util
import subprocess
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
WEEKLY_UPDATE_PATH = REPO_ROOT / "scripts" / "updates" / "weekly_update.py"


def load_weekly_module():
    module_name = f"weekly_update_test_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, str(WEEKLY_UPDATE_PATH))
    module = importlib.util.module_from_spec(spec)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load weekly update module spec")
    spec.loader.exec_module(module)
    return module


class WeeklyUpdateAlignmentTests(unittest.TestCase):
    def test_dailymed_refresh_runs_lookup_after_ingest(self):
        module = load_weekly_module()
        with mock.patch.object(module.subprocess, "run") as run_mock:
            module.run_dailymed_refresh(weeks_back=2)

        scripts = [Path(call.args[0][1]).name for call in run_mock.call_args_list]
        self.assertEqual(
            scripts,
            [
                "03_download_dailymed.py",
                "04_prepare_dailymed_updates.py",
                "07_ingest_dailymed.py",
                "generate_drug_lookup.py",
            ],
        )
        self.assertIn("2", run_mock.call_args_list[0].args[0])
        self.assertIn("--delete-source", run_mock.call_args_list[2].args[0])

    def test_pmc_refresh_downloads_both_datasets_and_deletes_source(self):
        module = load_weekly_module()
        with mock.patch.object(module.subprocess, "run") as run_mock:
            module.run_pmc_refresh()

        pmc_download_cmd = run_mock.call_args_list[0].args[0]
        pmc_ingest_cmd = run_mock.call_args_list[1].args[0]
        self.assertIn("--datasets", pmc_download_cmd)
        self.assertIn("pmc_oa,author_manuscript", pmc_download_cmd)
        self.assertIn("--delete-source", pmc_ingest_cmd)

    def test_dailymed_refresh_fails_when_lookup_generation_fails(self):
        module = load_weekly_module()
        with mock.patch.object(module.subprocess, "run") as run_mock:
            run_mock.side_effect = [
                None,
                None,
                None,
                subprocess.CalledProcessError(returncode=1, cmd="generate_drug_lookup.py"),
            ]
            with self.assertRaises(subprocess.CalledProcessError):
                module.run_dailymed_refresh()

    def test_qdrant_dns_success_path(self):
        module = load_weekly_module()
        with (
            mock.patch.object(module.socket, "gethostbyname", return_value="172.18.0.2"),
            mock.patch.object(module.subprocess, "run") as run_mock,
        ):
            resolved = module._resolve_qdrant_url_for_weekly("http://qdrant:6333", "qdrant")
        self.assertEqual(resolved, "http://qdrant:6333")
        run_mock.assert_not_called()

    def test_qdrant_dns_failure_docker_inspect_success_path(self):
        module = load_weekly_module()
        with (
            mock.patch.object(module.socket, "gethostbyname", side_effect=OSError("dns failed")),
            mock.patch.object(
                module.subprocess,
                "run",
                return_value=subprocess.CompletedProcess(
                    args=["docker", "inspect"],
                    returncode=0,
                    stdout="172.18.0.2\n",
                    stderr="",
                ),
            ),
        ):
            resolved = module._resolve_qdrant_url_for_weekly("http://qdrant:6333", "qdrant")
        self.assertEqual(resolved, "http://172.18.0.2:6333")

    def test_qdrant_total_failure_is_actionable(self):
        module = load_weekly_module()
        with (
            mock.patch.object(module.socket, "gethostbyname", side_effect=OSError("dns failed")),
            mock.patch.object(
                module.subprocess,
                "run",
                return_value=subprocess.CompletedProcess(
                    args=["docker", "inspect"],
                    returncode=1,
                    stdout="",
                    stderr="Error: No such container",
                ),
            ),
        ):
            with self.assertRaises(RuntimeError) as exc:
                module._resolve_qdrant_url_for_weekly("http://qdrant:6333", "qdrant")
        self.assertIn("Set QDRANT_URL explicitly", str(exc.exception))

    def test_enforce_hnsw_indexing_uses_threshold_10000(self):
        module = load_weekly_module()
        client = mock.Mock()
        module.enforce_hnsw_indexing(client, "rag_pipeline")

        self.assertTrue(client.update_collection.called)
        kwargs = client.update_collection.call_args.kwargs
        self.assertEqual(kwargs["collection_name"], "rag_pipeline")
        self.assertEqual(kwargs["optimizers_config"].indexing_threshold, 10000)

    def test_main_enforces_hnsw_after_stages(self):
        module = load_weekly_module()
        args = SimpleNamespace(
            max_files=None,
            skip_pubmed=True,
            skip_dailymed=True,
            skip_pmc=True,
            min_year=2015,
            dailymed_weeks_back=1,
            throttle_seconds=0.5,
            batch_size=0,
        )
        client = mock.Mock()
        client.get_collection.return_value = SimpleNamespace(points_count=123)

        with (
            mock.patch.dict("os.environ", {"DEEPINFRA_API_KEY": "test-key"}, clear=False),
            mock.patch.object(module.argparse.ArgumentParser, "parse_args", return_value=args),
            mock.patch.object(module, "ensure_data_dirs"),
            mock.patch.object(module.IngestionConfig, "QDRANT_API_KEY", "qdrant-key"),
            mock.patch.object(module.IngestionConfig, "QDRANT_URL", "http://qdrant:6333"),
            mock.patch.object(module.IngestionConfig, "COLLECTION_NAME", "rag_pipeline"),
            mock.patch.object(module, "_resolve_qdrant_url_for_weekly", return_value="http://127.0.0.1:6333"),
            mock.patch.object(module, "EmbeddingProvider", return_value=object()),
            mock.patch.object(module, "QdrantClient", return_value=client),
        ):
            module.main()

        kwargs = client.update_collection.call_args.kwargs
        self.assertEqual(kwargs["collection_name"], "rag_pipeline")
        self.assertEqual(kwargs["optimizers_config"].indexing_threshold, 10000)


if __name__ == "__main__":
    unittest.main()
