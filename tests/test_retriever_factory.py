from __future__ import annotations

import unittest
from unittest import mock

import src.retriever_factory as retriever_factory


class RetrieverFactoryTests(unittest.TestCase):
    def test_turbopuffer_backend_builds_turbopuffer_retriever_only(self):
        sentinel = mock.sentinel.turbopuffer_retriever

        with (
            mock.patch.object(retriever_factory, "RETRIEVAL_BACKEND", "turbopuffer"),
            mock.patch("src.retriever_turbopuffer.TurbopufferRetriever", autospec=True) as tpuf_cls,
        ):
            tpuf_cls.return_value = sentinel
            retriever = retriever_factory.create_retriever(n_retrieval=17)

        self.assertIs(retriever, sentinel)
        tpuf_cls.assert_called_once_with(n_retrieval=17)

    def test_invalid_backend_surfaces_construction_failure(self):
        with (
            mock.patch.object(retriever_factory, "RETRIEVAL_BACKEND", "does-not-exist"),
            mock.patch("src.retriever_turbopuffer.TurbopufferRetriever", autospec=True) as tpuf_cls,
        ):
            with self.assertRaisesRegex(ValueError, "RETRIEVAL_BACKEND must be 'turbopuffer'"):
                retriever_factory.create_retriever(n_retrieval=9)

        tpuf_cls.assert_not_called()


if __name__ == "__main__":
    unittest.main()
