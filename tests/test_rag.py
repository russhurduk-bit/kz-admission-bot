import unittest
import uuid
from pathlib import Path

import rag


class RetrievalIsolationTests(unittest.TestCase):
    def test_search_returns_only_the_requesting_users_memories(self):
        artifact_dir = Path(__file__).parent / "artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        rag.DB_PATH = str(artifact_dir / f"memory_{uuid.uuid4().hex}.db")
        rag.init_db()
        rag.store_memory(101, "Tesla internship in Berlin", "applications")
        rag.store_memory(202, "Tesla internship in another account", "applications")

        own = rag.search_memories(101, "Tesla")
        other = rag.search_memories(202, "Tesla")
        self.assertEqual(len(own), 1)
        self.assertEqual(len(other), 1)
        self.assertEqual(own[0]["content"], "Tesla internship in Berlin")
        self.assertEqual(other[0]["content"], "Tesla internship in another account")
