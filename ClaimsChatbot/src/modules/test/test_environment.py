import os
import unittest

import ollama
from dotenv import load_dotenv, find_dotenv
from neo4j import GraphDatabase

load_dotenv()


class TestEnvironment(unittest.TestCase):

    def test_env_file_exists(self):
        self.assertTrue(
            find_dotenv(),
            ".env file not found."
        )

    def env_variable_exists(self, variable_name):
        value = os.getenv(variable_name)

        self.assertIsNotNone(
            value,
            f"{variable_name} not found in .env file"
        )

        self.assertNotEqual(
            value.strip(),
            "",
            f"{variable_name} is empty"
        )

    def test_neo4j_variables(self):
        self.test_env_file_exists()

        self.env_variable_exists("NEO4J_URI")
        self.env_variable_exists("NEO4J_USERNAME")
        self.env_variable_exists("NEO4J_PASSWORD")
        self.env_variable_exists("NEO4J_DATABASE")

    def test_ollama_connection(self):
        try:
            client = ollama.Client(
                host="http://localhost:11434"
            )

            response = client.list()

            self.assertIsNotNone(response)

            print("\nOllama models:")
            for model in response.models:
                print(f"  - {model.model}")

        except Exception as e:
            self.fail(f"Ollama connection failed: {e}")

    def test_neo4j_connection(self):
        self.test_neo4j_variables()

        driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(
                os.getenv("NEO4J_USERNAME"),
                os.getenv("NEO4J_PASSWORD")
            )
        )

        try:
            driver.verify_connectivity()

            driver.execute_query(
                "RETURN true",
                database_=os.getenv("NEO4J_DATABASE")
            )

        except Exception as e:
            self.fail(f"Neo4j connection/query failed: {e}")

        finally:
            driver.close()


def suite():
    suite = unittest.TestSuite()

    suite.addTest(TestEnvironment("test_env_file_exists"))
    suite.addTest(TestEnvironment("test_neo4j_variables"))
    suite.addTest(TestEnvironment("test_ollama_connection"))
    suite.addTest(TestEnvironment("test_neo4j_connection"))

    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())
