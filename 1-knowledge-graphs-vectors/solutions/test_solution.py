def test_create_and_embed_chunks(test_helpers, monkeypatch):
    output = test_helpers.run_module(monkeypatch, "create_and_embed_chunks")
    assert len(output) > 0

def test_build_graph(test_helpers, monkeypatch):
    import os
    from neo4j import GraphDatabase

    test_helpers.run_module(
        monkeypatch,
        "build_graph"
    )
    
    URI = os.getenv("NEO4J_URI")
    AUTH = (os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
    
    with GraphDatabase.driver(URI, auth=AUTH) as driver:

        driver.verify_connectivity()

        result = test_helpers.run_cypher(
            driver,
            "RETURN EXISTS ((:Chunk)-[]-()) as exists"
            )

        print(result)

    assert result[0]["exists"]