def test_vector_retriever(test_helpers, monkeypatch):
    output = test_helpers.run_module(monkeypatch, "vector_retriever")
    assert len(output) > 0

def test_vector_cypher_retriever(test_helpers, monkeypatch):
    output = test_helpers.run_module(monkeypatch, "vector_cypher_retriever")
    assert len(output) > 0

def test_hybrid_retriever(test_helpers, monkeypatch):
    output = test_helpers.run_module(monkeypatch, "hybrid_retriever")
    assert len(output) > 0

def test_hybrid_cypher_retriever(test_helpers, monkeypatch):
    output = test_helpers.run_module(monkeypatch, "hybrid_cypher_retriever")
    assert len(output) > 0

def test_text2cypher_retriever(test_helpers, monkeypatch):
    output = test_helpers.run_module(monkeypatch, "text2cypher_retriever")
    assert len(output) > 0

def test_multimodel_app(test_helpers, monkeypatch):
    output = test_helpers.run_module(monkeypatch, "multimodal_app")
    assert "Movie title:"  in output

