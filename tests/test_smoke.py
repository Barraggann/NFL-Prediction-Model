def test_dependencias_se_importan():
    import matplotlib  # noqa: F401
    import nflreadpy  # noqa: F401
    import pandas  # noqa: F401
    import polars  # noqa: F401
    import sklearn  # noqa: F401


def test_paquete_se_importa():
    import nfl_model

    assert nfl_model.ROOT.name == "nfl-model"
