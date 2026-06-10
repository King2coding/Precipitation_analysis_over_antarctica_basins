from antarctic_precip_pmb.config import load_yaml, missing_path_keys, path_status


def test_example_paths_has_required_keys():
    cfg = load_yaml("config/example_paths.yaml")
    assert missing_path_keys(cfg) == []


def test_example_paths_are_placeholders_or_repo_outputs():
    cfg = load_yaml("config/example_paths.yaml")
    statuses = {row["key"]: row["status"] for row in path_status(cfg)}
    assert statuses["basins_dir"] == "placeholder"
    assert statuses["output_dir"] in {"exists", "missing"}
