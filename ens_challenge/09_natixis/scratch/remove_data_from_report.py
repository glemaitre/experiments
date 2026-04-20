# %%
import skore
from pathlib import Path

workspace = Path("../skore-artifacts")
project = skore.Project(name="natixis", mode="local", workspace=workspace)

# %%
report = project.get("283129269553560288153433140892403241521")

# %%
# remove data from the cross-validation report
del report._data["_skrub_X"]
del report._data["_skrub_y"]

# %%
# remove the data from the underlying estimator report
for estimator_report in report.estimator_reports_:
    del estimator_report._train_data["_skrub_X"]
    del estimator_report._train_data["_skrub_y"]
    del estimator_report._test_data["_skrub_X"]
    del estimator_report._test_data["_skrub_y"]
    estimator_report.clear_cache()

# %%
import cloudpickle

out_path = Path(__file__).resolve().parent / "report_cloudpickle.pkl"
out_path.write_bytes(cloudpickle.dumps(report))
print(f"Wrote {out_path} ({out_path.stat().st_size} bytes)")

# %%
out_path = Path(__file__).resolve().parent / "estimator.pkl"
out_path.write_bytes(cloudpickle.dumps(report._raw_estimator))
print(f"Wrote {out_path} ({out_path.stat().st_size} bytes)")

# %%
split_indices_path = Path(__file__).resolve().parent / "split_indices.pkl"
split_indices_path.write_bytes(cloudpickle.dumps(report._split_indices))
print(f"Wrote {split_indices_path} ({split_indices_path.stat().st_size} bytes)")

# %%
