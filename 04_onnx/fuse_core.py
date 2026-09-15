#!/usr/bin/env python3
"""
Build a single fused ONNX graph for the v3 (RH 5-99) M_aug surrogate: 4 independent
CatBoost regressors (Anet_sim, Ci_sim, Tleaf_sim, gs_sim), all sharing the exact same
15-feature input vector (STAGE1_FEATURES) -- no inter-target dependency chain, so fusing
is just "load each model's ONNX graph, prefix its internal node/init names, point every
graph's single input at one shared 'features' input, collect the 4 outputs".

Usage:
  python fuse_core.py --rep rep01 --n 10000000 --output ../onnx/Maug_rep01_n10000000_fused.onnx
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper
from catboost import CatBoostRegressor

DEFAULT_MODELS_ROOT = Path(__file__).resolve().parents[1] / "03_training" / "out" / "Maug"
TARGETS = ["Anet_sim", "Ci_sim", "Tleaf_sim", "gs_sim"]

FEATURES = [
    "Aj_init", "Ac_init", "Ap_init",
    "T_air", "RH", "wind", "PFD", "CO2", "w", "VPD",
    "Vcm25", "Jm25", "g0", "g1", "Rd25",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--rep", default="rep01")
    p.add_argument("--n", type=int, default=10_000_000)
    p.add_argument("--production-root", type=Path, default=DEFAULT_MODELS_ROOT)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--metrics-csv", type=Path, default=None,
                    help="per_model_metrics.csv for the Zenodo deposit layout (best_iter source "
                         "when production-root has no stage2_metrics.csv alongside the models)")
    return p.parse_args()


def _inline_model(model: onnx.ModelProto, prefix: str, new_input: str, new_output: str
                   ) -> Tuple[List[onnx.NodeProto], List[onnx.TensorProto], List[onnx.ValueInfoProto]]:
    m = copy.deepcopy(model)
    g = m.graph
    old_input = g.input[0].name
    old_output = g.output[0].name
    mapping: Dict[str, str] = {old_input: new_input, old_output: new_output}

    def rename(name: str) -> str:
        if name == "":
            return name
        if name in mapping:
            return mapping[name]
        mapped = f"{prefix}{name}"
        mapping[name] = mapped
        return mapped

    for init in g.initializer:
        init.name = rename(init.name)
    for vi in g.value_info:
        vi.name = rename(vi.name)
    for node in g.node:
        for i, v in enumerate(node.input):
            node.input[i] = rename(v)
        for i, v in enumerate(node.output):
            node.output[i] = rename(v)
        node.name = f"{prefix}{node.name}" if node.name else f"{prefix}node"

    return list(g.node), list(g.initializer), list(g.value_info)


def _merge_opsets(models: List[onnx.ModelProto]) -> List[onnx.OperatorSetIdProto]:
    versions: Dict[str, int] = {}
    for m in models:
        for op in m.opset_import:
            versions[op.domain] = max(versions.get(op.domain, 0), op.version)
    merged = [helper.make_opsetid(d, v) for d, v in versions.items()]
    if "" not in versions:
        merged.append(helper.make_opsetid("", 13))
    return merged


def cbm_to_onnx_graph(cbm_path: Path, best_iter: int | None) -> onnx.ModelProto:
    """Load a trained .cbm CatBoost model, shrink it to the best_iteration actually used
    (early-stopping means the saved model already has best_iteration <= max_iter trees, but
    CatBoost's .cbm keeps every trained tree up to `stopped_iteration`; shrink() drops the
    dead tail so the ONNX export only encodes trees that matter), then return its
    single-input/single-output ONNX graph in-memory, output reshaped to (N, 1).

    `best_iter` is CatBoost's 0-based best_iteration_ index, so keeping trees [0, best_iter]
    inclusive means ntree_end=best_iter+1 -- shrink()'s ntree_end is exclusive."""
    model = CatBoostRegressor()
    model.load_model(str(cbm_path))
    n_before = model.tree_count_
    if best_iter is not None and best_iter + 1 < n_before:
        model.shrink(ntree_end=best_iter + 1)
        print(f"    shrunk {cbm_path.stem}: {n_before} -> {model.tree_count_} trees")
    else:
        print(f"    {cbm_path.stem}: {n_before} trees (no shrink)")
    tmp_path = cbm_path.with_suffix(".tmp_export.onnx")
    model.save_model(str(tmp_path), format="onnx",
                      export_parameters={"onnx_domain": "ai.catboost", "onnx_graph_name": cbm_path.stem})
    m = onnx.load(str(tmp_path))
    for output in m.graph.output:
        shape = output.type.tensor_type.shape
        shape.ClearField("dim")
        d0 = shape.dim.add(); d0.dim_param = "N"
        d1 = shape.dim.add(); d1.dim_value = 1
    tmp_path.unlink()
    return m


# Zenodo's deposited models/rep01/{1k,10k,100k,1m,10m}/ (no "models" subfolder, no
# stage2_metrics.csv alongside) uses these shorthand size labels instead of "n<N>".
SIZE_LABELS = {1000: "1k", 10000: "10k", 100000: "100k", 1000000: "1m", 10000000: "10m"}


def _resolve_run(production_root: Path, rep: str, n: int, metrics_csv: Path | None):
    """Returns (models_dir, best_iter). Supports two layouts: a self-trained run
    (production_root/rep/n<N>/{models/*.cbm, stage2_metrics.csv}) or the Zenodo deposit
    (production_root/rep/<size-label>/*.cbm, best_iter from a separate per_model_metrics.csv,
    config=Maug -- pass its path via `metrics_csv` or FVCB_METRICS_CSV)."""
    import csv
    own_dir = production_root / rep / f"n{n}"
    own_metrics = own_dir / "stage2_metrics.csv"
    if own_metrics.is_file():
        best_iter = {row["target"]: int(row["best_iter"])
                     for row in csv.DictReader(open(own_metrics, newline="", encoding="utf-8"))}
        return own_dir / "models", best_iter

    label = SIZE_LABELS.get(n)
    zenodo_dir = production_root / rep / label if label else None
    if zenodo_dir is not None and zenodo_dir.is_dir():
        best_iter: Dict[str, int] = {}
        mcsv = metrics_csv or (production_root.parent / "results" / "per_model_metrics.csv")
        if mcsv.is_file():
            rep_num = rep.replace("rep", "").lstrip("0") or "0"
            for row in csv.DictReader(open(mcsv, newline="", encoding="utf-8")):
                if (row["config"] == "Maug" and row["replicate"].replace("rep", "").lstrip("0") == rep_num
                        and int(row["training_rows"]) == n):
                    best_iter[row["target"]] = int(row["best_iter"])
        else:
            print(f"  warning: no metrics CSV at {mcsv} -- exporting without shrink (all trees kept)")
        return zenodo_dir, best_iter

    raise FileNotFoundError(f"No models for {rep} n={n} under {production_root} "
                             f"(tried {own_dir} and {zenodo_dir})")


def build_fused(rep: str, n: int, production_root: Path, output_path: Path,
                 metrics_csv: Path | None = None) -> None:
    models_dir, best_iter = _resolve_run(production_root, rep, n, metrics_csv)
    print(f"best_iter per target: {best_iter}")

    print(f"=== building fused ONNX: {models_dir} ===")
    graphs: Dict[str, onnx.ModelProto] = {}
    for target in TARGETS:
        cbm_path = models_dir / f"stage2_{target}_model.cbm"
        if not cbm_path.is_file():
            raise FileNotFoundError(f"Missing model: {cbm_path}")
        graphs[target] = cbm_to_onnx_graph(cbm_path, best_iter.get(target))
        print(f"  loaded {target} -> {cbm_path.name}")

    input_name = "features"
    nodes: List[onnx.NodeProto] = []
    initializers: List[onnx.TensorProto] = []
    value_infos: List[onnx.ValueInfoProto] = []
    output_names: List[str] = []

    for target in TARGETS:
        out_name = f"pred_{target}"
        s_nodes, s_inits, s_vis = _inline_model(graphs[target], f"{target}/", input_name, out_name)
        nodes.extend(s_nodes)
        initializers.extend(s_inits)
        value_infos.extend(s_vis)
        output_names.append(out_name)

    graph_input = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, ["N", len(FEATURES)])
    graph_outputs = [helper.make_tensor_value_info(o, TensorProto.FLOAT, ["N", 1]) for o in output_names]

    graph = helper.make_graph(
        nodes=nodes, name="v3_Maug_FusedPipeline",
        inputs=[graph_input], outputs=graph_outputs,
        initializer=initializers, value_info=value_infos,
    )
    opsets = _merge_opsets(list(graphs.values()))
    model = helper.make_model(graph, producer_name="v3-Maug-fused-builder", opset_imports=opsets)
    model.ir_version = 10
    # Can exceed protobuf's 2GB single-message limit even after shrinking; save with large
    # tensors external instead of inlined. onnx.checker would still hit the 2GB limit (it
    # serializes the whole model in-memory), so skip it -- the sanity check below suffices.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(
        model, str(output_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=output_path.name + ".data",
        size_threshold=1024,
    )
    # save_as_external_data only externalizes tensors >= size_threshold bytes; CatBoost's
    # TreeEnsembleRegressor stores trees as node attributes, not initializer tensors, so in
    # practice no tensor ever crosses that threshold and no .data file is written. Report
    # what actually happened rather than assuming the sidecar was created.
    data_path = output_path.parent / (output_path.name + ".data")
    suffix = f" (+ {data_path.name})" if data_path.exists() else " (no external .data sidecar)"
    print(f"saved: {output_path}{suffix}")

    # sanity check: run a zero-row batch through onnxruntime and confirm 4 outputs of width 1
    sess = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    x = np.zeros((3, len(FEATURES)), dtype=np.float32)
    out = sess.run(None, {input_name: x})
    for name, arr in zip(output_names, out):
        assert arr.shape == (3, 1), f"bad output shape for {name}: {arr.shape}"
    print(f"onnxruntime sanity check OK: {len(out)} outputs, shape {out[0].shape}")

    feat_path = output_path.with_suffix(".features.txt")
    feat_path.write_text("\n".join(FEATURES) + "\n", encoding="utf-8")
    print(f"feature order written: {feat_path}")


if __name__ == "__main__":
    args = parse_args()
    output = args.output or (Path(__file__).resolve().parents[1] / "onnx" / f"Maug_{args.rep}_n{args.n}_fused.onnx")
    build_fused(args.rep, args.n, args.production_root, output, metrics_csv=args.metrics_csv)
