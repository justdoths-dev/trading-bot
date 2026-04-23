from src.research._compat import reexport_module, run_module_main

_TARGET_MODULE = (
    "src.research.diagnostics.selected_strategy_rule_engine_confidence_"
    "compressed_low_band_origin_diagnosis_report"
)

reexport_module(globals(), _TARGET_MODULE)

if __name__ == "__main__":
    run_module_main(_TARGET_MODULE)