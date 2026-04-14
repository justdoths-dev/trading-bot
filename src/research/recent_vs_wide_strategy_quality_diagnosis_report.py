from src.research._compat import reexport_module, run_module_main

_TARGET_MODULE = "src.research.diagnostics.recent_vs_wide_strategy_quality_diagnosis_report"

reexport_module(globals(), _TARGET_MODULE)

if __name__ == "__main__":
    run_module_main(_TARGET_MODULE)
