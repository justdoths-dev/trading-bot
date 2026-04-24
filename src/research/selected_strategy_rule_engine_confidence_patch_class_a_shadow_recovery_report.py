from src.research._compat import reexport_module, run_module_main

_TARGET_MODULE = (
    "src.research.diagnostics.selected_strategy_rule_engine_confidence_"
    "patch_class_a_shadow_recovery_report"
)

reexport_module(globals(), _TARGET_MODULE)

if __name__ == "__main__":
    run_module_main(_TARGET_MODULE)
