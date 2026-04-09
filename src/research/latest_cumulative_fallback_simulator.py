from src.research._compat import reexport_module, run_module_main

_TARGET_MODULE = "src.research.diagnostics.latest_cumulative_fallback_simulator"

reexport_module(globals(), _TARGET_MODULE)

if __name__ == "__main__":
    run_module_main(_TARGET_MODULE)
