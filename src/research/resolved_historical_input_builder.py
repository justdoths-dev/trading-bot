from src.research._compat import reexport_module, run_module_main

_TARGET_MODULE = "src.research.replay.resolved_historical_input_builder"

reexport_module(globals(), _TARGET_MODULE)

if __name__ == "__main__":
    run_module_main(_TARGET_MODULE)
