import sys
import importlib

import toy_vm
import gui
import log

logger = log.logging.getLogger(__name__)

SWITCH_ARG_PREFIX = "--"

if __name__ == "__main__":
    debug = False
    if len(sys.argv) > 1 and sys.argv[1].startswith(SWITCH_ARG_PREFIX):
        match arg_key := sys.argv.pop(1).split(SWITCH_ARG_PREFIX):
            case _, "debug":
                debug = True

    log.init_logging(debug=debug)
    logger.debug(sys.argv)

    vm_instances = {
        "toy_vm_default": toy_vm.instance_factory(),
    }

    if len(sys.argv) > 1:
        for vm_module_name in sys.argv[1:]:
            module = importlib.import_module(vm_module_name)
            vm_instances[vm_module_name] = module.instance_factory()
            logger.debug(f"Loaded VM instance from {vm_module_name=}")

    gui_instance = gui.VirualMachineGUI(available_vms=vm_instances)
    gui_instance.run()
