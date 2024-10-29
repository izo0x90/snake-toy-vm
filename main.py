import sys
import importlib

import toy_vm
import gui
import log

logger = log.logging.getLogger(__name__)

if __name__ == "__main__":
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
