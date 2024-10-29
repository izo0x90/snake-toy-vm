import gui
import toy_vm
import log

if __name__ == "__main__":
    assembler_instance = toy_vm.Assembler(
        toy_vm.TEST_PROG, toy_vm.InstructionCodes, toy_vm.WORD_SIZE
    )

    memory = bytearray(4 * 1024)

    cpu_instance = toy_vm.CentralProcessingUnit(
        HALT_INS_CODE=toy_vm.HALT_INS_CODE,
        RAM=memory,
        FLAGS=toy_vm.CPUFlags(),
    )

    cpu_instance.REGISTERS.SP = len(memory) - 1

    vm_instance = toy_vm.VirtualMachine(
        memory=memory, cpu=cpu_instance, assembler=assembler_instance, current_run=None
    )

    vm_instance.load_program_at(0, toy_vm.TEST_PROG)
    vm_instance.restart()

    gui_instance = gui.VirualMachineGUI(virtual_machine=vm_instance)
    gui_instance.run()
