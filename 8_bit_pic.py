from dataclasses import dataclass
import logging
from typing import (
    Generator,
    Sequence,
)

import vm_types
import virtual_machine

logger = logging.getLogger(__name__)

WORD_SIZE = 8


class InstructionCodes(vm_types.GenericInstructionSet):
    NOP = 0x00


TEST_PROG = """
; Will @zayfod (Kaloyan) implement PIC 8-bit ?! 
; *eyes*

STATUS  equ 03h
TRISA   equ 85h
PORTA   equ 05h
        bsf STATUS,5
        movlw 00h
        movwf TRISA
        bcf STATUS,5
Start   movlw 02h
        movwf PORTA
        movlw 00h
        movwf PORTA
        goto Star
"""


class Assembler:
    def __init__(
        self,
        program_text,
        instruction_codes: type[vm_types.GenericInstructionSet],
        word_size,
    ):
        self.text = program_text
        self.byte_code = bytearray()

    def load_program(self, program_text):
        self.text = program_text

    def tokenize(self) -> Sequence[vm_types.AssemblerToken]:
        return []

    def assemble(self):
        return self.byte_code

    def link(self):
        return self.byte_code

    def compile(self):
        return self.byte_code


@dataclass
class CPUFlags: ...


@dataclass
class CPURegisters: ...


@dataclass
class CentralProcessingUnit:
    @classmethod
    def register_instruction(cls, inst_code) -> vm_types.DecoratorCallable:
        def decorator(f):
            return f

        return decorator

    @property
    def word_in_bytes(self) -> int:
        return 1

    @property
    def nible_in_bytes(self) -> int:
        raise NotImplementedError

    def fetch(self):
        raise NotImplementedError

    def reset(self):
        return

    def exec(self):
        raise NotImplementedError

    def run(self) -> Generator:
        for _ in range(100):
            yield
            raise NotImplementedError("Kaloyan has not yet implemented 8-bit PIC")

    def dump_registers(self):
        return {}

    @property
    def current_inst_address(self) -> int:
        return 0x00

    @property
    def current_stack_address(self) -> int:
        return 0x00

    @current_stack_address.setter
    def current_stack_address(self, address: int) -> int:
        return 0x00


@CentralProcessingUnit.register_instruction(InstructionCodes.NOP)
def noop(instance: CentralProcessingUnit):
    pass


def instance_factory() -> vm_types.GenericVirtualMachine:
    assembler_instance = Assembler(TEST_PROG, InstructionCodes, WORD_SIZE)

    memory = bytearray(4 * 1024)

    cpu_instance = CentralProcessingUnit()

    vm_instance = virtual_machine.VirtualMachine(
        memory=memory, cpu=cpu_instance, assembler=assembler_instance
    )

    vm_instance.load_program_at(0, TEST_PROG)
    vm_instance.restart()

    return vm_instance
