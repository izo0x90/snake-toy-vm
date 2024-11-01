"""

Brainfuck CPU implementation.

Regerences:
- https://en.wikipedia.org/wiki/Brainfuck

"""

from dataclasses import dataclass
import logging
from typing import (
    Callable,
    ClassVar,
    Generator,
    Mapping,
    MutableMapping,
    Self,
    Sequence,
)

import vm_types
import virtual_machine

logger = logging.getLogger(__name__)

WORD_SIZE = 16
CODE_SIZE = 2 * 2**13  # 8192 code words
DATA_SIZE = 128


class InstructionCodes(vm_types.GenericInstructionSet):
    INC_PTR = 0
    DEC_PTR = 1
    INC_DATA = 2
    DEC_DATA = 3
    OUTPUT = 4
    INPUT = 5
    LOOP_START = 6
    LOOP_END = 7


INST_MAP = {
    ">": InstructionCodes.INC_PTR,
    "<": InstructionCodes.DEC_PTR,
    "+": InstructionCodes.INC_DATA,
    "-": InstructionCodes.DEC_DATA,
    ".": InstructionCodes.OUTPUT,
    ",": InstructionCodes.INPUT,
    "[": InstructionCodes.LOOP_START,
    "]": InstructionCodes.LOOP_END,
}


class BrainfuckToken(vm_types.AssemblerToken):
    """Brainfuck token."""

    def __init__(self, inst: str, arg: int) -> None:
        super().__init__()
        self.inst = str(inst)
        self.arg = int(arg)

    def __repr__(self) -> str:
        return f"{self.inst}"

    def encode(self, _) -> bytes:
        code = (INST_MAP[self.inst].value << 13) | (self.arg & 0x1FFF)
        return code.to_bytes(2, "little")


TEST_PROG = """
++++++++                ; Set Cell #0 to 8
[
    >++++               ; Add 4 to Cell #1; this will always set Cell #1 to 4
    [                   ; as the cell will be cleared by the loop
        >++             ; Add 2 to Cell #2
        >+++            ; Add 3 to Cell #3
        >+++            ; Add 3 to Cell #4
        >+              ; Add 1 to Cell #5
        <<<<-           ; Decrement the loop counter in Cell #1
    ]                   ; Loop until Cell #1 is zero; number of iterations is 4
    >+                  ; Add 1 to Cell #2
    >+                  ; Add 1 to Cell #3
    >-                  ; Subtract 1 from Cell #4
    >>+                 ; Add 1 to Cell #6
    [<]                 ; Move back to the first zero cell you find; this will
                        ; be Cell #1 which was cleared by the previous loop
    <-                  ; Decrement the loop Counter in Cell #0
]                       ; Loop until Cell #0 is zero; number of iterations is 8

; The result of this is:
; Cell no :   0   1   2   3   4   5   6
; Contents:   0   0  72 104  88  32   8
; Pointer :   ^

>>.                     ; Cell #2 has value 72 which is 'H'
>---.                   ; Subtract 3 from Cell #3 to get 101 which is 'e'
+++++++..+++.           ; Likewise for 'llo' from Cell #3
>>.                     ; Cell #5 is 32 for the space
<-.                     ; Subtract 1 from Cell #4 for 87 to give a 'W'
<.                      ; Cell #3 was set to 'o' from the end of 'Hello'
+++.------.--------.    ; Cell #3 for 'rl' and 'd'
>>+.                    ; Add 1 to Cell #5 gives us an exclamation point
>++.                    ; And finally a newline from Cell #6
"""


def look_forward(insts: str, i: int) -> int:
    try:
        cnt = 0
        while True:
            if insts[i] == "[":
                cnt += 1
            elif insts[i] == "]":
                cnt -= 1
            if not cnt:
                return i - 1
            i += 1
    except IndexError:
        raise Exception("Unclosed loop.")


class BrainfuckAssembler(vm_types.GenericAssembler):
    def __init__(
        self,
        program_text,
        instruction_codes: type[vm_types.GenericInstructionSet],
        word_size,
        instructions_meta: Mapping[str, Callable],
        macros_meta: Mapping[str, Callable],
    ):
        self.text = program_text
        self.codes = instruction_codes
        self.word_size = word_size
        self.instructions_meta = instructions_meta
        self.macros_meta = macros_meta
        self.word_size_bytes = word_size // vm_types.BITS_IN_BYTE
        self.symbol_table = {}  # Not used.
        self.byte_code = bytearray()

    def load_program(self, program_text):
        self.text = program_text

    def tokenize(self) -> Sequence[vm_types.AssemblerToken]:
        # Extract instructions.
        insts = ""
        lines = self.text.split("\n")
        for i, line in enumerate(lines):
            line_num = i + 1

            line = line.split(";", 1)[0].strip()
            for ch in line:
                if ch in " \t\r\f\v":
                    # Skip whitespace.
                    pass
                elif ch in "><+-.,[]":
                    # Command.
                    insts += ch
                else:
                    raise Exception(f"Syntax error on line {line_num}.")
        logger.debug(insts)

        # Convert instructions to tokens
        loop_indexes = []
        tokens: list[vm_types.AssemblerToken] = []
        for i, inst in enumerate(insts):
            if inst == "[":
                end = look_forward(insts, i)
                tokens.append(BrainfuckToken(inst, end))
                loop_indexes.append(i)
            elif inst == "]":
                try:
                    start = loop_indexes.pop(-1)
                except IndexError:
                    raise Exception("Unexpected loop end.")
                tokens.append(BrainfuckToken(inst, start))
            else:
                tokens.append(BrainfuckToken(inst, 0))
        logger.debug(tokens)

        return tokens

    def assemble(self):
        self.byte_code = bytearray()

        for token in self.tokenize():
            if code_bytes := token.encode(self):
                self.byte_code.extend(code_bytes)

        logger.debug(self.byte_code)

        return self.byte_code

    def link(self):
        return self.byte_code

    def compile(self):
        self.assemble()
        self.link()
        return self.byte_code


@dataclass
class BrainfuckCentralProcessingUnit(vm_types.GenericCentralProcessingUnit):
    RAM: bytearray
    INSTRUCTION_MAP: ClassVar[
        MutableMapping[InstructionCodes, Callable[[Self, int], None]]
    ] = {}
    ic: int = 0  # Instruction code
    pc: int = 0  # Program counter
    dp: int = 0  # Data pointer

    @classmethod
    def register_instruction(cls, inst_code) -> vm_types.DecoratorCallable:
        def decorator(f):
            cls.INSTRUCTION_MAP[inst_code] = f
            return f

        return decorator

    @property
    def word_in_bytes(self) -> int:
        return 2

    @property
    def nible_in_bytes(self) -> int:
        raise NotImplementedError

    def fetch(self):
        next_ip = self.pc + self.word_in_bytes
        self.ic = int.from_bytes(self.RAM[self.pc : next_ip], "little")
        self.pc = next_ip

    def reset(self):
        self.ic = 0
        self.pc = 0
        self.dp = 0

    def exec(self):
        ic = (self.ic & 0xE000) >> 13
        arg = self.ic & 0x1FFF
        ic_key = InstructionCodes(ic)
        logger.debug(f"Executing {ic_key.name}({arg})...")
        inst_func = self.INSTRUCTION_MAP[ic_key]
        inst_func(self, arg)

    def run(self) -> Generator:
        while True:
            self.fetch()
            yield
            self.exec()

    def dump_registers(self):
        return {
            "IC": self.ic,
            "PC": self.pc,
            "DP": self.dp,
            "*DP": self.RAM[CODE_SIZE + self.dp],
        }

    @property
    def current_inst_address(self) -> int:
        return self.pc

    @property
    def current_stack_address(self) -> int:
        return 0

    @current_stack_address.setter
    def current_stack_address(self, address: int) -> int:
        return 0


@BrainfuckCentralProcessingUnit.register_instruction(InstructionCodes.INC_PTR)
def inc_ptr_inst(cpu: BrainfuckCentralProcessingUnit, _):
    """Increment the data pointer by one (to point to the next cell to the right)."""
    cpu.dp = (cpu.dp + 1) % DATA_SIZE


@BrainfuckCentralProcessingUnit.register_instruction(InstructionCodes.DEC_PTR)
def dec_ptr_inst(cpu: BrainfuckCentralProcessingUnit, _):
    """Decrement the data pointer by one (to point to the next cell to the left)."""
    cpu.dp = (cpu.dp - 1) % DATA_SIZE


@BrainfuckCentralProcessingUnit.register_instruction(InstructionCodes.INC_DATA)
def inc_data_inst(cpu: BrainfuckCentralProcessingUnit, _):
    """Increment the byte at the data pointer by one."""
    cpu.RAM[CODE_SIZE + cpu.dp] = (cpu.RAM[CODE_SIZE + cpu.dp] + 1) % 256


@BrainfuckCentralProcessingUnit.register_instruction(InstructionCodes.DEC_DATA)
def dec_data_inst(cpu: BrainfuckCentralProcessingUnit, _):
    """Decrement the byte at the data pointer by one."""
    cpu.RAM[CODE_SIZE + cpu.dp] = (cpu.RAM[CODE_SIZE + cpu.dp] - 1) % 256


@BrainfuckCentralProcessingUnit.register_instruction(InstructionCodes.OUTPUT)
def output_inst(cpu: BrainfuckCentralProcessingUnit, _):
    """Output the byte at the data pointer."""
    logger.info(chr(cpu.RAM[CODE_SIZE + cpu.dp]))


@BrainfuckCentralProcessingUnit.register_instruction(InstructionCodes.INPUT)
def input_inst(cpu: BrainfuckCentralProcessingUnit, _):
    """Accept one byte of input, storing its value in the byte at the data pointer."""
    cpu.RAM[CODE_SIZE + cpu.dp] = 0


@BrainfuckCentralProcessingUnit.register_instruction(InstructionCodes.LOOP_START)
def loop_start_inst(cpu: BrainfuckCentralProcessingUnit, arg: int):
    """If the byte at the data pointer is zero, then instead of moving the instruction pointer forward to the next command, jump it forward to the command after the matching ] command."""
    if not cpu.RAM[CODE_SIZE + cpu.dp]:
        cpu.pc = 2 * arg


@BrainfuckCentralProcessingUnit.register_instruction(InstructionCodes.LOOP_END)
def loop_end_inst(cpu: BrainfuckCentralProcessingUnit, arg: int):
    """If the byte at the data pointer is nonzero, then instead of moving the instruction pointer forward to the next command, jump it back to the command after the matching [ command."""
    if cpu.RAM[CODE_SIZE + cpu.dp]:
        cpu.pc = 2 * arg


def instance_factory() -> vm_types.GenericVirtualMachine:
    memory = bytearray(CODE_SIZE + DATA_SIZE)
    cpu = BrainfuckCentralProcessingUnit(RAM=memory)
    assembler = BrainfuckAssembler(TEST_PROG, InstructionCodes, WORD_SIZE, {}, {})
    vm = virtual_machine.VirtualMachine(memory=memory, cpu=cpu, assembler=assembler)
    vm.load_program_at(0, TEST_PROG)
    vm.restart()
    return vm
