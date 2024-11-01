"""

Microchip PICmicro MCU implementation.

Specifics:
- Harvard architecture - separate program and data memories
- 21-bit program address - up to 2MB program memory (flash)
- 12 to 16-bit instruction bus
- big-endian instruction words
- 12-bit data address - up to 4kB data memory (RAM)
- 8-bit data bus

Regerences:
- https://ww1.microchip.com/downloads/en/DeviceDoc/31029a.pdf
- https://developerhelp.microchip.com/xwiki/bin/view/products/mcu-mpu/8bit-pic/8-bitbl/8bitblis/
- https://infonics.wordpress.com/wp-content/uploads/2015/03/introduction-to-pic-18-microcontrollers1.pdf
- https://en.wikipedia.org/wiki/PIC_instruction_listings#Mid-range_core_devices_(14_bit)
- https://ww1.microchip.com/downloads/en/DeviceDoc/MPLAB%20XC8%20PIC%20Assembler%20User%27s%20Guide%2050002974A.pdf

"""

from dataclasses import dataclass
import logging
import re
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

WORD_SIZE = 8
FLASH_SIZE = 2 * 1024 * 1024
RAM_SIZE = 4 * 1024


class InstructionCodes(vm_types.GenericInstructionSet):
    NOP = 0x0000
    MOVWF = 0x0080
    BCF = 0x1000
    BSF = 0x1400
    GOTO = 0x2800
    MOVLW = 0x3000


class LabelToken(vm_types.AssemblerToken):
    """Label token."""

    def __init__(self, label: str) -> None:
        super().__init__()
        self.label = label

    def __repr__(self) -> str:
        return f"Label('{self.label}')"

    def encode(self, assembler: vm_types.GenericAssembler) -> bytes:
        assembler.symbol_table["map"][self.label] = len(assembler.byte_code)
        return b""


class ByteOpToken(vm_types.AssemblerToken):
    """Byte-oriented file register operation token."""

    def __init__(self, inst: str, op: int, f: str, d: str) -> None:
        super().__init__()
        self.inst = str(inst)
        self.op = int(op)
        self.f = str(f)
        self.d = str(d)

    def __repr__(self) -> str:
        return f"{self.inst}(f={self.f}, d={self.d})"

    def encode(self, assembler: vm_types.GenericAssembler) -> bytes:
        f = resolve_int_value(assembler, self.f, 0x7F)
        d = resolve_int_value(assembler, self.d, 0x1)
        code = self.op | (d << 7) | f
        return code.to_bytes(2, "big")


class NOPToken(ByteOpToken):
    """No operation token."""

    def __init__(self) -> None:
        super().__init__("NOP", 0x0000, "0", "0")

    def __repr__(self) -> str:
        return f"{self.inst}()"


class MOVWFToken(ByteOpToken):
    """Move W to f token."""

    def __init__(self, f: str) -> None:
        super().__init__("MOVWF", 0x0080, f, "1")

    def __repr__(self) -> str:
        return f"{self.inst}(f={self.f})"


class BitOpToken(vm_types.AssemblerToken):
    """Bit-oriented file register operation token."""

    def __init__(self, inst: str, op: int, f: str, b: str) -> None:
        super().__init__()
        self.inst = str(inst)
        self.op = int(op)
        self.f = str(f)
        self.b = str(b)

    def __repr__(self) -> str:
        return f"{self.inst}(f={self.f}, b={self.b})"

    def encode(self, assembler: vm_types.GenericAssembler) -> bytes:
        f = resolve_int_value(assembler, self.f, 0x7F)
        b = resolve_int_value(assembler, self.b, 0x7)
        code = self.op | (b << 7) | f
        return code.to_bytes(2, "big")


class BSFToken(BitOpToken):
    """Bit set f token."""

    def __init__(self, f: str, b: str) -> None:
        super().__init__("BSF", 0x1400, f, b)


class BCFToken(BitOpToken):
    """Bit clear f token."""

    def __init__(self, f: str, b: str) -> None:
        super().__init__("BCF", 0x1000, f, b)


class LiteralOpToken(vm_types.AssemblerToken):
    """Literal or control operation token."""

    def __init__(self, inst: str, op: int, k: str, mask: int) -> None:
        super().__init__()
        self.inst = str(inst)
        self.op = int(op)
        self.k = str(k)
        self.mask = int(mask)

    def __repr__(self) -> str:
        return f"{self.inst}(k={self.k})"

    def encode(self, assembler: vm_types.GenericAssembler) -> bytes:
        k = resolve_int_value(assembler, self.k, self.mask)
        code = self.op | k
        return code.to_bytes(2, "big")


class GOTOToken(LiteralOpToken):
    """Goto address token."""

    def __init__(self, k: str) -> None:
        super().__init__("GOTO", 0x2800, k, 0x7FF)


class MOVLWToken(LiteralOpToken):
    """Move literal to W token."""

    def __init__(self, k: str) -> None:
        super().__init__("MOVLW", 0x3000, k, 0xFF)


TEST_PROG = """
STATUS  equ 03h
TRISA   equ 85h
PORTA   equ 05h

        nop
        bsf STATUS,5
        movlw 00h
        movwf TRISA
        bcf STATUS,5

Start:  movlw 02h
        movwf PORTA
        movlw 00h
        movwf PORTA
        goto Start
"""


class PICAssembler(vm_types.GenericAssembler):
    PSEUDOOP_RE = re.compile(
        r"(?P<name>[A-Za-z]\w*)\s+(?P<pseudoop>\w+)\s+(?P<operands>\S+)(\s+?P<comment>;.*)?"
    )
    OPCODE_RE = re.compile(
        r"((?P<label>[A-Za-z]\w*):\s+)?(?P<opcode>\w+)(\s+(?P<operands>\S+))?(\s+?P<comment>;.*)?"
    )

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
        self.symbol_table = {"map": {}, "symbols": {}}
        self.byte_code = bytearray()

    def load_program(self, program_text):
        self.text = program_text

    def _process_pseudoop(
        self, name: str, pseudoop: str, operands: list[str], line_num: int
    ) -> None:
        match pseudoop:
            case "EQU":
                # Defines symbol value.
                if name in self.symbol_table["symbols"]:
                    raise Exception(
                        f"Symbol '{pseudoop}' redefinition on line {line_num}."
                    )
                self.symbol_table["symbols"][name] = operands[0]
            case "SET":
                # Defines or re-defines symbol value.
                self.symbol_table["symbols"][name] = operands[0]
            case "ERROR":
                # Generates a user-defined error.
                raise Exception(f"Error {operands[0]}.")
            case _:
                raise Exception(f"Unexpected pseudoop '{pseudoop}' on line {line_num}.")

    def _process_opcode(
        self,
        label: str,
        opcode: str,
        operands: list[str],
        line_num: int,
        tokens: list[vm_types.AssemblerToken],
    ) -> None:
        if label:
            tokens.append(LabelToken(label))
        match opcode:
            case "MOVWF":
                tokens.append(MOVWFToken(operands[0]))
            case "NOP":
                tokens.append(NOPToken())
            case "BCF":
                tokens.append(BCFToken(operands[0], operands[1]))
            case "BSF":
                tokens.append(BSFToken(operands[0], operands[1]))
            case "GOTO":
                tokens.append(GOTOToken(operands[0]))
            case "MOVLW":
                tokens.append(MOVLWToken(operands[0]))
            case _:
                raise Exception(f"Unexpected opcode '{opcode}' on line {line_num}.")

    def tokenize(self) -> Sequence[vm_types.AssemblerToken]:
        tokens: list[vm_types.AssemblerToken] = []

        lines = self.text.split("\n")
        for i, line in enumerate(lines):
            line_num = i + 1
            line = line.strip()
            if not line:
                # Skip empty lines.
                pass
            elif line[0] == ";":
                # Skip comment lines.
                pass
            elif m := self.PSEUDOOP_RE.match(line):
                # Pseudo-op
                pseudoop = m.group("pseudoop").upper()
                operands = m.group("operands").split(",")
                self._process_pseudoop(m.group("name"), pseudoop, operands, line_num)
            elif m := self.OPCODE_RE.match(line):
                # Opcode
                opcode = m.group("opcode").upper()
                operands = m.group("operands") or ""
                operands = operands.split(",")
                self._process_opcode(
                    m.group("label"), opcode, operands, line_num, tokens
                )
            else:
                raise Exception(f"Syntax error on line {line_num}.")

        logger.debug(tokens)

        return tokens

    def assemble(self):
        self.byte_code = bytearray()

        for token in self.tokenize():
            if code_bytes := token.encode(self):
                self.byte_code.extend(code_bytes)

        logger.debug(self.byte_code)
        logger.debug(self.symbol_table)

        return self.byte_code

    def link(self):
        return self.byte_code

    def compile(self):
        self.assemble()
        self.link()
        return self.byte_code


def resolve_int_value(
    assembler: vm_types.GenericAssembler, value: str, mask: int
) -> int:
    """Resolve integer referene"""
    if value in assembler.symbol_table["symbols"]:
        # Symbol reference.
        value = assembler.symbol_table["symbols"][value]
    elif value in assembler.symbol_table["map"]:
        # Address reference.
        value = assembler.symbol_table["map"][value]
    else:
        # Literal.
        pass
    try:
        int_value = int(value)
    except ValueError:
        # Try to handle hexadecimal literals with "h" suffix.
        int_value = int(value.lower().rstrip("h"), 16)
    int_value = int(int_value) & mask
    return int_value


@dataclass
class PICCentralProcessingUnit(vm_types.GenericCentralProcessingUnit):
    RAM: bytearray
    INSTRUCTION_MAP: ClassVar[
        MutableMapping[InstructionCodes, Callable[[Self, int], None]]
    ] = {}
    ic: int = 0
    pc: int = 0
    wreg: int = 0

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
        self.ic = int.from_bytes(self.RAM[self.pc : next_ip], "big")
        self.pc = next_ip

    def reset(self):
        self.ic = 0
        self.pc = 0
        self.wreg = 0

    def exec(self):
        if self.ic & 0xFF80 == 0:
            ic = self.ic
        elif self.ic & 0xFF80 == 0x0080:
            ic = 0x0080
        elif self.ic & 0xFF00 == 0:
            ic = self.ic & 0xFF00
        elif self.ic & 0xF000 == 0x1000:
            ic = self.ic & 0xFC00
        elif self.ic & 0xF000 == 0x2000:
            ic = self.ic & 0xF800
        elif self.ic & 0xF000 == 0x3000:
            ic = self.ic & 0xFF00
        else:
            raise Exception(f"Unexpected instruction code 0x{self.ic:02x}.")
        ic_key = InstructionCodes(ic)
        logger.debug(f"Executing {ic_key.name} (0x{self.ic:04x}))...")
        inst_func = self.INSTRUCTION_MAP[ic_key]
        inst_func(self, self.ic)

    def run(self) -> Generator:
        while True:
            self.fetch()
            yield
            self.exec()

    def dump_registers(self):
        return {
            "IC": self.ic,
            "PC": self.pc,
            "WREG": self.wreg,
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


@PICCentralProcessingUnit.register_instruction(InstructionCodes.NOP)
def nop(cpu: PICCentralProcessingUnit, ic: int):
    """No operation."""
    pass


@PICCentralProcessingUnit.register_instruction(InstructionCodes.MOVWF)
def movwf(cpu: PICCentralProcessingUnit, ic: int):
    """Move W to f."""
    f = ic & 0x007F
    cpu.RAM[FLASH_SIZE + f] = cpu.wreg
    logger.debug(f"0x{FLASH_SIZE + f:05x}: 0x{cpu.RAM[FLASH_SIZE + f]:02x}")


@PICCentralProcessingUnit.register_instruction(InstructionCodes.BCF)
def bcf(cpu: PICCentralProcessingUnit, ic: int):
    """Bit clear f."""
    b = (ic & 0x0380) >> 7
    f = ic & 0x007F
    cpu.RAM[FLASH_SIZE + f] &= ~(1 << b) & 0xFF
    logger.debug(f"0x{FLASH_SIZE + f:05x}: 0x{cpu.RAM[FLASH_SIZE + f]:02x}")


@PICCentralProcessingUnit.register_instruction(InstructionCodes.BSF)
def bsf(cpu: PICCentralProcessingUnit, ic: int):
    """Bit set f."""
    b = (ic & 0x0380) >> 7
    f = ic & 0x007F
    cpu.RAM[FLASH_SIZE + f] |= 1 << b
    logger.debug(f"0x{FLASH_SIZE + f:05x}: 0x{cpu.RAM[FLASH_SIZE + f]:02x}")


@PICCentralProcessingUnit.register_instruction(InstructionCodes.GOTO)
def goto(cpu: PICCentralProcessingUnit, ic: int):
    """Goto address."""
    k = ic & 0x07FF
    cpu.pc = k


@PICCentralProcessingUnit.register_instruction(InstructionCodes.MOVLW)
def movlw(cpu: PICCentralProcessingUnit, ic: int):
    """Move literal to W."""
    k = ic & 0x00FF
    cpu.wreg = k


def instance_factory() -> vm_types.GenericVirtualMachine:
    memory = bytearray(FLASH_SIZE + RAM_SIZE)
    cpu = PICCentralProcessingUnit(RAM=memory)
    assembler = PICAssembler(TEST_PROG, InstructionCodes, WORD_SIZE, {}, {})
    vm = virtual_machine.VirtualMachine(memory=memory, cpu=cpu, assembler=assembler)
    vm.load_program_at(0, TEST_PROG)
    vm.restart()
    return vm
