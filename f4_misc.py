"""
F-4 MISC 16 bit implementation
instruction	opcode	operand	        operation	            clocks
ADDi imm      00 01	16 bit value	imm+(A) --> A       	3
ADDm addr     00 02	16 bit address	(addr)+(A) --> A    	4
ADDpc         00 04	null operand	PC+(A) --> A        	3
BVS addr      00 08	16 bit address	(addr) --> PC if <v>=1	3
LDAi imm      00 10	16 bit value	imm --> A	            3
LDAm addr     00 20	16 bit address	(addr) --> A	        3
LDApc         00 40	null operand	PC --> A	            3
STAm addr     00 80	16 bit address	A --> (addr)	        3
STApc PC      01 00	null operand	A --> PC	            3

References:
    http://www.dakeng.com/misc.html
"""

from dataclasses import asdict, field, dataclass
import functools
import logging
from typing import (
    Any,
    Callable,
    ClassVar,
    Generator,
    MutableMapping,
    Optional,
    Self,
    Sequence,
    Tuple,
)

import vm_types
import virtual_machine
from toy_assembler import Assembler

logger = logging.getLogger(__name__)

HALT_INS_CODE = 0xFFFF
WORD_SIZE = 16


class InstructionCodes(vm_types.GenericInstructionSet):
    ADDi = 1
    ADDm = 2
    ADDpc = 4
    BVS = 8
    LDAi = 10
    LDAm = 20
    LDApc = 40
    STAm = 80
    STApc = 100
    HALT = HALT_INS_CODE


@dataclass
class InlineParamToken:
    value: str

    def encode(
        self, assembler_instance: vm_types.GenericAssembler, offset: int
    ) -> bytes | bytearray:
        word_size_bytes = assembler_instance.word_size_bytes
        signed = False
        match self.value[:2]:
            case "0x":
                base = 16
            case "0b":
                base = 2
            case _:
                base = 10
                if self.value.strip().startswith("-"):
                    signed = True

        return int(self.value, base).to_bytes(signed=signed, length=word_size_bytes)


@dataclass
class LabelParamToken:
    value: str

    def encode(self, assembler_instance: vm_types.GenericAssembler, offset: int):
        word_size_bytes = assembler_instance.word_size_bytes
        assembler_instance.symbol_table["refs"].setdefault(self.value, []).append(
            len(assembler_instance.byte_code) + offset
        )

        return bytes(word_size_bytes)


@dataclass
class LabelOrInlineParamToken:
    value: str

    def encode(self, assembler_instance: vm_types.GenericAssembler, offset: int):
        if self.value.startswith("."):
            return LabelParamToken(value=self.value)
        else:
            return InlineParamToken(value=self.value)


class MultiInlineParamToken(InlineParamToken):
    value_list: Sequence[str]
    value: str

    def encode(self, assembler_instance: vm_types.GenericAssembler, offset: int):
        byte_code = bytearray()
        self.value_list = self.value.split(",")
        for str_val in self.value_list:
            self.value = str_val.strip()
            byte_val = super().encode(assembler_instance, offset)
            byte_code.extend(byte_val)

        return byte_code


@dataclass
class InstructionToken:
    code: InstructionCodes
    params: Sequence[vm_types.AssemblerParamToken]

    def encode(self, assembler_instance: vm_types.GenericAssembler) -> bytes:
        byte_code = bytearray()

        inst_bytes = self.code.value.to_bytes(
            length=assembler_instance.word_size_bytes,
        )

        byte_code.extend(inst_bytes)
        for param_token in self.params:
            bytes_or_token = param_token
            while not (
                isinstance(bytes_or_token, bytes)
                or isinstance(bytes_or_token, bytearray)
            ):
                # TODO: ADD MAX DEPTH ERROR
                bytes_or_token = bytes_or_token.encode(
                    assembler_instance=assembler_instance, offset=len(byte_code)
                )

            byte_code.extend(bytes_or_token)

        return bytes(byte_code)


@dataclass
class InstructionMeta:
    ins: type[vm_types.AssemblerToken] = InstructionToken
    params: Sequence[type[vm_types.AssemblerParamToken]] = field(default_factory=list)


GenericOneLabelOrInlineParamIns = InstructionMeta(params=[LabelOrInlineParamToken])


class InstructionsMeta:
    def __getitem__(self, key):
        try:
            inst = InstructionCodes(key)
            if inst in (
                InstructionCodes.HALT,
                InstructionCodes.LDApc,
                InstructionCodes.STApc,
            ):
                return InstructionMeta()
            return GenericOneLabelOrInlineParamIns
        except ValueError:
            raise ValueError(f"Unrecognized instruction with mnemonic {key}")


@dataclass
class MacroMeta:
    params: Sequence[type[vm_types.AssemblerParamToken]] = field(default_factory=list)


@dataclass
class SetLabelParamToken:
    value: str

    def encode(self, assembler_instance: vm_types.GenericAssembler):
        pass


@dataclass
class SetLabelToken:
    params: Sequence[vm_types.AssemblerParamToken]

    def encode(self, assembler_instance: vm_types.GenericAssembler) -> None:
        assembler_instance.symbol_table["map"][self.params[0].value] = len(
            assembler_instance.byte_code
        )


@dataclass
class DefineWordToken:
    params: Sequence[vm_types.AssemblerParamToken]

    def encode(self, assembler_instance: vm_types.GenericAssembler) -> bytes:
        byte_code = bytearray()

        for param_token in self.params:
            bytes_or_token = param_token
            while not (
                isinstance(bytes_or_token, bytes)
                or isinstance(bytes_or_token, bytearray)
            ):
                # TODO: ADD MAX DEPTH ERROR
                bytes_or_token = bytes_or_token.encode(
                    assembler_instance=assembler_instance, offset=len(byte_code)
                )

            byte_code.extend(bytes_or_token)

        return bytes(byte_code)


MACROS_META = {
    "LABEL": (SetLabelToken, MacroMeta(params=[SetLabelParamToken])),
    "DWORD": (DefineWordToken, MacroMeta(params=[MultiInlineParamToken])),
}

TEST_PROG = """
# F-4 16-bit word tests
# Code examples from source listed in module comments
LABEL   .START:
        LDAi .FIRST_JUMP
        STApc

LABEL   .NUMBER:
DWORD   0b1000_0000_0000_0001,0xFFFF

LABEL   .FIRST_JUMP:
# SHL (arithmetic shift left) is a quick addition of a number to itself.
# The overflow flag is set if the bit shifted out is 1, clear if it's 0.
        LDAm .NUMBER
        ADDm .NUMBER
        STAm .NUMBER

# Branch overflow on first add, exit second
        LDAi 0xFFFF
LABEL   .BRANCH_OV:
        ADDi 1
        BVS .BRANCH_OV

        HALT
"""


@dataclass
class AddressModes:
    IMMEDIATE = 0
    DIRECT = 1
    REGISTER = 2


@dataclass
class CPUFlags:
    overflow: bool = False


@dataclass
class CPURegisters:
    PC: int = 0
    A: int = 0
    IC: int = 0
    HIDDEN: int = 0


@dataclass
class CentralProcessingUnit:
    HALT_INS_CODE: int
    RAM: bytearray
    FLAGS: CPUFlags
    REGISTERS: CPURegisters = field(default_factory=CPURegisters)
    WORD_SIZE_BITS: int = WORD_SIZE
    INSTRUCTION_MAP: ClassVar[
        MutableMapping[InstructionCodes, Callable[[Self], None]]
    ] = {}

    @classmethod
    def register_instruction(cls, inst_code) -> vm_types.DecoratorCallable:
        def decorator(f) -> Callable:
            cls.INSTRUCTION_MAP[inst_code] = f
            return f

        return decorator

    @property
    def word_in_bytes(self) -> int:
        return self.WORD_SIZE_BITS // vm_types.BITS_IN_BYTE

    @property
    def nible_in_bytes(self) -> int:
        return self.word_in_bytes // 2

    def fetch(self):
        next_ip = self.REGISTERS.PC + self.word_in_bytes
        self.REGISTERS.IC = int.from_bytes(self.RAM[self.REGISTERS.PC : next_ip])
        self.REGISTERS.PC = next_ip

    def reset(self):
        self.REGISTERS = CPURegisters()
        self.FLAGS = CPUFlags()

    def exec(self):
        inst_code = InstructionCodes(self.REGISTERS.IC)
        logger.debug(f"Instruction: {inst_code}")
        inst_func = self.INSTRUCTION_MAP[inst_code]
        inst_func(self)

    def run(self) -> Generator:
        while not self.fetch() and self.REGISTERS.IC != self.HALT_INS_CODE:
            yield
            logger.debug("Running CPU step ...")
            self.exec()
            logger.debug(f"{self.REGISTERS=}")
            logger.debug(f"{self.FLAGS=}")

    def dump_registers(self):
        return asdict(self.REGISTERS)

    @property
    def current_inst_address(self) -> int:
        return self.REGISTERS.PC

    @property
    def current_stack_address(self) -> int:
        return len(self.RAM)

    @current_stack_address.setter
    def current_stack_address(self, address: int) -> int:
        return self.current_stack_address

    @classmethod
    def bind_to_registers(
        cls, bind_list: Sequence[Tuple[InstructionCodes, dict[str, Any]]]
    ) -> vm_types.DecoratorCallable:
        def decorator(f: Callable) -> Callable:
            for ins_code, kwargs in bind_list:
                bound_func = functools.partial(f, **kwargs)
                cls.register_instruction(ins_code)(bound_func)
            return f

        return decorator


def load(
    instance: CentralProcessingUnit,
    reg_name: str,
    *,
    source_reg_name: Optional[str] = None,
    ip_unmodified=False,
    address_mode=AddressModes.IMMEDIATE,
):
    bytes_val = instance.RAM[
        instance.REGISTERS.PC : instance.REGISTERS.PC + instance.word_in_bytes
    ]
    val = int.from_bytes(bytes_val)

    match address_mode:
        case AddressModes.IMMEDIATE:
            pass
        case AddressModes.DIRECT:
            address = val
            val = int.from_bytes(
                instance.RAM[address : address + instance.word_in_bytes]
            )
        case AddressModes.REGISTER:
            if not source_reg_name:
                raise ValueError("Missing source register name")
            val = getattr(instance.REGISTERS, source_reg_name)
            ip_unmodified = True

        case _:
            raise ValueError(f"Unknown {address_mode=}")

    setattr(instance.REGISTERS, reg_name, val)
    if not ip_unmodified:
        instance.REGISTERS.PC += instance.word_in_bytes


def store(
    instance: CentralProcessingUnit,
    source_reg_name: str,
    *,
    dest_reg_name: Optional[str] = None,
    address_mode=AddressModes.IMMEDIATE,
):
    val = getattr(instance.REGISTERS, source_reg_name)

    match address_mode:
        case AddressModes.DIRECT:
            val = val.to_bytes(length=instance.word_in_bytes)
            bytes_val = instance.RAM[
                instance.REGISTERS.PC : instance.REGISTERS.PC + instance.word_in_bytes
            ]
            address = int.from_bytes(bytes_val)

            for idx in range(instance.word_in_bytes):
                instance.RAM[address + idx] = val[idx]

            instance.REGISTERS.PC += instance.word_in_bytes
        case AddressModes.REGISTER:
            if not dest_reg_name:
                raise ValueError("Missing destination register name")
            setattr(instance.REGISTERS, dest_reg_name, val)

        case _:
            raise ValueError(f"Unknown {address_mode=}")


def add(
    instance: CentralProcessingUnit,
    source_reg_name: Optional[str] = None,
    *,
    address_mode=AddressModes.IMMEDIATE,
):
    instance.FLAGS.overflow = False
    load(instance, "HIDDEN", source_reg_name=source_reg_name, address_mode=address_mode)
    val = instance.REGISTERS.HIDDEN + instance.REGISTERS.A
    instance.REGISTERS.A = val & 0xFFFF
    instance.FLAGS.overflow = bool((val & 0x010000))


@CentralProcessingUnit.register_instruction(InstructionCodes.LDAi)
def loada_i(instance: CentralProcessingUnit):
    load(instance, "A")


@CentralProcessingUnit.register_instruction(InstructionCodes.LDAm)
def loada_m(instance: CentralProcessingUnit):
    load(instance, "A", address_mode=AddressModes.DIRECT)


@CentralProcessingUnit.register_instruction(InstructionCodes.LDApc)
def loada_pc(instance: CentralProcessingUnit):
    load(instance, "A", source_reg_name="PC", address_mode=AddressModes.REGISTER)


@CentralProcessingUnit.register_instruction(InstructionCodes.STAm)
def staa_m(instance: CentralProcessingUnit):
    store(instance, source_reg_name="A", address_mode=AddressModes.DIRECT)


@CentralProcessingUnit.register_instruction(InstructionCodes.STApc)
def staa_pc(instance: CentralProcessingUnit):
    store(
        instance,
        source_reg_name="A",
        dest_reg_name="PC",
        address_mode=AddressModes.REGISTER,
    )


@CentralProcessingUnit.register_instruction(InstructionCodes.BVS)
def branch_if_overflow(instance: CentralProcessingUnit):
    if instance.FLAGS.overflow:
        load(instance, "PC", ip_unmodified=True)
    else:
        instance.REGISTERS.PC += instance.word_in_bytes


@CentralProcessingUnit.register_instruction(InstructionCodes.ADDi)
def add_inst(instance: CentralProcessingUnit):
    add(instance, address_mode=AddressModes.IMMEDIATE)


@CentralProcessingUnit.register_instruction(InstructionCodes.ADDm)
def add_m(instance: CentralProcessingUnit):
    add(instance, address_mode=AddressModes.DIRECT)


@CentralProcessingUnit.register_instruction(InstructionCodes.ADDpc)
def add_pc(instance: CentralProcessingUnit):
    add(instance, source_reg_name="PC", address_mode=AddressModes.DIRECT)


def instance_factory() -> vm_types.GenericVirtualMachine:
    assembler_instance = Assembler(
        TEST_PROG, InstructionCodes, WORD_SIZE, InstructionsMeta(), MACROS_META
    )

    memory = bytearray(4 * 1024)

    cpu_instance = CentralProcessingUnit(
        HALT_INS_CODE=HALT_INS_CODE,
        RAM=memory,
        FLAGS=CPUFlags(),
    )

    vm_instance = virtual_machine.VirtualMachine(
        memory=memory, cpu=cpu_instance, assembler=assembler_instance
    )

    vm_instance.load_program_at(0, TEST_PROG)
    vm_instance.restart()

    return vm_instance


if __name__ == "__main__":
    import log  # noqa

    memory = bytearray(4 * 1024)

    vm_instance = instance_factory()

    vm_instance.run()
