from dataclasses import asdict, field, dataclass
import functools
import logging
from typing import (
    Any,
    Callable,
    ClassVar,
    Generator,
    MutableMapping,
    Self,
    Sequence,
    Tuple,
)

import vm_types
import virtual_machine

logger = logging.getLogger(__name__)

HALT_INS_CODE = 0xFFFF
WORD_SIZE = 16


class InstructionCodes(vm_types.GenericInstructionSet):
    NOP = 0x00
    LOADA = 0x1
    LOADB = 0x2
    ADD = 0x3
    MLOADA = 0x4
    MSTOREA = 0x5
    MLOADB = 0x6
    LOADIX = 0x7
    JMP = 0x10
    JZ = 0x11
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
    params: Sequence[type[vm_types.AssemblerParamToken]] = field(default_factory=list)


GenericOneInlineParamIns = InstructionMeta(params=[InlineParamToken])
GenericOneLabelOrInlineParamIns = InstructionMeta(params=[LabelOrInlineParamToken])

INSTRUCTIONS_META = {
    InstructionCodes.NOP: InstructionMeta(),
    InstructionCodes.LOADA: GenericOneInlineParamIns,
    InstructionCodes.MLOADA: GenericOneLabelOrInlineParamIns,
    InstructionCodes.LOADB: GenericOneInlineParamIns,
    InstructionCodes.MLOADB: GenericOneLabelOrInlineParamIns,
    InstructionCodes.LOADIX: GenericOneInlineParamIns,
    InstructionCodes.MSTOREA: GenericOneLabelOrInlineParamIns,
    InstructionCodes.ADD: InstructionMeta(),
    InstructionCodes.JMP: GenericOneLabelOrInlineParamIns,
    InstructionCodes.JZ: GenericOneLabelOrInlineParamIns,
    InstructionCodes.HALT: InstructionMeta(),
}


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
# 16-bit word tests

JMP .START
LABEL .DATA
DWORD 0xABAB,0xCDCD

LABEL .START: NOP
# Loop
LOADA 0x000A
LOADB -1

LABEL .LOOP
ADD
JZ .END
JMP .LOOP

# Memory loads
MLOADA .END
MLOADA .DATA
LOADIX 0x0001
MLOADB .DATA
HALT

# No overflow or carry
LOADA 0xFFFD
LOADB 0x0002
ADD

# Signed overflow and carry
LOADA 0x8000
LOADB 0xffff
ADD

# Ex. Signed overflow 0x7fff (32767) + 0x0001 (1) and signed flag
LOADA 0x7fff
LOADB 0x0001
ADD

LABEL .ONE_ADD:
# Ex. Carry but no Signed overflow 00x8001 (-32767) + 0x7fff (32767) and zero flag
LOADA 0x8001
LOADB 0x7fff
ADD

LABEL .END:
HALT
"""


class Assembler:
    def __init__(
        self,
        program_text,
        instruction_codes: type[vm_types.GenericInstructionSet],
        word_size,
    ):
        self.text = program_text
        self.codes = instruction_codes
        self.word_size = word_size
        self.word_size_bytes = word_size // vm_types.BITS_IN_BYTE
        self._reset_state()

    def _reset_state(self):
        self.symbol_table = {"map": {}, "refs": {}}

    def load_program(self, program_text):
        self._reset_state()
        self.text = program_text
        self.byte_code = bytearray()

    def tokenize(self) -> Sequence[vm_types.AssemblerToken]:
        partial_split_lines = self.text.split("\n")
        label_split_code_lines = []
        for line in partial_split_lines:
            label_split_code_lines.extend(line.split(":"))

        code_lines = []
        for line in label_split_code_lines:
            potential_code = line.split("#", 1)[0].strip()
            if potential_code:
                code_lines.append(potential_code)

        tokens = []
        for line in code_lines:
            tokens.extend(line.split(" "))

        assembler_tokens = []
        for token in (tokens := filter(lambda x: x, tokens)):
            if code := getattr(InstructionCodes, token, None):
                meta = INSTRUCTIONS_META[code]
                param_values = []
                for param_type in meta.params:
                    param_values.append(param_type(value=next(tokens)))
                assembler_tokens.append(
                    InstructionToken(code=code, params=param_values)
                )
            elif token in MACROS_META:
                token_class, meta = MACROS_META[token]
                param_values = []
                for param_type in meta.params:
                    next_token = next(tokens)
                    param_values.append(param_type(value=next_token))

                assembler_tokens.append(token_class(params=param_values))

        logger.debug(assembler_tokens)
        return assembler_tokens

    def assemble(self):
        self.byte_code = bytearray()
        for token in self.tokenize():
            if code_bytes := token.encode(assembler_instance=self):
                self.byte_code.extend(code_bytes)

        logger.debug(self.byte_code)
        logger.debug(self.symbol_table)
        return self.byte_code

    def link(self):
        for ref_label, ref_locations in self.symbol_table["refs"].items():
            for location in ref_locations:
                symbol_data = self.symbol_table["map"][ref_label]
                for idx, singe_byte in enumerate(
                    symbol_data.to_bytes(length=self.word_size_bytes)
                ):
                    self.byte_code[location + idx] = singe_byte

        return self.byte_code

    def compile(self):
        self.assemble()
        self.link()
        return self.byte_code


@dataclass
class AddressModes:
    IMMEDIATE = 0
    DIRECT = 1
    DIRECT_INDEXED = 2


@dataclass
class CPUFlags:
    overflow: bool = False
    carry: bool = False
    signed: bool = False
    zero: bool = False


@dataclass
class CPURegisters:
    IP: int = 0
    SP: int = 0
    IC: int = 0
    AA: int = 0
    AB: int = 0
    IX: int = 0


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
        next_ip = self.REGISTERS.IP + self.word_in_bytes
        self.REGISTERS.IC = int.from_bytes(self.RAM[self.REGISTERS.IP : next_ip])
        self.REGISTERS.IP = next_ip

    def reset(self):
        self.REGISTERS = CPURegisters()
        self.FLAGS = CPUFlags()

    def exec(self):
        inst_func = self.INSTRUCTION_MAP[InstructionCodes(self.REGISTERS.IC)]
        inst_func(self)

    def run(self) -> Generator:
        while not self.fetch() and self.REGISTERS.IC != self.HALT_INS_CODE:
            yield
            logger.debug("Running CPU step ...")
            logger.debug(f"{hex(self.REGISTERS.IC)=}")
            self.exec()
            logger.debug(
                f"{InstructionCodes(self.REGISTERS.IC)=}, {self.REGISTERS.AA=}, {self.REGISTERS.AB=}"
            )
            logger.debug(
                f"{InstructionCodes(self.REGISTERS.IC)=}, {hex(self.REGISTERS.AA)=}, {hex(self.REGISTERS.AB)=}"
            )
            logger.debug(f"{self.FLAGS=}")

    def dump_registers(self):
        return asdict(self.REGISTERS)

    @property
    def current_inst_address(self) -> int:
        return self.REGISTERS.IP

    @property
    def current_stack_address(self) -> int:
        return self.REGISTERS.SP

    @current_stack_address.setter
    def current_stack_address(self, address: int) -> int:
        self.REGISTERS.SP = address
        return self.REGISTERS.SP

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
    ip_unmodified=False,
    address_mode=AddressModes.IMMEDIATE,
):
    bytes_val = instance.RAM[
        instance.REGISTERS.IP : instance.REGISTERS.IP + instance.word_in_bytes
    ]
    val = int.from_bytes(bytes_val)

    match address_mode:
        case AddressModes.IMMEDIATE:
            pass
        case AddressModes.DIRECT_INDEXED:
            address = val + instance.REGISTERS.IX
            val = int.from_bytes(
                instance.RAM[address : address + instance.word_in_bytes]
            )

        case _:
            raise ValueError(f"Unknown {address_mode=}")

    setattr(instance.REGISTERS, reg_name, val)
    if not ip_unmodified:
        instance.REGISTERS.IP += instance.word_in_bytes


@CentralProcessingUnit.register_instruction(InstructionCodes.NOP)
def noop(instance: CentralProcessingUnit):
    pass


@CentralProcessingUnit.bind_to_registers(
    [
        (InstructionCodes.LOADA, {"reg_name": "AA"}),
        (InstructionCodes.LOADB, {"reg_name": "AB"}),
        (InstructionCodes.LOADIX, {"reg_name": "IX"}),
    ]
)
def load_immediate(instance: CentralProcessingUnit, reg_name: str):
    load(instance, reg_name)


@CentralProcessingUnit.bind_to_registers(
    [
        (InstructionCodes.MLOADA, {"reg_name": "AA"}),
        (InstructionCodes.MLOADB, {"reg_name": "AB"}),
        # (InstructionCodes.MLOADIX, {"reg_name": "IX"}),
    ]
)
def mload_direct(instance: CentralProcessingUnit, reg_name: str):
    load(instance, reg_name, address_mode=AddressModes.DIRECT_INDEXED)


@CentralProcessingUnit.register_instruction(InstructionCodes.ADD)
def add_inst(instance: CentralProcessingUnit):
    instance.FLAGS.carry = False
    instance.FLAGS.overflow = False
    instance.FLAGS.signed = False
    instance.FLAGS.zero = False

    sign_bit_mask = 1 << (instance.WORD_SIZE_BITS - 1)
    sign_bit_A = instance.REGISTERS.AA & sign_bit_mask
    sign_bit_B = instance.REGISTERS.AB & sign_bit_mask

    instance.REGISTERS.AA = instance.REGISTERS.AA + instance.REGISTERS.AB
    try:
        instance.REGISTERS.AA.to_bytes(length=instance.word_in_bytes)
    except OverflowError:
        # If result is too big to fit the A register set carry bit
        # Ex. 0xFFFF + 0xFFFF
        instance.FLAGS.carry = True
        instance.REGISTERS.AA = instance.REGISTERS.AA - (2**instance.WORD_SIZE_BITS)

    sign_bit_result = instance.REGISTERS.AA & sign_bit_mask

    # Is MSB of result set, aka. result could be negative if on overflow
    instance.FLAGS.signed = bool(sign_bit_result)

    # Is result exactly zero (ignoring carry and overflow)
    instance.FLAGS.zero = instance.REGISTERS.AA == 0

    if sign_bit_A == sign_bit_B and sign_bit_A != sign_bit_result:
        # If the result has incorrect sign for two's compliment numbers set overflow
        # Ex. 0x8000 (-32768) + 0xFFFF (-1)
        # Ex. 0x7fff (32767) + 0x0001 (1)
        instance.FLAGS.overflow = True


@CentralProcessingUnit.register_instruction(InstructionCodes.JMP)
def jump(instance: CentralProcessingUnit):
    load(instance, "IP", ip_unmodified=True)


@CentralProcessingUnit.register_instruction(InstructionCodes.JZ)
def jump_zero(instance: CentralProcessingUnit):
    if instance.FLAGS.zero:
        load(instance, "IP", ip_unmodified=True)
    else:
        instance.REGISTERS.IP += instance.word_in_bytes


def instance_factory() -> vm_types.GenericVirtualMachine:
    assembler_instance = Assembler(TEST_PROG, InstructionCodes, WORD_SIZE)

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

    assembler_instance = Assembler(TEST_PROG, InstructionCodes, WORD_SIZE)

    memory = bytearray(4 * 1024)
    # Manual load test prog. in mem.
    # memory[0] = 0x01
    # memory[1] = 0xFF
    # memory[2] = 0xFF
    # memory[3] = 0x02
    # memory[4] = 0x00
    # memory[5] = 0x02
    # memory[6] = 0x03
    # memory[10] = HALT_INS_CODE >> 8
    # memory[11] = HALT_INS_CODE - ((HALT_INS_CODE >> 8) << 8)

    vm_instance = instance_factory()

    vm_instance.run()
