from dataclasses import field, dataclass
from enum import Enum
from typing import Callable, ClassVar, MutableMapping, Protocol, Self, Sequence

import log

logger = log.logging.getLogger(__name__)

HALT_INS_CODE = 0xFFFF
WORD_SIZE = 16
BITS_IN_BYTE = 8


def inst_is_full_word(firts_byte: int) -> bool:
    return firts_byte > 0x7F


class InstructionCodes(Enum):
    NOP = 0x00
    LOADA = 0x1
    LOADB = 0x2
    ADD = 0x3
    HALT = HALT_INS_CODE


class AssemblerToken(Protocol):
    def encode(self, assembler_instance: "Assembler") -> bytes: ...


class AssemblerParamToken(Protocol):
    def __init__(self, value: str): ...

    def encode(self, assembler_instance: "Assembler") -> bytes: ...


@dataclass
class InlineParamToken:
    value: str

    def encode(self, assembler_instance: "Assembler"):
        word_size_bytes = assembler_instance.word_size_bytes
        base = 10
        match self.value[:2]:
            case "0x":
                base = 16
            case "0b":
                base = 2

        return int(self.value, base).to_bytes(length=word_size_bytes)


@dataclass
class InstructionToken:
    code: InstructionCodes
    params: Sequence[AssemblerParamToken]

    def encode(self, assembler_instance: "Assembler") -> bytes:
        byte_code = bytearray()

        inst_bytes = self.code.value.to_bytes(
            length=assembler_instance.word_size_bytes,
        )

        byte_code.extend(inst_bytes)
        for param_token in self.params:
            byte_code.extend(param_token.encode(assembler_instance=assembler_instance))

        return bytes(byte_code)


@dataclass
class InstructionMeta:
    params: Sequence[type[AssemblerParamToken]] = field(default_factory=list)


GenericOneInlineParamIns = InstructionMeta(params=[InlineParamToken])

INSTRUCTIONS_META = {
    InstructionCodes.NOP: InstructionMeta(),
    InstructionCodes.LOADA: GenericOneInlineParamIns,
    InstructionCodes.LOADB: GenericOneInlineParamIns,
    InstructionCodes.ADD: InstructionMeta(),
    InstructionCodes.HALT: InstructionMeta(),
}


@dataclass
class MacroMeta:
    num_params: int = 0


MACROS_META = {"LABEL": MacroMeta(num_params=1)}

TEST_PROG = """
# 16-bit word tests

LABEL START: NOP
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

# Ex. Carry but no Signed overflow 00x8001 (-32767) + 0x7fff (32767) and zero flag
LOADA 0x8001
LOADB 0x7fff
ADD

HALT
"""


class Assembler:
    def __init__(self, program_text, instruction_codes, word_size):
        self.text = program_text
        self.codes = instruction_codes
        self.word_size = word_size
        self.word_size_bytes = word_size // BITS_IN_BYTE
        self._reset_state()

    def _reset_state(self):
        pass

    def load_program(self, program_text):
        self._reset_state()
        self.text = program_text

    def tokenize(self) -> Sequence[AssemblerToken]:
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

        logger.debug(assembler_tokens)
        return assembler_tokens

    def assemble(self):
        byte_code = bytearray()
        for token in self.tokenize():
            byte_code.extend(token.encode(assembler_instance=self))

        logger.debug(byte_code)
        return byte_code


@dataclass
class CPUFlags:
    overflow: bool = False
    carry: bool = False
    signed: bool = False
    zero: bool = False


@dataclass
class CentralProcessingUnit:
    HALT_INS_CODE: int
    RAM: bytearray
    FLAGS: CPUFlags
    IP: int = 0
    SP: int = 0
    IC: int = 0
    AA: int = 0
    AB: int = 0
    WORD_SIZE_BITS: int = WORD_SIZE
    INSTRUCTION_MAP: ClassVar[
        MutableMapping[InstructionCodes, Callable[[Self], None]]
    ] = {}

    @classmethod
    def register_instruction(cls, inst_code):
        def decorator(f):
            cls.INSTRUCTION_MAP[inst_code] = f
            return f

        return decorator

    @property
    def word_in_bytes(self):
        return self.WORD_SIZE_BITS // BITS_IN_BYTE

    @property
    def nible_in_bytes(self):
        return self.word_in_bytes // 2

    def fetch(self):
        next_ip = self.IP + self.word_in_bytes
        self.IC = int.from_bytes(self.RAM[self.IP : next_ip])
        self.IP = next_ip

    def reset(self):
        """TODO"""

    def exec(self):
        inst_func = self.INSTRUCTION_MAP[InstructionCodes(self.IC)]
        inst_func(self)

    def run(self):
        while not self.fetch() and self.IC != self.HALT_INS_CODE:
            yield
            logger.debug("Running CPU step ...")
            logger.debug(f"{hex(self.IC)=}")
            self.exec()
            logger.debug(f"{InstructionCodes(self.IC)=}, {self.AA=}, {self.AB=}")
            logger.debug(
                f"{InstructionCodes(self.IC)=}, {hex(self.AA)=}, {hex(self.AB)=}"
            )
            logger.debug(f"{self.FLAGS=}")


def load(instance: CentralProcessingUnit, reg_name: str):
    bytes_val = instance.RAM[instance.IP : instance.IP + instance.word_in_bytes]
    val = int.from_bytes(bytes_val)
    setattr(instance, reg_name, val)
    instance.IP += instance.word_in_bytes


@CentralProcessingUnit.register_instruction(InstructionCodes.NOP)
def noop(instance: CentralProcessingUnit):
    pass


@CentralProcessingUnit.register_instruction(InstructionCodes.LOADA)
def load_A(instance: CentralProcessingUnit):
    load(instance, "AA")


@CentralProcessingUnit.register_instruction(InstructionCodes.LOADB)
def load_B(instance: CentralProcessingUnit):
    load(instance, "AB")


@CentralProcessingUnit.register_instruction(InstructionCodes.ADD)
def add_inst(instance: CentralProcessingUnit):
    instance.FLAGS.carry = False
    instance.FLAGS.overflow = False
    instance.FLAGS.signed = False
    instance.FLAGS.zero = False

    sign_bit_mask = 1 << (instance.WORD_SIZE_BITS - 1)
    sign_bit_A = instance.AA & sign_bit_mask
    sign_bit_B = instance.AB & sign_bit_mask

    instance.AA = instance.AA + instance.AB
    try:
        instance.AA.to_bytes(length=instance.word_in_bytes)
    except OverflowError:
        # If result is too big to fit the A register set carry bit
        # Ex. 0xFFFF + 0xFFFF
        instance.FLAGS.carry = True
        instance.AA = instance.AA - (2**instance.WORD_SIZE_BITS)

    sign_bit_result = instance.AA & sign_bit_mask

    # Is MSB of result set, aka. result could be negative if on overflow
    instance.FLAGS.signed = bool(sign_bit_result)

    # Is result exactly zero (ignoring carry and overflow)
    instance.FLAGS.zero = instance.AA == 0

    if sign_bit_A == sign_bit_B and sign_bit_A != sign_bit_result:
        # If the result has incorrect sign for two's compliment numbers set overflow
        # Ex. 0x8000 (-32768) + 0xFFFF (-1)
        # Ex. 0x7fff (32767) + 0x0001 (1)
        instance.FLAGS.overflow = True


@dataclass
class VirtualMachine:
    memory: bytearray
    cpu: CentralProcessingUnit
    assembler: Assembler

    def load_at(self, address: int, data: bytearray, force=False):
        data_len = len(data)
        memory_len = len(self.memory)

        if not force and address % self.cpu.word_in_bytes != 0:
            raise Exception("TODO")

        if address < 0 or (address + data_len - 1) >= memory_len:
            raise Exception("TODO")

        self.memory = (
            self.memory[:address] + data + self.memory[address + data_len - 1 :]
        )

        self.cpu.RAM = self.memory

        logger.debug(
            f"Loaded data at {hex(address)=}\n" f"data=({self.memory[:102].hex()}...)"
        )

    def load_program_at(self, address: int, program_text: str):
        self.assembler.load_program(program_text)
        logger.debug(f"Loaded {program_text=}")
        byte_code = self.assembler.assemble()
        self.load_at(address, byte_code)

    def run(self):
        for _ in self.cpu.run():
            logger.debug("Virtual machine ran step...\n\n")


assembler_instance = Assembler(TEST_PROG, InstructionCodes, WORD_SIZE)

# byte_code = assembler.assemble()
# memory = byte_code + bytearray((4 * 1024) - len(byte_code))
memory = bytearray(4 * 1024)
# memory[0] = 0x01
# memory[1] = 0xFF
# memory[2] = 0xFF
# memory[3] = 0x02
# memory[4] = 0x00
# memory[5] = 0x02
# memory[6] = 0x03
# memory[10] = HALT_INS_CODE >> 8
# memory[11] = HALT_INS_CODE - ((HALT_INS_CODE >> 8) << 8)


cpu_instance = CentralProcessingUnit(
    HALT_INS_CODE=HALT_INS_CODE, RAM=memory, FLAGS=CPUFlags(), SP=len(memory) - 1
)

vm_instance = VirtualMachine(
    memory=memory, cpu=cpu_instance, assembler=assembler_instance
)

vm_instance.load_program_at(12, TEST_PROG)


vm_instance.run()
